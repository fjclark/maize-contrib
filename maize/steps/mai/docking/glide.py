"""Schrodinger GLIDE docking interface"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pytest

from maize.core.interface import Input, Output, Parameter, Suffix, FileParameter
from maize.utilities.testing import TestRig
from maize.utilities.utilities import unique_id
from maize.utilities.validation import FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger
from maize.utilities.chem import (
    IsomerCollection,
    Isomer,
    load_sdf_library,
    save_sdf_library,
    merge_libraries,
)
from maize.utilities.io import Config


GlideConfigType = dict[str, str | int | float | bool | Path | list[str]]


def _write_glide_input(path: Path, data: GlideConfigType, constraints: Path | None = None) -> None:
    """Writes a GLIDE ``.in`` input file"""

    cons_lines = []
    cons_section = False
    if constraints is not None:
        with constraints.open("r") as con:
            for line in con.readlines():
                if line.startswith("[") or cons_section:
                    cons_lines.append(line)
                    cons_section = True

    with path.open("w") as file:
        for key, value in data.items():
            match value:
                case Path():
                    file.write(f"{key.upper()}  {value.as_posix()}\n")
                case list():
                    cons = ", ".join(f'"{val}"' for val in value)
                    file.write(f"{key.upper()}  {cons}\n")
                case _:
                    file.write(f"{key.upper()}  {value}\n")
        file.writelines(cons_lines)


class Glide(Schrodinger):
    """
    Calls Schrodinger's GLIDE to dock small molecules.

    Notes
    -----
    Due to Schrodinger's licensing system, each call to a tool requires
    going through Schrodinger's job server. This is run separately for
    each job to avoid conflicts with a potentially running main server.

    See Also
    --------
    :class:`~maize.steps.mai.docking.Vina` :
        A popular open-source docking program
    :class:`~maize.steps.mai.docking.AutoDockGPU` :
        Another popular open-source docking tool with GPU support

    """

    N_LICENSES = 4
    DEFAULT_OUTPUT_NAME = "glide"
    GLIDE_SCORE_TAG = "r_i_docking_score"

    required_callables = ["glide"]

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to dock"""

    inp_grid: Input[Annotated[Path, Suffix("zip")]] = Input()
    """Previously prepared GLIDE grid file"""

    ref_ligand: Input[Isomer] = Input(optional=True)
    """Optional reference ligand"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with poses and energies included"""

    precision: Parameter[Literal["SP", "XP", "HTVS"]] = Parameter(default="SP")
    """GLIDE docking precision"""

    constraints: FileParameter[Annotated[Path, Suffix("in")]] = FileParameter(optional=True)
    """A GLIDE input file containing all desired constraints"""

    keywords: Parameter[GlideConfigType] = Parameter(default_factory=dict)
    """
    Additional GLIDE keywords to use, see the `GLIDE documentation
    <https://www.schrodinger.com/sites/default/files/s3/release/current/Documentation/html/glide/glide_command_reference/glide_command_glide.htm>`_ for details.

    """

    def run(self) -> None:
        mols = self.inp.receive()
        inp_file = Path("input.sdf")
        grid_obj = self.inp_grid.receive()

        config: GlideConfigType = {
            "GRIDFILE": grid_obj.as_posix(),
            "PRECISION": self.precision.value,
            "LIGANDFILE": inp_file,
            "POSE_OUTTYPE": "ligandlib_sd",
            "POSES_PER_LIG": 4,
            "COMPRESS_POSES": False,
            "NOSORT": True,
        }
        config.update(self.keywords.value)

        # Optional reference ligand
        ref = self.ref_ligand.receive_optional()
        if ref:
            ref_path = Path("ref.sdf")
            ref.to_sdf(ref_path)
            self.logger.info("Using reference ligand '%s'", ref.to_smiles())
            config["REF_LIGAND_FILE"] = ref_path
            config["USE_REF_LIGAND"] = True
            config["CORE_RESTRAIN"] = True
            config["CORE_DEFINITION"] = "mcssmarts"

        save_sdf_library(inp_file, mols, split_strategy="schrodinger")
        glide_inp_file = Path("glide.in")

        # We get the constraints from a reference input file and add them
        # to the config, as they don't fall into a simple key-value scheme
        additional_constraints = None
        if self.constraints.is_set:
            additional_constraints = self.constraints.filepath

        _write_glide_input(glide_inp_file, config, constraints=additional_constraints)
        self.logger.debug("Prepared GLIDE input for %s molecules", len(mols))

        # Wait for licenses
        self.logger.info("Waiting for %s licenses...", self.N_LICENSES * self.n_jobs.value)
        key = "GLIDE_XP_DOCKING" if self.precision.value == "XP" else "GLIDE_SP_DOCKING"
        self.guard.wait(key, number=self.N_LICENSES * self.n_jobs.value)

        # Run
        name = f"glide-{unique_id(12)}"
        self.logger.info("Found licenses, docking...")
        output = Path(f"{name}_raw.sdf")
        command = (
            f"{self.runnable['glide']} -HOST {self.host.value} "
            f"-NJOBS {self.n_jobs.value} {glide_inp_file.as_posix()}"
        )
        self._run_schrodinger_job(
            command, name=name, validators=[FileValidator(output)]
        )

        self.logger.info("Parsing output")
        docked = load_sdf_library(output, split_strategy="schrodinger")
        self.logger.debug("Received %s docked molecules", len(docked))
        mols = merge_libraries(mols, docked)
        for mol in mols:
            for iso in mol.molecules:
                iso.score_tag = self.GLIDE_SCORE_TAG
                iso.set_tag("origin", self.name)

                if not iso.has_tag(iso.score_tag):
                    iso.set_tag(iso.score_tag, np.nan)

        self.out.send(mols)


# From IcolosData
@pytest.fixture
def grid(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_grid_no_constraints.zip"


@pytest.fixture
def grid_constraints(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_grid_constraints.zip"


@pytest.fixture
def constraint_input(shared_datadir: Path) -> Path:
    return shared_datadir / "example.in"


class TestSuiteGlide:
    @pytest.mark.needs_node("glide")
    def test_Glide(
        self, temp_working_dir: Path, test_config: Config, example_smiles: list[str], grid: Path
    ) -> None:
        rig = TestRig(Glide, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in example_smiles]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs], "inp_grid": [grid]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 5
        assert mols[0].molecules[0].n_conformers == 4
        assert -8 < mols[0].molecules[0].scores[0] < -5
        assert mols[0].n_isomers == 1

    @pytest.mark.skip()
    def test_Glide_licensing(
        self, temp_working_dir: Path, test_config: Config, example_smiles: list[str], grid: Path
    ) -> None:
        rig = TestRig(Glide, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in example_smiles] * 20
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs], "inp_grid": [grid]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 5 * 20
        assert mols[0].molecules[0].n_conformers == 4
        assert -8 < mols[0].molecules[0].scores[0] < -5
        assert mols[0].n_isomers == 1

    @pytest.mark.needs_node("glide")
    def test_Glide_constraints(
        self,
        temp_working_dir: Path,
        test_config: Config,
        example_smiles: list[str],
        grid_constraints: Path,
        constraint_input: Path,
    ) -> None:
        rig = TestRig(Glide, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in example_smiles]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(
            inputs={"inp": [inputs], "inp_grid": [grid_constraints]},
            parameters={"constraints": constraint_input},
        )
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 5
        assert mols[0].molecules[0].n_conformers == 4
        assert -8 < mols[0].molecules[0].scores[0] < -5
        assert mols[0].n_isomers == 1
