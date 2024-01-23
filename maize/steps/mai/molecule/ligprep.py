"""Schrodinger Ligprep prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Literal, cast

import pytest

from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.utilities import unique_id

from maize.steps.mai.common.schrodinger import Schrodinger
from maize.utilities.chem import IsomerCollection, save_smiles, load_sdf_library, save_sdf_library
from maize.utilities.io import Config


class Ligprep(Schrodinger):
    """
    Calls Schrodinger's Ligprep tool to embed small molecules and create isomers.

    Notes
    -----
    Due to Schrodinger's licensing system, each call to a tool requires
    going through Schrodinger's job server. This is run separately for
    each job to avoid conflicts with a potentially running main server.

    See Also
    --------
    :class:`~maize.steps.mai.molecule.Smiles2Molecules` :
        A simple, fast, and less accurate alternative to
        Gypsum and Ligprep, using RDKit embedding functionality.
    :class:`~maize.steps.mai.molecule.Gypsum` :
        A more advanced procedure for producing different isomers and
        high-energy conformers, and an open-source alternative to ligprep.

    """

    N_LICENSES = 1

    required_callables = ["ligprep"]

    inp: Input[list[str] | list[IsomerCollection]] = Input()
    """SMILES  or Maize IsomerCollection objects as input"""

    out: Output[list[IsomerCollection]] = Output()
    """Embedded isomer collection output"""

    epik: Flag = Flag(default=True)
    """Whether to use Epik for ionization and tautomerization"""

    ionization: Parameter[Literal[0, 1, 2]] = Parameter(default=1)
    """Ionization treatment: 0 - do not ionize / neutralize, 1 - only neutralize, 2 - both"""

    ph: Parameter[float] = Parameter(optional=True)
    """Target pH"""

    ph_tolerance: Parameter[float] = Parameter(optional=True)
    """pH tolerance"""

    max_stereo: Parameter[int] = Parameter(default=32)
    """Maximum number of stereoisomers to generate"""

    def run(self) -> None:
        input_list = self.inp.receive()
        if isinstance(input_list[0], IsomerCollection):
            # input is Isomercollection
            # This is needed to shutup mypy
            input_list = cast(list[IsomerCollection], input_list)
            input_file_path = Path("input.sdf")
            save_sdf_library(input_file_path, input_list, split_strategy="schrodinger")
            in_format_flag = "isd"

        else:
            # SMILES input
            input_list = cast(list[str], input_list)
            smiles = [smi.strip() for smi in input_list]
            input_file_path = Path("input.smi")
            save_smiles(input_file_path, smiles)
            in_format_flag = "ismi"

        name = f"ligprep-{unique_id(12)}"
        output_sdf = Path(f"{name}.sdf")
        # While it would be enticing to add '-LOCAL' here, this will
        # cause a DeprecationWarning that actually crashes the program :(
        command = (
            f"{self.runnable['ligprep']} -{in_format_flag} {input_file_path.as_posix()} "
            f"-osd {output_sdf.as_posix()} -i {self.ionization.value} "
            f"-s {self.max_stereo.value} -NJOBS {self.n_jobs.value} "
            f"-HOST {self.host.value} "
        )
        if self.epik.value:
            command += "-epik "
        if self.ph.is_set:
            command += f"-ph {self.ph.value} "
        if self.ph_tolerance.is_set:
            command += f"-pht {self.ph_tolerance.value} "

        self.guard.wait("LIGPREP_MAIN", number=self.N_LICENSES * self.n_jobs.value)
        self._run_schrodinger_job(command, name=name, verbose=True)
        mols = load_sdf_library(output_sdf, split_strategy="schrodinger-tag")
        self.out.send(mols)


@pytest.fixture
def sdf_input(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_ligands.sdf"


@pytest.fixture
def protonation_test_input(shared_datadir: Path) -> Path:
    return shared_datadir / "fluorophenol.sdf"


class TestSuiteLigprep:
    @pytest.mark.needs_node("ligprep")
    def test_Ligprep(
        self,
        temp_working_dir: Path,
        test_config: Config,
        example_smiles: list[str],
        sdf_input: Path,
        protonation_test_input: Path,
    ) -> None:
        rig = TestRig(Ligprep, config=test_config)
        res = rig.setup_run(inputs={"inp": [example_smiles]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 5
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 1

        isoc = load_sdf_library(sdf_input, split_strategy="none")
        res = rig.setup_run(inputs={"inp": [isoc]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 1
        assert mols[1].n_isomers == 1
        assert mols[2].n_isomers == 1

        isoc = load_sdf_library(protonation_test_input, split_strategy="none")
        res = rig.setup_run(inputs={"inp": [isoc]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 1
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 2
