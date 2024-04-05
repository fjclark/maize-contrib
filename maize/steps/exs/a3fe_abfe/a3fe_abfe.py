"""An example node for maize."""

# pylint: disable=import-outside-toplevel, import-error
# pylance: disable=import-outside-toplevel, import-error

import csv
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal

import numpy as np
import pytest

from maize.core.interface import FileParameter, Flag, Input, Output, Parameter, Suffix
from maize.core.node import Node
from maize.utilities.chem import Isomer
from maize.utilities.io import Config
from maize.utilities.testing import TestRig


class A3feException(Exception):
    """Base class for exceptions in this module."""

    ...


class A3feSetupException(A3feException):
    """Exception raised for errors in the setup of the A3FE calculation."""

    ...


@dataclass
class ABFEResult:
    smiles: str
    dgs: np.ndarray[np.float64]
    errors: np.ndarray[np.float64]
    exceptions: list[Exception] = field(default_factory=list)


class A3feABFE(Node):
    """
    Run a single alchemical absolute binding free energy calculation
    using the A3FE package (https://github.com/michellab/a3fe).
    """

    required_callables = ["sbatch"]
    """A3FE requires slurm."""

    required_packages = ["a3fe"]

    # Inputs. We need a) a path to a directory containing the receptor pdb, water pdb,
    # run_somd.sh file, and template_config.cfg file, and b) a ligand sdf file.

    inp: Input[list[Isomer]] = Input()
    """Ligands for which to compute the ABFE."""

    # Outputs
    out: Output[list[ABFEResult]] = Output()
    """List of free energy results for each ligand."""

    # Parameters
    inp_dir: Input[Path] = Parameter()
    """
    Path to the directory containing the receptor pdb, water pdb, run_somd.sh file, and
    template_config.cfg file.
    """

    speed: Parameter[Literal["fast", "slow"]] = Parameter(default="fast")
    """
    'fast' or 'slow. Fast uses quick equilibration with 0.1 ns of sampling per window.
    while slow uses 5 ns of sampling per window. Both use 5 replicate runs.
    """

    n_repeats: Parameter[int] = Parameter(default=1)
    """The number of replicate runs"""

    dump_to: FileParameter[Path] = FileParameter(optional=True)
    """A folder to dump all generated data to"""

    tolerate_errors: Flag = Flag(default=False)
    """
    If ``True``, the node will carry on if individual ligand calculations fail.
    """

    trial: Flag = Flag(default=False)
    """
    If ``True``, will not run FEP and produce random values,
    for debugging and workflow testing purposes
    """

    # TODO: Expand options to make more flexible.

    def run(self) -> None:
        import a3fe as a3

        self.logger.info(f"Running A3FE version {a3.__version__}")

        supplied_input_ligands = self.inp.receive()
        results = []

        for lig_idx, ligand in enumerate(supplied_input_ligands):

            try:
                self._check_ligand_neutral(ligand)
                supplied_input_dir = self.inp_dir.value
                speed = self.speed.value

                self.logger.debug(f"Work directory: {self.work_dir}")

                # Create a new input directory with all required files, including the
                # ligand sdf.
                parent_workdir = self.work_dir
                created_input_dir = parent_workdir / "input"
                created_input_dir.mkdir()
                for file in supplied_input_dir.iterdir():
                    (created_input_dir / file.name).symlink_to(file)
                ligand_sdf = created_input_dir / "ligand.sdf"
                ligand.to_sdf(ligand_sdf)

                # Set up the calculation
                calc = a3.Calculation(
                    ensemble_size=self.n_repeats.value,
                    stream_log_level=logging.WARNING,
                )
                cfg = a3.SystemPreparationConfig()
                cfg.forcefields["ligand"] = "gaff2"
                lambda_values = {
                    a3.LegType.BOUND: {
                        a3.StageType.RESTRAIN: [0.0, 1.0],
                        a3.StageType.DISCHARGE: [0.0, 0.291, 0.54, 0.776, 1.0],
                        a3.StageType.VANISH: [
                            0.0,
                            0.026,
                            0.054,
                            0.083,
                            0.111,
                            0.14,
                            0.173,
                            0.208,
                            0.247,
                            0.286,
                            0.329,
                            0.373,
                            0.417,
                            0.467,
                            0.514,
                            0.564,
                            0.623,
                            0.696,
                            0.833,
                            1.0,
                        ],
                    },
                    a3.LegType.FREE: {
                        a3.StageType.DISCHARGE: [0.0, 0.222, 0.447, 0.713, 1.0],
                        a3.StageType.VANISH: [
                            0.0,
                            0.026,
                            0.055,
                            0.09,
                            0.126,
                            0.164,
                            0.202,
                            0.239,
                            0.276,
                            0.314,
                            0.354,
                            0.396,
                            0.437,
                            0.478,
                            0.518,
                            0.559,
                            0.606,
                            0.668,
                            0.762,
                            1.0,
                        ],
                    },
                }
                cfg.lambda_values = lambda_values
                cfg.slurm = True
                if speed == "fast":  # Drop the equilibration times
                    cfg.runtime_npt_unrestrained = 50
                    cfg.runtime_npt = 50
                    cfg.ensemble_equilibration_time = 100

                # Only run MD if we are not in trial mode
                if not self.trial.value:
                    calc.setup(bound_leg_sysprep_config=cfg, free_leg_sysprep_config=cfg)

                    # Run the calculation
                    runtime = 0.1 if speed == "fast" else 5
                    calc.run(adaptive=False, runtime=runtime)
                    calc.wait()

                    # Analyse the results
                    # Set the equilibration time to 0 (otherwise analysis cannot be performed)
                    calc.recursively_set_attr("_equil_time", 0)
                    calc.recursively_set_attr("_equilibrated", True)
                    dgs, errors = calc.analyse(slurm=True)
                    # Save a csv with detailed results breakdown
                    calc.get_results_df()
                    # Save the calculation
                    calc._dump()

                else:  #  We're testing
                    dgs = np.random.uniform(-10, 0, 5)
                    errors = np.random.uniform(0, 2, 5)

                result = ABFEResult(
                    smiles=ligand.to_smiles(),
                    dgs=dgs,
                    errors=errors,
                )

            except Exception as e:
                if self.tolerate_errors.value:
                    self.logger.error(f"Error processing ligand {lig_idx + 1}: {e}")
                    result = ABFEResult(
                        smiles=ligand.to_smiles(),
                        dgs=np.array([np.nan] * 5),
                        errors=np.array([np.nan] * 5),
                        exceptions=[e],
                    )
                else:
                    raise e

            finally:
                # Always copy over the output if requested, so that we can debug

                # Move results + raw data to dumping location. Make a sub-folder
                # named with the smiles string of the ligand.
                if self.dump_to.is_set:
                    dump_folder = self.dump_to.value / f"lig_{lig_idx + 1}"
                    shutil.move(self.work_dir, dump_folder)
                    # Save the smiles string of the ligand to a file in the dump folder.
                    with open(dump_folder / "smiles.txt", "w") as f:
                        f.write(ligand.to_smiles())

            results.append(result)

        # Send the output
        self.out.send(results)

    @staticmethod
    def _check_ligand_neutral(ligand: Isomer) -> None:
        """
        Check that the ligand is neutral.
        """
        if ligand.charge != 0:
            raise A3feSetupException("Ligand must be neutral.")


class SaveA3feResults(Node):
    """Save an A3fe result object to a CSV"""

    inp: Input[list[ABFEResult]] = Input()
    """ABFE results for each ligand"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV location"""

    def run(self) -> None:
        results = self.inp.receive()
        with open(self.file.filepath, "a") as out:
            writer = csv.writer(out, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            # Only write header if it's empty
            if not self.file.filepath.exists() or self.file.filepath.stat().st_size == 0:
                writer.writerow(["smiles", "repeat_no", "dg", "error"])
            for result in results:
                for i in range(len(result.dgs)):
                    writer.writerow([result.smiles, i + 1, result.dgs[i], result.errors[i]])


@pytest.fixture
def ligand_path(shared_datadir: Any) -> Any:
    return shared_datadir / "ligand" / "ligand.sdf"


@pytest.fixture
def input_dir_path(shared_datadir: Any) -> Any:
    return shared_datadir / "inp_dir"


class TestSuiteA3fe:
    def test_A3feABFE(
        self,
        temp_working_dir: Any,
        ligand_path: Path,
        input_dir_path: Path,
        test_config: Config,
    ) -> None:
        rig = TestRig(A3feABFE, config=test_config)
        inp_lig = Isomer.from_sdf(ligand_path)
        res = rig.setup_run(
            inputs={"inp": [[inp_lig]]},
            parameters={
                "speed": "fast",
                "inp_dir": input_dir_path,
                "trial": True,
            },
        )
        abfe_results = res["out"].get()
        assert len(abfe_results) == 1, "There should be one free energy estimate."
        abfe_result = abfe_results[0]
        assert len(abfe_result.dgs) == 5, "There should be 5 free energy estimates."
        assert all(
            isinstance(dg, float) for dg in abfe_result.dgs
        ), "All free energies should be floats."
        assert len(abfe_result.dgs) == len(
            abfe_result.errors
        ), "There should be an error for each free energy."
        assert all(
            isinstance(error, float) for error in abfe_result.errors
        ), "All errors should be floats."

    def test_SaveA3feResults(
        self,
        tmp_path: Path,
        test_config: Config,
        ligand_path: Path,
    ) -> None:
        mol = Isomer.from_sdf(ligand_path)
        data = [
            [
                ABFEResult(
                    smiles=mol.to_smiles(),
                    dgs=np.array([-1.0, -2.0, -3.0]),
                    errors=np.array([0.1, 0.2, 0.3]),
                )
            ]
        ]
        path = tmp_path / "out.csv"
        rig = TestRig(SaveA3feResults, config=test_config)
        rig.setup_run(
            inputs={"inp": data},
            parameters={"file": path},
        )

        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            for i, row in enumerate(reader):
                if i == 0:
                    assert row == ["smiles", "repeat_no", "dg", "error"]
                else:
                    assert row[0] == f"{mol.to_smiles()}"
                    assert row[1] == str(i)
                    assert float(row[2]) == -i
                    assert float(row[3]) == i / 10
