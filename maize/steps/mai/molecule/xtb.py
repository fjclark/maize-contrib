from pathlib import Path
from typing import Any, List, Dict
from subprocess import CompletedProcess
import re
import json
import logging
import pytest
import numpy as np


from maize.utilities.testing import TestRig
from maize.steps.mai.molecule.compchem_utils import Loader, atom_string
from maize.core.node import Node
from maize.steps.io import LoadData, Return
from maize.steps.mai.molecule import Smiles2Molecules
from maize.steps.mai.molecule.crest import Crest
from maize.core.workflow import Workflow
from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.execution import JobResourceConfig


log = logging.getLogger("run")

SP_ENERGY_REGEX = re.compile(r"\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s*")
SP_GRADIENT_REGEX = re.compile(r"\s*GRADIENT NORM\s+(-?\d+\.\d+)\s*")

AtomType = dict[str, str | int | list[float]]

def create_constraints_xtb(iso: Isomer, path: Path) -> str:
    """
    Create constraint file for XTB calculations starting from
    the isomer object and the constrained tag associated

    """
    constr_indexes = iso.tags["constraints"].replace("[", "").replace("]", "")
    constraint_file = "{}/{}_xtb_constraints.inp".format(path, iso.name)

    with open(constraint_file, "w") as f:
        f.write("{}\n".format("$fix"))
        f.write("{}\n".format("   atoms: " + constr_indexes))
        f.write("{}\n".format("$end"))

    return constraint_file


def _xtb_energy_parser_sp(stdout: str) -> float:
    """
    Parse energy from xtb output.

    Parameters
    ----------
    stdout
        string with path of stdout

    Returns
    -------
    float
        energy value
    """
    res = re.search(SP_ENERGY_REGEX, stdout)
    if res:
        return float(res.group(1))
    return np.nan


def _xtb_gradient_parser_sp(stdout: str) -> float:
    """
    Parse Gradient from xtb output.

    Parameters
    ----------
    stdout
        string with path of stdout

    Returns
    -------
    float
        gradient value
    """
    res = re.search(SP_GRADIENT_REGEX, stdout)
    if res:
        return float(res.group(1))
    return np.nan


class Xtb(Node):
    """
    Runs XTB semiempirical method on IsomerCollection class.

    Currently, the capabilities are
        * Performing geometry optimisation at GFN2-xTB level using approximate normal
          coordinate rational function optimizer (ANCopt)
        * Calculate partial atomic charges (Mulliken)
        * Calculate Wieberg partial bond orders
        * Return optimised 3D coordinates with the final energy and optimisation trajectory points

    References
    ----------
    API documentation: https://xtb-docs.readthedocs.io/en/latest/contents.html
    Key citation reference for the XTB methods and current implementation:
    {C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher, S. Grimme WIREs Comput. Mol. Sci., 2020, 11, e01493. DOI: 10.1002/wcms.1493}

    """

    required_callables = ["xtb"]

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    opt: Parameter[bool] = Parameter(default=True)
    """Enable geometry optimisation"""

    batch: Flag = Flag(default=True)
    """Flag to submit to SLURM queueing system"""

    n_jobs: Parameter[int] = Parameter(default=100)
    """Number of parallel processes to use"""

    _ENERGY_REGEX = re.compile(r"energy:\s*(-?\d+\.\d+)")
    _GRADIENT_REGEX = re.compile(r"gnorm:\s*(-?\d+\.\d+)")

    @staticmethod
    def get_atomic_charges(iso_dirname: Path) -> List[Dict[str, float]]:
        """
        gets atomic charge of individual atoms in molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files

        Returns
        -------
        List[Dict[str, float]]
            list of dictionaries contaning label of atomic element and
            respective atomic charge
        """
        charges_file = iso_dirname / "charges"
        atom_id = 1
        charges = []
        with open(charges_file, "r") as f:
            for line in f.readlines():
                if len(line) != 0:
                    charges.append({"atom_id": atom_id, "charge": float(line)})
                    atom_id += 1
        if not charges:
            raise ValueError("Charges not found in molecule")
        return charges

    @staticmethod
    def get_wieberg_bo(iso_dirname: Path) -> List[Dict[str, float]]:
        """
        Gets wieberg bond orders for Bonds in molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files

        Returns
        -------
        List[Dict[str, float]]
            list of dictionaries contaning label of atomic elements and
            respective bond order
        """
        charges_file = iso_dirname / "wbo"
        wbos = []
        with open(charges_file, "r") as f:
            for line in f.readlines():
                line_lst = line.split()
                if len(line_lst) == 3:
                    wbos.append(
                        {
                            "atom_id1": int(line_lst[0]),
                            "atom_id2": int(line_lst[1]),
                            "wbo": float(line_lst[2]),
                        }
                    )
        if not wbos:
            raise ValueError("WBOs not found in molecule")
        return wbos

    @staticmethod
    def get_final_energy(iso_dirname: Path) -> tuple[float, float]:
        """
        Gets energy and gradient of the molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files

        Returns
        -------
        tuple[float, float]
            energy of the molecule, gradient of the energy
        """

        final_file = iso_dirname / "xtbopt.xyz"
        energy = np.nan
        gradient = np.nan
        with open(final_file, "r") as f:
            for line in f.readlines():
                if "energy:" in line:
                    energy_match = re.search(Xtb._ENERGY_REGEX, line)
                    gradient_match = re.search(Xtb._GRADIENT_REGEX, line)
                    if energy_match and gradient_match:
                        energy = float(energy_match.group(1))
                        gradient = float(gradient_match.group(1))
                        break
        return (energy, gradient)


    @staticmethod
    def get_trajectory(
        iso_dirname: Path, filename: str
    ) -> list[dict[str, list[dict[str, str | int | list[float]]] | float | float]]:
        """
        Gets energy and gradient of the molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files
        filename
            Path of output file with trajectory

        Returns
        -------
        List[dict[str, list[dict[str, str|int|list[float]]] | float | float ]]
           list of dictionaries contaning results about single conformers and about
           individual atoms in the conformer
        """
        trajectory_file = iso_dirname / filename
        with open(trajectory_file, "r") as f:
            read_coords = 0
            conformers: list[dict[str, list[AtomType] | float | float]] = []
            conformer_energy = np.nan
            gradient = np.nan
            number_atoms = 0
            atoms: list[AtomType] = []
            atom_id = int(1)
            for line in f.readlines():
                line_lst = line.split()
                if read_coords == 1 and line_lst[0] == "energy:" and line_lst[2] == "gnorm:":
                    conformer_energy = float(line_lst[1])
                    gradient = float(line_lst[3])
                elif read_coords == 1 and line_lst[0] in atom_string:
                    atoms.append(
                        {
                            "element": str(line_lst[0]),
                            "atom_id": atom_id,
                            "coords": [float(line_lst[1]), float(line_lst[2]), float(line_lst[3])],
                        }
                    )
                    atom_id += 1
                elif len(line_lst) == 1:
                    if number_atoms != 0:
                        conformers.append(
                            {"atoms": atoms, "energy": conformer_energy, "gradient": gradient}
                        )
                    number_atoms = int(line_lst[0])
                    atom_id = 1
                    read_coords = 1
                    atoms = []
            conformers.append({"atoms": atoms, "energy": conformer_energy, "gradient": gradient})

        if not conformers:
            raise ValueError("Trajectory not found in molecule")
        return conformers

    @staticmethod
    def get_xyz_json(iso_dirname: Path) -> list[AtomType]:
        """
        Gets atomic information as list of dictionaries

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files


        Returns
        -------
        list[dict[str, str | int | list[float]]]
           list of dictionaries contaning results about individual atoms in the conformer
        """

        trajectory_file = iso_dirname / "input.sdf"
        sdf_loader = Loader(str(trajectory_file))
        molecule = sdf_loader.molecule()
        atoms: list[AtomType] = []
        for atom in molecule.atoms:
            atoms.append({"element": atom.label, "atom_id": atom.number, "coords": atom.position})
        return atoms

    def _parse_xtb_outputs(
        self,
        mols: list[IsomerCollection],
        mol_outputs: list[list[list[Path]]],
        results: list[CompletedProcess[bytes]],
    ) -> None:
        """
        Parse xtb outputs

        Parameters
        ----------
        mols
            List of IsomerCollection objects corresponding to the molecules in
            the calculation
        mol_outputs
            list containing list of paths for individual calculation output files
        results
            Results of the jobs
        """

        iso_dict = {}
        for mol in mols:
            for iso in mol.molecules:
                iso_dict[iso.get_tag("xtb_iso_idx")] = iso
                
        count = 0
        for i, mol_folder in enumerate(mol_outputs):
            for j, iso_dirname in enumerate(mol_folder):
                isomer = iso_dict[f"{i}_{j}"]
                isomer_final_geometries = {}
                isomer_trajectories = {}
                isomer_charges = {}
                isomer_wbos = {}
                isomer_gradients = {}
                isomer_energies = {}
                isomer_xtb_exit_codes = {}
                
                
                for k, conf_name in enumerate(iso_dirname):
                    conformer = isomer.conformers[k]
                    conf_output = conf_name / "xtbopt.xyz"
                    conf_stdout = results[count].stdout.decode()
                    
                    with open(conf_name / 'xtb_out.txt', 'w') as out:
                        out.write(conf_stdout)


                    if not conf_output.exists() and self.opt.value:
                        self.logger.warning("XTB failed for '%s'", conformer)
                        continue

                    ### check calculation status
                    exit_code = 1
                    pattern = "convergence criteria satisfied after"
                    for line in conf_stdout.split("\n"):
                        if pattern in line:
                            exit_code = 0
                        else:
                            continue

                    isomer_xtb_exit_codes[k] = exit_code

                    try:
                        isomer_charges[k] = Xtb.get_atomic_charges(conf_name)
                        isomer_wbos[k] = Xtb.get_wieberg_bo(conf_name)
                    except ValueError:
                        log.info(f"charges and wbo not available for conformer {k} {conf_name}")

                    if self.opt.value:
                        try:
                            isomer_trajectories[k] = Xtb.get_trajectory(conf_name, "xtbopt.log")
                            isomer_final_geometries[k] = Xtb.get_trajectory(conf_name, "xtbopt.xyz")
                            isomer_energies[k] = Xtb.get_final_energy(conf_name)[0]
                            isomer_gradients[k] = Xtb.get_final_energy(conf_name)[1]
                        except ValueError:
                            log.info(
                                f"trajectories, energies and gradient not "
                                f"available for conformer {k} {conf_name}"
                            )
                    else:
                        xtb_energy = _xtb_energy_parser_sp(conf_stdout)
                        xtb_gradient = _xtb_gradient_parser_sp(conf_stdout)
                        sp_atoms = Xtb.get_xyz_json(conf_name)
                        xtb_trajectory = json.dumps(
                            [{"atoms": sp_atoms, "energy": xtb_energy, "gradient": xtb_gradient}]
                        )
                    count += 1

                isomer.set_tag("XTB_exit_codes", json.dumps(isomer_xtb_exit_codes))
                isomer.set_tag("final_geometries", json.dumps(isomer_final_geometries))
                isomer.set_tag("xtb_charges", json.dumps(isomer_charges))
                isomer.set_tag("xtb_wbos", json.dumps(isomer_wbos))
                isomer.set_tag("xtb_trajectory", json.dumps(isomer_trajectories))
                isomer.set_tag("xtb_energy", json.dumps(isomer_energies))
                isomer.set_tag("xtb_gradient", json.dumps(isomer_gradients))

    def run(self) -> None:
        mols = self.inp.receive()

        commands: list[str] = []
        confs_paths: list[Path] = []
        mol_outputs: list[list[list[Path]]] = []
        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}")
            mol_path.mkdir()
            isomer_outputs: list[list[Path]] = []
            self.logger.info("XTB optimisation for molecule %s: '%s'", i, mol)
            for j, isomer in enumerate(mol.molecules):
                self.logger.info("  XTB optimisation for isomer %s: '%s'", j, isomer)
                isomer.set_tag("xtb_iso_idx", f"{i}_{j}")
                iso_path = mol_path / f"isomer-{j}"
                iso_path.mkdir()
                conformer_outputs: list[Path] = []
                conformer_tag_dict = {}
                for k, conformer in enumerate(isomer.conformers):
                    self.logger.info("  XTB optimisation for conformer %s: '%s'", k, conformer)
                    conformer_tag_dict[k] = f"{i}_{j}_{k}"
                    conf_path = iso_path / f"conformer-{k}"
                    conf_path.mkdir()
                    confs_paths.append(conf_path)
                    input_flname = "input.xyz"
                    output_dirname = conf_path
                    if isomer.has_tag("constraints"):
                        constraints = "--input " + create_constraints_xtb(isomer, conf_path)
                        self.logger.info(f"found constraint {constraints} for isomer {j}")
                    else:
                        constraints = ""
                        self.logger.info(f"no constraint for isomer {j}")

                    try:
                        conformer.to_xyz(path=conf_path / input_flname, tag_name=f"{i}_{j}_{k}")
                    except ValueError as err:
                        self.logger.warning(
                            "Skipping '%s' due to XYZ conversion error:\n %s", conformer, err
                        )

                    if isomer.has_tag("constraints"):
                        keywords = f" {constraints} "
                    else:
                        keywords = ""
                    if self.opt.value:
                        keywords += " --opt"
                    command = f"{self.runnable['xtb']} {input_flname} {keywords}"
                    commands.append(command)
                    conformer_outputs.append(output_dirname)
                isomer_outputs.append(conformer_outputs)
                print(isomer.tags)
            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=confs_paths,
            verbose=True,
            raise_on_failure=True,
            batch_options=JobResourceConfig(
                cores_per_process=1,
                custom_attributes={"mem": "16G"},
                exclusive_use=False,
                walltime="02:00:00",
            ),
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_xtb_outputs(mols, mol_outputs, results)
        self.out.send(mols)


@pytest.fixture
def testing() -> list[str]:
    return ["CNC(=O)", "CCO"]


class TestSuiteXtb:
    @pytest.mark.needs_node("xtb")
    def test_Xtb(
        self,
        temp_working_dir: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(Xtb, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in ["C", "N"]]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert np.isfinite(mol.molecules[0].tags["xtb_energy"])
            assert np.isfinite(mol.molecules[0].tags["xtb_gradient"])
            assert len(json.loads(mol.molecules[0].tags["xtb_trajectory"])) > 1
        assert len(json.loads(mols[0].molecules[0].tags["xtb_charges"])) == 5
        assert len(json.loads(mols[0].molecules[0].tags["xtb_wbos"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_charges"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_wbos"])) == 3

    @pytest.mark.needs_node("xtb")
    @pytest.mark.needs_node("crest")
    def test_Crest_Xtb(
        self,
        testing: Any,
        test_config: Any,
    ) -> None:
        flow = Workflow(name="test_xtb", level="INFO", cleanup_temp=False)
        flow.config = test_config

        load = flow.add(LoadData[list[str]])
        embe = flow.add(Smiles2Molecules)
        crest_nod = flow.add(Crest)
        opt = flow.add(Xtb)
        ret = flow.add(Return[list[IsomerCollection]])

        flow.connect_all(
            (load.out, embe.inp),
            (embe.out, crest_nod.inp),
            (crest_nod.out, opt.inp),
            (opt.out, ret.inp),
        )

        load.data.set(testing)
        embe.n_variants.set(1)

        flow.check()
        flow.execute()

        mols = ret.get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert np.isfinite(mol.molecules[0].tags["xtb_energy"])
            assert np.isfinite(mol.molecules[0].tags["xtb_gradient"])
            assert len(json.loads(mol.molecules[0].tags["xtb_trajectory"])) > 1
        assert len(json.loads(mols[0].molecules[0].tags["xtb_charges"])) == 5
        assert len(json.loads(mols[0].molecules[0].tags["xtb_wbos"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_charges"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_wbos"])) == 3

    @pytest.mark.needs_node("xtb")
    def test_Xtb_SP(
        self,
        temp_working_dir: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(Xtb, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in ["C", "N"]]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]}, parameters={"opt": False})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert len(mol.molecules[0].tags["xtb_energy"]) > 1
            assert len(mol.molecules[0].tags["xtb_gradient"]) > 1
            assert len(json.loads(mol.molecules[0].tags["xtb_trajectory"])) == 1
        assert len(json.loads(mols[0].molecules[0].tags["xtb_charges"])) == 5
        assert len(json.loads(mols[0].molecules[0].tags["xtb_wbos"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_charges"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_wbos"])) == 3
