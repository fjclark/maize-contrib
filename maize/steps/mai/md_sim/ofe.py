"""OpenFE RBFE implementation"""

# pylint: disable=import-outside-toplevel, import-error

import copy
import csv
from dataclasses import dataclass
import itertools
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Annotated, Any, Literal, Sequence, cast

import networkx as nx
import numpy as np
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, Suffix, Input, Output, FileParameter
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


log = logging.getLogger(f"run-{os.getpid()}")


# TODO figure out n_jobs params etc
def _parametrise_mols(
    mols: Sequence[Isomer], out_file: Path, n_cores: int = 2, n_workers: int = 1
) -> None:
    """Uses OpenFF bespokefit to parametrise small molecules"""
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from openff.bespokefit.executor import BespokeExecutor, BespokeWorkerConfig, wait_until_complete
    from openff.qcsubmit.common_structures import QCSpec
    from openff.toolkit.topology import Molecule
    from openff.toolkit import ForceField
    from openff.toolkit.utils.exceptions import ParameterLookupError

    ffmols = [Molecule.from_rdkit(mol._molecule) for mol in mols]
    spec = QCSpec(method="ani2x", basis=None, program="torchani", spec_name="ani2x")
    factory = BespokeWorkflowFactory(
        initial_force_field="openff-2.1.0.offxml", default_qc_specs=[spec]
    )
    log.info("Creating parametrisation schemas...")
    schemas = factory.optimization_schemas_from_molecules(ffmols, processors=n_cores * n_workers)

    # Do the fitting
    with BespokeExecutor(
        n_fragmenter_workers=1,
        n_optimizer_workers=1,
        n_qc_compute_workers=max(n_workers - 2, 1),
        qc_compute_worker_config=BespokeWorkerConfig(n_cores=n_cores),
    ) as exec:
        log.info("Fitting...")
        tasks = [exec.submit(input_schema=schema) for schema in schemas]
        outputs = [wait_until_complete(task) for task in tasks]

    # Combine FFs into single file, this is just following the `combine.py` CLI tool
    ffs: ForceField = []
    for i, out in enumerate(outputs):
        ff_file = f"mol-{i}.offxml"
        if out.bespoke_force_field is not None:
            out.bespoke_force_field.to_file(ff_file)
            ffs.append(ForceField(ff_file, load_plugins=True, allow_cosmetic_attributes=True))
        else:
            log.warning("Parameterisation failed for %s", mols[i])

    # Combine torsions
    master = copy.deepcopy(ffs[0])
    for ff in ffs[1:]:
        for parameter in ff.get_parameter_handler("ProperTorsions").parameters:
            try:
                master.get_parameter_handler("ProperTorsions")[parameter.smirks]
            except ParameterLookupError:
                master.get_parameter_handler("ProperTorsions").add_parameter(parameter=parameter)

    # Save combined FF
    master.to_file(filename=out_file, discard_cosmetic_attributes=True)


@dataclass
class FEPResult:
    smiles: tuple[str, str]
    ddg: float
    ddg_error: float


class IndexingDict(dict[Any, int]):
    """Dictionary that converts each entry into a unique index"""

    def __getitem__(self, __key: Any) -> int:
        if __key not in self:
            super().__setitem__(__key, len(self))
        return super().__getitem__(__key)


EPS = 1e-2


class MakeAbsolute(Node):
    """Convert FEP results to an absolute free energy"""

    inp: Input[dict[tuple[str, str], FEPResult]] = Input()
    """FEP result input"""

    inp_mols: Input[list[IsomerCollection]] = Input()
    """Original molecules"""

    inp_ref: Input[Isomer] = Input(cached=True)
    """Reference molecule to compute absolute binding energies"""

    ref_score: Parameter[float] = Parameter(optional=True)
    """Reference score if not included as tag in reference mol (kJ/mol)"""

    out: Output[list[IsomerCollection]] = Output()
    """Tagged mol output"""

    def run(self) -> None:
        from cinnabar.stats import mle

        results = self.inp.receive()
        mols = self.inp_mols.receive()
        ref = self.inp_ref.receive()
        if ref.scored:
            ref_score = ref.scores[0]
        elif self.ref_score.is_set:
            ref_score = self.ref_score.value
        else:
            ref_score = 0.0
        self.logger.info("Using reference score of %s", ref_score)

        isos = {iso.inchi: iso for mol in mols for iso in mol.molecules}
        if ref.inchi not in isos:
            isos[ref.inchi] = ref

        # In some cases we might be calculating a partial network only, which can
        # make it impossible to provide real absolute BFE values. Considering usage
        # in RL loops, the safest thing is to just return NaNs and a warning.
        if ref.inchi not in set(itertools.chain(*results)):
            self.logger.warning(
                "Reference molecule '%s' not found in results or molecules (%s)",
                ref.inchi,
                isos.keys(),
            )
            for iso in isos.values():
                iso.set_tag("fep", np.nan)
                iso.set_tag("fep_error", np.nan)
                iso.score_tag = "fep"
            self.out.send(mols)
            return

        # Build graph to get maximum likelihood estimate
        graph = nx.DiGraph()

        # Use the same 'paranoid' approach as OpenFE and convert to indices (and back again)
        # https://github.com/OpenFreeEnergy/openfe/blob/main/openfecli/commands/gather.py
        name2idx = IndexingDict()
        for (a, b), res in results.items():
            idx_a, idx_b = name2idx[a], name2idx[b]

            # A NaN result implies a failed edge and thus can cause the network
            # to be split into disconnected subgraphs. We will try to save the
            # campaign by using the subgraph including the reference later.
            if np.isnan(res.ddg):
                graph.add_node(idx_a)
                graph.add_node(idx_b)
                continue

            # MLE fails when the error is 0
            graph.add_edge(idx_a, idx_b, f_ij=res.ddg, f_dij=max(res.ddg_error, EPS))

        graph.nodes[name2idx[ref.inchi]]["f_i"] = ref_score
        graph.nodes[name2idx[ref.inchi]]["f_di"] = 0.1

        idx2name = {v: k for k, v in name2idx.items()}

        # Failed edges can partition the graph, in those cases we have no choice but to
        # only compute absolute values for the largest subgraph containing the reference
        if nx.number_weakly_connected_components(graph) > 1:
            node_lists = list(nx.weakly_connected_components(graph))

            # Find subgraph containing reference, set all other nodes to NaN
            for nodes in node_lists:
                if name2idx[ref.inchi] in nodes:
                    graph = graph.subgraph(nodes)
                    continue

                for node in nodes:
                    iso = isos[idx2name[node]]
                    iso.set_tag("fep", np.nan)
                    iso.set_tag("fep_error", 0.0)
                    iso.score_tag = "fep"

        # Absolute FEs and covariance, we're only interested in the variance
        f_i, df_i = mle(graph, factor="f_ij", node_factor="f_i")
        df_i = np.sqrt(np.diagonal(df_i))

        for idx, dg, dg_err in zip(graph.nodes, f_i, df_i):
            iso = isos[idx2name[idx]]
            iso.set_tag("fep", dg)
            iso.set_tag("fep_error", dg_err)
            iso.score_tag = "fep"
        self.out.send(mols)


class SaveOpenFEResults(Node):
    """Save OpenFE result objects to a CSV"""

    inp: Input[dict[tuple[str, str], FEPResult]] = Input()
    """FEP result input"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV location"""

    def run(self) -> None:
        results = self.inp.receive()
        with self.file.filepath.open("w") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["origin", "target", "origin-smiles", "target-smiles", "ddg", "error"])
            for (origin, target), edge in results.items():
                writer.writerow([origin, target, *edge.smiles, edge.ddg, edge.ddg_error])


class OpenRFE(Node):
    """Run an RBFE campaign using Open Free Energy"""

    SM_FFS = Literal[
        "gaff-2.11", "openff-1.3.1", "openff-2.1.0", "espaloma-0.2.2", "smirnoff99Frosst-1.1.0"
    ]

    required_callables = ["openfe"]
    """
    openfe
        OpenFE executable, included with the package

    """

    required_packages = ["openfe"]
    """
    openfe
        OpenFE python package

    """

    inp: Input[list[Isomer]] = Input()
    """Molecule inputs"""

    inp_ref: Input[Isomer] = Input(optional=True)
    """Reference molecule input for star-maps"""

    inp_protein: Input[Annotated[Path, Suffix("pdb")]] = Input(cached=True)
    """Protein structure"""

    out: Output[dict[tuple[str, str], FEPResult]] = Output()
    """Calculated edges"""

    continue_from: FileParameter[Path] = FileParameter(optional=True)
    """A folder containing a partially-run OpenFE campaign to continue from"""

    dump_to: FileParameter[Path] = FileParameter(optional=True)
    """A folder to dump all generated data to"""

    mapping: Parameter[Literal["star", "minimal", "custom", "existing"]] = Parameter(
        default="minimal"
    )
    """Type of network to use for mapping"""

    network: FileParameter[Annotated[Path, Suffix("edge")]] = FileParameter(optional=True)
    """An optional alternative FEPMapper atom mapping file, use ``mapping = "custom"``"""

    temperature: Parameter[float] = Parameter(default=298.15)
    """Temperature in Kelvin to use for all simulations"""

    ion_concentration: Parameter[float] = Parameter(default=0.15)
    """Ion concentration in molar"""

    ion_pair: Parameter[tuple[str, str]] = Parameter(default=("Na+", "Cl-"))
    """Positive and negative ions to use for neutralization"""

    neutralize: Flag = Flag(default=True)
    """Whether to neutralize the system"""

    nonbonded_cutoff: Parameter[float] = Parameter(default=1.2)
    """Nonbonded cutoff in nm"""

    equilibration_length: Parameter[int] = Parameter(default=2000)
    """Length of equilibration simulation in ps"""

    production_length: Parameter[int] = Parameter(default=5000)
    """Length of production simulation in ps"""

    molecule_forcefield: Parameter[str] = Parameter(default="openff-2.1.0")
    """
    Force field to use for the small molecule. For recommended options see
    :attr:`OpenFE.SM_FFS`, for all options :meth:`OpenFE.available_ffs`. If
    you want to use bespoke fitting using OpenFF-bespokefit, specify `bespoke`.

    """

    solvent: Parameter[Literal["tip3p", "spce", "tip4pew", "tip5p"]] = Parameter(default="tip3p")
    """Water model to use"""

    padding: Parameter[float] = Parameter(default=1.2)
    """Minimum distance of the solute to the box edge"""

    sampler: Parameter[Literal["repex", "sams", "independent"]] = Parameter(default="repex")
    """Sampler to use"""

    n_repeats: Parameter[int] = Parameter(default=3)
    """Number of simulation repeats"""

    n_replicas: Parameter[int] = Parameter(default=11)
    """Number of replicas to use"""

    n_lambda: Parameter[int] = Parameter(default=11)
    """Number of lambda windows to use"""

    n_jobs: Parameter[int] = Parameter(default=1)
    """
    Number of calculations to perform simultaneously. Should be equal
    to the number of GPUs if local, or the number of batch submissions.

    """

    platform: Parameter[Literal["CUDA", "CPU", "OpenCL", "Reference"]] = Parameter(optional=True)
    """The OpenMM compute platform"""

    trial: Flag = Flag(default=False)
    """
    If ``True``, will not run FEP and produce random values,
    for debugging and workflow testing purposes

    """

    @classmethod
    def available_ffs(cls) -> list[str]:
        """Lists all currently available small molecule force fields"""
        from openmmforcefields.generators import SystemGenerator

        return cast(list[str], SystemGenerator.SMALL_MOLECULE_FORCEFIELDS)

    def run(self) -> None:
        import openfe
        import gufe
        from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
        from pint import DimensionalityError
        from openff.units import unit

        # Prepare mapper
        mapper = openfe.LomapAtomMapper()
        scorer = openfe.lomap_scorers.default_lomap_score

        # Create molecules
        isomers = self.inp.receive()
        if not self.mapping.value == "custom":
            for iso in isomers:
                iso.name = iso.inchi

        isos = {iso.name: iso for iso in isomers}
        mols = {
            name: openfe.SmallMoleculeComponent.from_rdkit(iso._molecule)
            for name, iso in isos.items()
        }

        # Generate network
        self.logger.info("Generating '%s' network", self.mapping.value)
        if self.mapping.value == "star":
            isomer_ref = self.inp_ref.receive()
            isomer_ref.name = isomer_ref.inchi
            ref = openfe.SmallMoleculeComponent.from_rdkit(isomer_ref._molecule)
            planner = openfe.ligand_network_planning.generate_radial_network
            network = planner(mols.values(), central_ligand=ref, mappers=[mapper], scorer=scorer)

            # Add reference to library for later lookup
            isos[isomer_ref.name] = isomer_ref

        elif self.mapping.value == "minimal":
            mols_to_map = list(mols.values())
            if (opt_ref := self.inp_ref.receive_optional()) is not None:
                opt_ref.name = opt_ref.inchi
                ref = openfe.SmallMoleculeComponent.from_rdkit(opt_ref._molecule)
                self.logger.info("Using '%s' as a reference", opt_ref.name)
                mols_to_map.append(ref)

                # Add reference to library for later lookup
                isos[opt_ref.name] = opt_ref

            planner = openfe.ligand_network_planning.generate_minimal_spanning_network
            try:
                network = planner(mols_to_map, mappers=[mapper], scorer=scorer)

            # This is a bit of a hack, it would be nicer if we could just get a list
            # of failed mappings as an additional return value. For now we just catch
            # the error, inform the user about missing edges, and try again.
            except RuntimeError as err:
                missing = re.findall(r"name=([A-Z\-]*)", err.args[0])
                if not missing:
                    raise
                for missing_mol in missing:
                    self.logger.warning("Mapper was unable to create an edge for '%s'", missing_mol)
                    mols.pop(missing_mol)
                network = planner(mols_to_map, mappers=[mapper], scorer=scorer)

        elif self.mapping.value == "custom" and self.network.is_set:
            network = openfe.ligand_network_planning.load_fepplus_network(
                ligands=mols.values(), mapper=mapper, network_file=self.network.filepath
            )
        elif self.mapping.value == "existing" and self.continue_from.is_set:
            with (self.continue_from.filepath / "network.graphml").open("r") as file:
                network = openfe.LigandNetwork.from_graphml(file.read())

        msg = "Created network with following mappings:"
        for mapping in network.edges:
            msg += (
                f"\n  {mapping.componentA.name}-{mapping.componentB.name}"
                f"  score={scorer(mapping):4.4f}"
            )
        self.logger.info(msg)

        # Save the network for reference / reuse
        with Path("network.graphml").open("w") as file:
            file.write(network.to_graphml())

        # Solvation
        p_ion, n_ion = self.ion_pair.value
        solvent = openfe.SolventComponent(
            positive_ion=p_ion,
            negative_ion=n_ion,
            neutralize=self.neutralize.value,
            ion_concentration=self.ion_concentration.value * unit.molar,
        )

        # Receptor
        protein_file = self.inp_protein.receive()
        protein = openfe.ProteinComponent.from_pdb_file(protein_file.as_posix())

        # Small molecule FF
        sm_ff = self.molecule_forcefield.value
        if sm_ff == "bespoke":
            _parametrise_mols(list(isos.values()), out_file=Path("bespoke.offxml"))
            sm_ff = "bespoke.offxml"

        # RBFE settings
        settings = RelativeHybridTopologyProtocol.default_settings()
        settings.thermo_settings.temperature = self.temperature.value * unit.kelvin
        settings.engine_settings.compute_platform = (
            self.platform.value if self.platform.is_set else None
        )
        settings.forcefield_settings.small_molecule_forcefield = sm_ff
        settings.alchemical_sampler_settings.n_repeats = self.n_repeats.value
        settings.alchemical_sampler_settings.sampler_method = self.sampler.value
        settings.alchemical_sampler_settings.n_replicas = self.n_replicas.value
        settings.alchemical_settings.lambda_windows = self.n_lambda.value
        settings.solvation_settings.solvent_model = self.solvent.value
        settings.solvation_settings.solvent_padding = self.padding.value * unit.nanometers
        settings.system_settings.nonbonded_cutoff = self.nonbonded_cutoff.value * unit.nanometers
        settings.simulation_settings.equilibration_length = (
            self.equilibration_length.value * unit.picosecond
        )
        settings.simulation_settings.production_length = (
            self.production_length.value * unit.picosecond
        )
        protocol = RelativeHybridTopologyProtocol(settings)

        # Setup transforms
        self.logger.info("Generating transforms")
        transforms: list[dict[str, gufe.tokenization.GufeTokenizable]] = []
        for mapping in network.edges:
            dags: dict[str, gufe.tokenization.GufeTokenizable] = {}

            # Filter out self-mappings (charge changes are not allowed at this time)
            if mapping.componentA.name == mapping.componentB.name:
                self.logger.warning(
                    "Cannot run edge between identical components ('%s')", mapping.componentA.name
                )
                continue

            for leg in ("solvent", "complex"):
                a_setup = {"ligand": mapping.componentA, "solvent": solvent}
                b_setup = {"ligand": mapping.componentB, "solvent": solvent}

                if leg == "complex":
                    a_setup["protein"] = protein
                    b_setup["protein"] = protein

                a = openfe.ChemicalSystem(a_setup, name=f"{mapping.componentA.name}_{leg}")
                b = openfe.ChemicalSystem(b_setup, name=f"{mapping.componentB.name}_{leg}")
                transform = openfe.Transformation(
                    stateA=a,
                    stateB=b,
                    mapping={"ligand": mapping},
                    protocol=protocol,
                    name=f"rbfe_{a.name}_{b.name}_{leg}",
                )
                dags[leg] = transform
            transforms.append(dags)

        if self.continue_from.is_set:
            for res_file in self.continue_from.filepath.glob("*_res.json"):
                self.logger.info("Found existing result, copying %s", res_file)
                shutil.copy(res_file, Path())

        # Prepare commands
        commands = []
        for dags in transforms:
            for transform in dags.values():
                # Only run required edges
                if not (res_file := Path(f"{transform.name}_res.json")).exists():
                    tf_dir = Path(f"tf-{transform.name}")
                    tf_dir.mkdir()
                    tf_json = tf_dir / f"{transform.name}.json"
                    transform.dump(tf_json)
                    commands.append(
                        f"{self.runnable['openfe']} quickrun -d {tf_dir.as_posix()} "
                        f"-o {res_file.as_posix()} {tf_json.as_posix()}"
                    )

        # Run
        use_mps = (
            self.platform.is_set and self.platform.value == "CUDA" and self.batch_options.is_set
        )
        self.logger.info("Running %s transforms", 2 * len(transforms))

        if not self.trial.value:
            self.run_multi(
                commands,
                n_jobs=self.n_jobs.value,
                raise_on_failure=False,
                cuda_mps=use_mps,
            )

        def _failed_edge(a: str, b: str, test_data: bool = False) -> FEPResult:
            return FEPResult(
                ddg=np.random.normal(scale=2) if test_data else np.nan,
                ddg_error=np.random.random() if test_data else np.nan,
                smiles=(isos[a].to_smiles(remove_h=True), isos[b].to_smiles(remove_h=True)),
            )

        # Parse results
        msg = "Parsing results"
        results = {}
        for dags in transforms:
            data = {}

            transform = list(dags.values())[0]
            a = transform.stateA.name.removesuffix("_solvent")
            b = transform.stateB.name.removesuffix("_solvent")

            # Catch failed edges
            if any(not Path(f"{tf.name}_res.json").exists() for tf in dags.values()):
                msg += f"\n  {a} -> {b}:  failed (no result)"
                results[(a, b)] = _failed_edge(a, b, test_data=self.trial.value)
                continue

            parsing_error = False
            for leg, transform in dags.items():
                with Path(f"{transform.name}_res.json").open("r") as res:
                    try:
                        data[leg] = json.load(res, cls=gufe.tokenization.JSON_HANDLER.decoder)
                    except (json.JSONDecodeError, DimensionalityError) as err:
                        parsing_error = True
                        self.logger.warning(
                            "Error parsing %s (Error: %s)", f"{transform.name}_res.json", err
                        )

            if parsing_error:
                msg += f"\n  {a} -> {b}:  failed (result parsing)"
                results[(a, b)] = _failed_edge(a, b)
                continue

            # Legs
            dat_complex = data["complex"]["estimate"]
            dat_solvent = data["solvent"]["estimate"]

            # Catch failed edges
            if any(leg is None for leg in (dat_complex, dat_solvent)):
                msg += f"\n  {a} -> {b}:  failed (cmpl={dat_complex}, solv={dat_solvent})"
                results[(a, b)] = _failed_edge(a, b)
                continue

            # Compute ddG + error
            ddg = dat_complex - dat_solvent
            complex_err, solvent_err = (
                data["complex"]["uncertainty"],
                data["solvent"]["uncertainty"],
            )
            ddg_err = np.sqrt(complex_err**2 + solvent_err**2 - 2 * complex_err * solvent_err)
            msg += f"\n  {a} -> {b}:  ddG={ddg:4.4f}  err={ddg_err:4.4f}"
            results[(a, b)] = FEPResult(
                ddg=ddg.magnitude,
                ddg_error=ddg_err.magnitude,
                smiles=(isos[a].to_smiles(remove_h=True), isos[b].to_smiles(remove_h=True)),
            )
        self.logger.info(msg)

        # Move results + raw data to dumping location
        if self.dump_to.is_set:
            dump_folder = self.dump_to.value
            dump_folder.mkdir(exist_ok=True)
            for folder in Path().glob("tf-*"):
                if not (dump_folder / folder.name).exists():
                    shutil.move(folder, dump_folder)
            for res_file in Path().glob("*_res.json"):
                if not (dump_folder / res_file.name).exists():
                    shutil.copy(res_file, dump_folder)
            shutil.copy(Path("network.graphml"), dump_folder)

        self.out.send(results)


# 1UYD previously published with Icolos (IcolosData/molecules/1UYD)
@pytest.fixture
def protein_path(shared_datadir: Path) -> Path:
    return shared_datadir / "tnks.pdb"


@pytest.fixture
def ligand_path(shared_datadir: Path) -> Path:
    return shared_datadir / "target.sdf"


@pytest.fixture
def ref_path(shared_datadir: Path) -> Path:
    return shared_datadir / "ref.sdf"


@pytest.fixture
def result_network() -> tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]]:
    isos = {smi: Isomer.from_smiles(smi) for smi in ["C", "CC", "CCC", "CCCO"]}
    mols = [IsomerCollection([iso]) for iso in isos.values()]
    return mols, {
        (isos["C"].inchi, isos["CC"].inchi): FEPResult(smiles=("C", "CC"), ddg=-2.0, ddg_error=0.5),
        (isos["CC"].inchi, isos["CCC"].inchi): FEPResult(
            smiles=("CC", "CCC"), ddg=1.0, ddg_error=0.2
        ),
        (isos["CCC"].inchi, isos["CCCO"].inchi): FEPResult(
            smiles=("CCC", "CCCO"), ddg=0.5, ddg_error=0.8
        ),
        (isos["CCCO"].inchi, isos["CC"].inchi): FEPResult(
            smiles=("CCCO", "CC"), ddg=-1.3, ddg_error=0.1
        ),
    }


@pytest.fixture
def result_network_sub(
    result_network: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
) -> tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]]:
    mols, net = result_network
    c, cc, *_ = mols
    net[(c.molecules[0].inchi, cc.molecules[0].inchi)] = FEPResult(
        smiles=("C", "CC"), ddg=np.nan, ddg_error=0.1
    )
    return mols, net


class TestSuiteOpenFE:
    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    @pytest.mark.needs_node("openrfe")
    @pytest.mark.skip(reason="Bespokefit is not ready yet")
    def test_OpenRFE_bespoke(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "molecule_forcefield": "bespoke",
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_trial(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
            },
        )
        assert Path(
            "tf-rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex/rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_trial_cont(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol1 = Isomer.from_sdf(ligand_path)
        mol2 = Isomer.from_sdf(ligand_path)
        existing = Path("./existing")
        existing.mkdir()
        file = existing / (
            "rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex_res.json"
        )
        file.touch()

        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol1, mol2]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
                "continue_from": existing,
            },
        )
        assert not Path(
            "transforms/rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_trial_cont_map(
        self,
        shared_datadir: Path,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        existing = Path("./existing")
        existing.mkdir()
        file = existing / (
            "rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex_res.json"
        )
        file.touch()
        shutil.copy(shared_datadir / "network.graphml", existing)

        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "existing",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
                "continue_from": existing,
            },
        )
        assert not Path(
            "transforms/rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_trial_dump(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        dump = Path("dump")
        dump.mkdir()

        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
                "dump_to": dump,
            },
        )
        assert len(list(dump.iterdir())) == 3
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    def test_MakeAbsolute(
        self,
        tmp_path: Path,
        test_config: Config,
        result_network: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
    ) -> None:
        ref = Isomer.from_smiles("C")
        mols, data = result_network
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.allclose(isos["C"].get_tag("fep"), -10.0, 0.1)
        assert np.allclose(isos["CC"].get_tag("fep"), -12.0, 0.1)
        assert np.allclose(isos["CCC"].get_tag("fep"), -11.01, 0.1)
        assert np.allclose(isos["CCCO"].get_tag("fep"), -10.69, 0.1)

        ref = Isomer.from_smiles("CC")
        mols, data = result_network
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.allclose(isos["C"].get_tag("fep"), -8.0, 0.1)
        assert np.allclose(isos["CC"].get_tag("fep"), -10.0, 0.1)
        assert np.allclose(isos["CCC"].get_tag("fep"), -9.01, 0.1)
        assert np.allclose(isos["CCCO"].get_tag("fep"), -8.69, 0.1)

        ref = Isomer.from_smiles("CCCCC")
        mols, data = result_network
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.isnan(isos["C"].get_tag("fep"))
        assert np.isnan(isos["CC"].get_tag("fep"))
        assert np.isnan(isos["CCC"].get_tag("fep"))
        assert np.isnan(isos["CCCO"].get_tag("fep"))

    def test_MakeAbsolute_subgraph(
        self,
        tmp_path: Path,
        test_config: Config,
        result_network_sub: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
    ) -> None:
        ref = Isomer.from_smiles("CC")
        mols, data = result_network_sub
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.isnan(isos["C"].get_tag("fep"))
        assert np.allclose(isos["CC"].get_tag("fep"), -10.0, 0.1)
        assert np.allclose(isos["CCC"].get_tag("fep"), -9.01, 0.1)
        assert np.allclose(isos["CCCO"].get_tag("fep"), -8.69, 0.1)

    def test_SaveOpenFEResults(
        self,
        tmp_path: Path,
        test_config: Config,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        data = {
            (ref.inchi, mol.inchi): FEPResult(
                smiles=(ref.to_smiles(), mol.to_smiles()), ddg=-1.0, ddg_error=0.5
            )
        }
        path = tmp_path / "out.csv"
        rig = TestRig(SaveOpenFEResults, config=test_config)
        rig.setup_run(
            inputs={"inp": [data]},
            parameters={"file": path},
        )

        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert row == ["origin", "target", "origin-smiles", "target-smiles", "ddg", "error"]
            row = next(reader)
            assert row[0].startswith("ILBZVTXOJSVUIM")
            assert row[1].startswith("UZYTVPMNMQLENF")
            assert float(row[4]) == -1.0
            assert float(row[5]) == 0.5
