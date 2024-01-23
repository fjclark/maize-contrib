"""Free energy method subgraphs and workflows"""

from maize.core.workflow import Workflow, expose
from maize.steps.io import Void
from maize.steps.plumbing import Copy
from maize.steps.mai.docking import AutoDockGPU, RMSDFilter
from maize.steps.mai.md_sim import OpenRFE, SaveOpenFEResults
from maize.steps.mai.molecule import Gypsum, LoadSmiles, Mol2Isomers, LoadMolecule
from maize.utilities.chem import Isomer


@expose
def openfe_rbfe() -> Workflow:
    """Run an RBFE campaign using OpenFE"""
    flow = Workflow(name="openfe-rbfe")
    load = flow.add(LoadSmiles)
    embe = flow.add(Gypsum)
    dock = flow.add(AutoDockGPU)
    void = flow.add(Void)
    filt = flow.add(RMSDFilter)
    flat = flow.add(Mol2Isomers)
    rbfe = flow.add(OpenRFE)
    save = flow.add(SaveOpenFEResults)

    load_ref = flow.add(LoadMolecule)
    copy_ref = flow.add(Copy[Isomer])

    flow.connect_all(
        (load.out, embe.inp),
        (embe.out, dock.inp),
        (dock.out, filt.inp),
        (filt.out, flat.inp),
        (flat.out, rbfe.inp),
        (rbfe.out, save.inp),
    )

    flow.connect_all(
        (load_ref.out, copy_ref.inp),
        (copy_ref.out, filt.inp_ref),
        (copy_ref.out, rbfe.inp_ref),
        (dock.out_scores, void.inp),
    )

    filt.reference_charge_type.set("ref")
    filt.isomer_filter.set("rmsd")
    filt.conformer_combo_filter.set(False)
    rbfe.mapping.set("star")

    flow.combine_parameters(load.path, name="smiles")
    flow.combine_parameters(load_ref.path, name="reference")
    flow.combine_parameters(save.file, name="output")
    flow.combine_parameters(dock.grid_file, name="grid")
    flow.combine_parameters(rbfe.inp_protein, name="protein")
    flow.map(
        embe.n_variants,
        embe.ph_range,
        embe.thoroughness,
        rbfe.n_jobs,
        rbfe.n_lambda,
        rbfe.n_replicas,
        rbfe.n_repeats,
        rbfe.equilibration_length,
        rbfe.production_length,
    )
    return flow
