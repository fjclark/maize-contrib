"""Nodes for filtering down isomer collections"""

import copy
from pathlib import Path
from typing import Any, Callable, List, Literal, TypeVar, Annotated, cast
from typing_extensions import assert_never

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter, Flag, Suffix
from maize.utilities.chem import IsomerCollection, Isomer, rmsd as chemrmsd
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


class BestIsomerFilter(Node):
    """
    Filter a list of `IsomerCollection` to retain only the best
    compound according to their score or a user-defined tag.

    """

    inp: Input[List[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filters"""

    score_tag: Parameter[str] = Parameter(optional=True)
    """String to use for ranking"""

    descending: Parameter[bool] = Parameter(default=True)
    """Sort scores descending (lower = better)"""

    def run(self) -> None:
        isomer_collection_list = self.inp.receive()

        isomer_collection_list_c = copy.deepcopy(isomer_collection_list)
        descending = self.descending.value
        if self.score_tag.is_set:
            ranking_tag = self.score_tag.value
        else:
            ranking_tag = None

        for ic_num, ic in enumerate(isomer_collection_list_c):
            best_iso = None
            best_score = np.inf if descending else -np.inf

            for iso_num, isom in enumerate(ic.molecules):
                if ranking_tag is None:
                    this_score = cast(float, isom.get_tag(isom.score_tag))
                else:
                    this_score = cast(float, isom.get_tag(ranking_tag))

                if np.isnan(this_score):
                    this_score = np.inf if descending else -np.inf

                if descending:
                    if this_score < best_score:
                        best_score = this_score
                        best_iso = iso_num
                else:
                    if this_score > best_score:
                        best_score = this_score
                        best_iso = iso_num

            remove_list = []
            for iso_num, isom in enumerate(ic.molecules):
                if not iso_num == best_iso:
                    remove_list.append(isom)
                    self.logger.debug("removing iso %i from ic num %i" % (iso_num, ic_num))

            for isom in remove_list:
                ic.remove_isomer(isom)

        # send result
        self.out.send(isomer_collection_list_c)


# Originally: IsomerCollectionTagFilter
class TagFilter(Node):
    """
    Filter a list of `IsomerCollection` objects by their tags

    """

    inp: Input[List[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filtering"""

    must_have_tags: Parameter[List[str]] = Parameter(default_factory=list)
    """Tags that must be present in output, any value"""

    min_value_tags: Parameter[dict[str, Any]] = Parameter(default_factory=dict)
    """Tags whose numeric value must be >= to a minimum value"""

    max_value_tags: Parameter[dict[str, Any]] = Parameter(default_factory=dict)
    """Tags whose numeric value must be <= to a maximum value"""

    exact_value_tags: Parameter[dict[Any, Any]] = Parameter(default_factory=dict)
    """Tags whose value (any type) must correspond to the a provided key"""

    def run(self) -> None:
        isomer_collection_list = self.inp.receive()

        must_have_tags = self.must_have_tags.value
        min_value_tags = self.min_value_tags.value
        max_value_tags = self.max_value_tags.value
        exact_value_tags = self.exact_value_tags.value

        def _numeric_key_check(key: str, tag_dictionary: dict[str, Any]) -> bool:
            try:
                return not np.isnan(float(tag_dictionary[key]))
            except ValueError:
                return False

        def _generic_key_compare(
            key: str,
            value: Any,
            tag_dictionary: dict[str, Any],
            comparison: Callable[[Any, Any], bool] = lambda x, y: x > y,
        ) -> bool:
            return comparison(tag_dictionary[key], value)

        # this would otherwise modify the input in place
        isomer_collection_list_c = copy.deepcopy(isomer_collection_list)
        self.logger.info("entering filter with %i isomers" % len(isomer_collection_list_c))

        for ic_num, ic in enumerate(isomer_collection_list_c):
            remove_list = []
            for iso_num, isom in enumerate(ic.molecules):
                keep = True
                for tag in must_have_tags:
                    if not isom.has_tag(tag):
                        self.logger.debug(
                            "removing this iso #%i for mol %i for lack of key %s"
                            % (iso_num, ic_num, tag)
                        )
                        keep = False

                # get tags
                tag_dict = isom.tags

                # apply filters
                for tag_filter, enforce_numeric, comparison_op in zip(
                    (min_value_tags, max_value_tags, exact_value_tags),
                    (True, True, False),
                    (lambda x, y: x >= y, lambda x, y: x <= y, lambda x, y: x == y),
                ):
                    # filter
                    for key in tag_filter.keys():
                        if keep and enforce_numeric and not _numeric_key_check(key, tag_dict):
                            self.logger.debug(
                                "removing this iso #%i for mol %i for non-numeric key %s"
                                % (iso_num, ic_num, key)
                            )
                            keep = False
                        if keep and not _generic_key_compare(
                            key, tag_filter[key], tag_dict, comparison=comparison_op
                        ):
                            self.logger.debug(
                                "removing this iso #%i for mol %i, failing with value %s for key %s"
                                % (iso_num, ic_num, str(tag_dict[key]), key)
                            )
                            keep = False
                if not keep:
                    remove_list.append(isom)

            for isom in remove_list:
                ic.remove_isomer(isom)

        # clear out any empty isomer lists
        isomer_collection_list_c = [ic for ic in isomer_collection_list_c if ic.n_isomers > 0]
        self.logger.info("exiting filter with %i isomers" % len(isomer_collection_list_c))

        # send result
        self.out.send(isomer_collection_list_c)


# Originally: IsomerCollectionRankingFilter
class RankingFilter(Node):
    """
    Sorts a list of `IsomerCollection` objects by numeric
    tags and optionally filters to a max number.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filtering"""

    tags_to_rank: Parameter[List[tuple[str, Literal["ascending", "descending"]]]] = Parameter()
    """List of tags (tag, ascending/descending") """

    max_output_length: Parameter[int] = Parameter(optional=True)
    """Return only the top *n* isomer collections after sorting"""

    def run(self) -> None:
        isomer_collection_list = self.inp.receive()

        sorting_instructions = self.tags_to_rank.value

        self.logger.info("entering filter with %i isomer collections" % len(isomer_collection_list))

        index_set = []
        for sorting_instruction in sorting_instructions:
            local_order = np.array(
                [
                    isoc.molecules[0].get_tag(sorting_instruction[0])
                    for isoc in isomer_collection_list
                ]
            )
            if sorting_instruction[1] == "descending":
                local_order *= -1
            index_set.append(local_order)

        final_sort = [
            isomer_collection_list[i]
            for i in sorted(
                range(len(isomer_collection_list)), key=list(zip(*index_set)).__getitem__
            )
        ]

        if self.max_output_length.is_set:
            final_sort = final_sort[0 : self.max_output_length.value]

        self.logger.info("exiting filter with %i isomer collections" % len(final_sort))

        # send result
        self.out.send(final_sort)


# Originally: IsomerFilter
class SMARTSFilter(Node):
    """
    Filter isomers according to occurring or missing
    substructures expressed as lists of SMARTS strings.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules with isomers and conformations to filter"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with filtered isomers"""

    white_list_smarts: Parameter[list[str]] = Parameter(default_factory=list)
    """List of SMARTS that have to be in isomer to pass filter"""

    black_list_smarts: Parameter[list[str]] = Parameter(default_factory=list)
    """List of SMARTS that must not be in isomer to pass filter"""

    def run(self) -> None:
        mols = self.inp.receive()

        for mol in mols:
            n_before = mol.n_isomers
            # remove isomers that don't contain white list SMARTS
            for isomer in mol.molecules[:]:
                for smarts in self.white_list_smarts.value:
                    if not isomer.check_smarts(smarts):
                        mol.remove_isomer(isomer)
                        break

            # remove isomers that contain black list SMARTS
            for isomer in mol.molecules[:]:
                for smarts in self.black_list_smarts.value:
                    if isomer.check_smarts(smarts):
                        mol.remove_isomer(isomer)
                        break

            self.logger.info(f"Remaining isomers in {mol}: {mol.n_isomers}/{n_before}")

        self.out.send(mols)


T_arr_float = TypeVar("T_arr_float", NDArray[np.float32], float)


def score_combine(score1: T_arr_float, score2: T_arr_float, weight: float) -> T_arr_float:
    """Combines two normalized scores as geometric mean"""
    return cast(T_arr_float, np.sqrt(weight * np.square(score1 - 1) + np.square(score2 - 1)))


class RMSDFilter(Node):
    """
    Charge filtering for isomers and RMSD filtering for conformers.

    Only isomers with target charge pass filter. For each isomer, only conformers
    that minmize RMSD to a given reference ligand are considered. If several isomers
    with target charge remain after charge filtering, either the isomer with smallest
    RMSD or lowest docking score pass through the filter. At the end, only one isomer
    with one conformer (or none) per SMILES pass the filter.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules with isomers and conformations (from single SMILES) to filter"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with single isomer and conformer after filtering"""

    inp_ref: Input[Isomer] = Input(optional=True)
    """Reference ligand input"""

    ref_lig: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(optional=True)
    """Path to the reference ligand"""

    target_charge: Parameter[int] = Parameter(default=0)
    """Only isomers with this total charge pass filter"""

    reference_charge_type: Parameter[Literal["ref", "target", "no"]] = Parameter(default="target")
    """
    If 'ref' is given then the charge of the reference ligand is the target charge.
    If 'target' is given, the charge specified under ``target_charge`` is used. If
    'no' is given, every isomer charge is accepted.

    """

    strict_target_charge: Flag = Flag(default=True)
    """
    If true and no isomer with target charge is found, an empty isomer list passes
    the filter. This is useful for RBFE calculations where FEP edges with changes
    in charge are unsuitable. If false and no isomer with target charge is found,
    accept any other isomer charge. This is useful for a standard REINVENT run
    where for each SMILES a conformation is passing the filter.

    """

    isomer_filter: Parameter[Literal["dock", "rmsd", "combo"]] = Parameter(default="dock")
    """
    If after filtering out isomers with wrong charge more than one isomer remain 
    pass isomer with lowest docking score when set to 'dock', pass isomer with
    lowest rmsd when set to 'rmsd' or pass isomer with lowest combined score when
    set to 'combo'.

    """

    conformer_combo_filter: Flag = Flag(default=True)
    """
    If set to 'True', rmsd and docking score are combined to filter the best conformer
    for each isomer. Otherwise, only RMSD is used to find the best conformer.

    """

    mcs_timeout: Parameter[int] = Parameter(default=1)
    """
    Timeout for MCS calculation in seconds. This might need to be increased for certain
    pathogenic molecules, `see discussion here <https://github.com/rdkit/rdkit/issues/2581>`_.

    """

    def run(self) -> None:
        mols = self.inp.receive()

        # load in reference ligand
        if self.inp_ref.ready():
            ref_mol = self.inp_ref.receive()
        else:
            ref_mol = Isomer.from_sdf(self.ref_lig.filepath)

        # determine the target isomer charge
        if self.reference_charge_type.value == "target":
            charge = self.target_charge.value
            self.logger.info("Target charge explicitly given as %s", self.target_charge.value)
        elif self.reference_charge_type.value == "ref":
            charge = ref_mol.charge
            self.logger.info("Target charge from reference ligand is %s", charge)
        else:
            charge = None  # this means that any isomer charge is acceptable
            self.logger.info("Every isomer charge is accepted")

        if self.strict_target_charge.value and charge is not None:
            self.logger.info("Isomers with charges other than target charge won't pass filter")

        for mol in mols:
            # check if there is an isomer with acceptable charge
            # if not, behaviour depends on setting of strict_target_charge
            good_isocharge = any(isomer.charge == charge for isomer in mol.molecules)

            if not good_isocharge and charge is not None:
                self.logger.info("For molecule %s no isomer with target charge found!", mol)
                if not self.strict_target_charge.value:
                    charge = None

            # find best docking score and rmsd of all isomers and conformers in mol
            # needed for a combo score only
            rmsd_iso_min: float = 0.0
            dock_iso_min: float = 0.0
            if self.isomer_filter.value == "combo":
                rmsds = [
                    chemrmsd(isomer, ref_mol, timeout=self.mcs_timeout.value)
                    for isomer in mol.molecules
                ]
                if len(rmsds) > 0:
                    rmsd_iso_min = min(rmsd.min() if rmsd is not None else 0 for rmsd in rmsds)
                if len(mol.molecules) > 0:
                    dock_iso_min = min(
                        isomer.scores.min() if isomer.scores is not None else 0
                        for isomer in mol.molecules
                    )

            # find most suitable isomer
            best_iso_score = np.inf
            best_iso = None
            for isomer in mol.molecules[:]:
                # check if isomer has scores and correct charge
                if (
                    isomer.scores is not None
                    and (isomer.charge == charge or charge is None)
                    and (
                        (rmsd := chemrmsd(isomer, ref_mol, timeout=self.mcs_timeout.value))
                        is not None
                    )
                ):
                    # get best rmsd and docking score for all conformers
                    rmsd_conf_min = np.min(rmsd)
                    dock_conf_min = np.min(isomer.scores)

                    # combine scores if conformer_combo_filter is set
                    if self.conformer_combo_filter.value:
                        conf_score = score_combine(
                            isomer.scores / dock_conf_min,
                            rmsd / rmsd_conf_min,
                            100,
                        )
                    else:
                        conf_score = rmsd

                    # select best conformer
                    min_conf_idx = np.argmin(conf_score)
                    min_rmsd: float = rmsd[min_conf_idx]
                    min_dock: float = isomer.scores[min_conf_idx]
                    min_conf = isomer.conformers[min_conf_idx]

                    # To avoid an isomer with empty conformers it is necessary
                    # to first add the best conformer and then to delete all the previous
                    isomer.clear_conformers()
                    isomer.add_conformer(min_conf)
                    # for _ in range(isomer.n_conformers - 1):
                    #     isomer.remove_conformer(0)
                    isomer.scores = np.array([min_dock])

                    # check which isomer has lowest score
                    if self.isomer_filter.value == "dock":
                        score = min_dock
                    elif self.isomer_filter.value == "rmsd":
                        score = min_rmsd
                    elif self.isomer_filter.value == "combo":
                        # normalizing using best docking score and rmsd
                        # for all isomers and conformers in mol
                        score = score_combine(min_dock / dock_iso_min, min_rmsd / rmsd_iso_min, 100)
                    else:
                        assert_never()

                    if score < best_iso_score:
                        best_iso_score = score
                        best_iso = isomer

                else:
                    # remove unsuitable isomers
                    if isomer.scores is None:
                        self.logger.info(
                            "For isomer %s in molecule %s no docking scores found!",
                            isomer,
                            mol,
                        )
                    mol.remove_isomer(isomer)

            # remove all superflous isomers
            for isomer in mol.molecules[:]:
                if isomer != best_iso:
                    mol.remove_isomer(isomer)

        self.out.send(mols)


@pytest.fixture
def path_ref(shared_datadir: Path) -> Path:
    return shared_datadir / "rmsd-filter-ref.sdf"


@pytest.fixture
def iso_paths(shared_datadir: Path) -> list[Path]:
    return [shared_datadir / "rmsd-filter-iso1.sdf", shared_datadir / "rmsd-filter-iso2.sdf"]


class TestSuiteFilter:
    def test_BestIsomerFilter(self, test_config: Config) -> None:
        rig = TestRig(BestIsomerFilter, config=test_config)

        mols = [
            IsomerCollection([Isomer.from_smiles("C") for i in range(4)]),  # 1
            IsomerCollection([Isomer.from_smiles("C") for i in range(4)]),  # 2
        ]
        score_sets: list[dict[str, list[float | int]]] = [
            {"A": [1, 2, 3, 4], "B": [-1, -2, -3, -4]},
            {"A": [4.5, 3.5, 2.5, 1.5], "B": [-4.5, -3.5, -2.5, -1.5]},
        ]
        mol_counter = 0
        for mol, score_set in zip(mols, score_sets):
            mol_counter += 1
            for key in score_set.keys():
                iso_counter = 0
                for iso, prop_val in zip(mol.molecules, score_set[key]):
                    iso_counter += 1
                    iso.set_tag("iso_id", "-".join([str(mol_counter), str(iso_counter)]))
                    iso.set_tag(key, prop_val)
                    iso.score_tag = "B"

        result = rig.setup_run(
            inputs={"inp": [mols]}, parameters={"score_tag": "A", "descending": False}
        )
        res = result["out"].get()
        assert res is not None
        assert [len(r.molecules) == 1 for r in res]
        assert res[0].molecules[0].get_tag("iso_id") == "1-4"
        assert res[1].molecules[0].get_tag("A") == 4.5

        # now test with score tag
        result = rig.setup_run(inputs={"inp": [mols]}, parameters={"descending": True})
        res = result["out"].get()
        assert res is not None
        assert [len(r.molecules) == 1 for r in res]
        assert res[0].molecules[0].get_tag("iso_id") == "1-4"
        assert res[1].molecules[0].get_tag("B") == -4.5

    def test_TagFilter(self, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("C"),  # 1
            IsomerCollection.from_smiles("CC"),  # 2
            IsomerCollection.from_smiles("CCCC"),  # 3
            IsomerCollection.from_smiles("CCCCC"),  # 4
            IsomerCollection.from_smiles("CCCCCC"),  # 5
            IsomerCollection.from_smiles("CCCCCCC"),  # 6
            IsomerCollection.from_smiles("CCCCCCCC"),  # 7
        ]
        tags_and_values: list[dict[str, str | int | float]] = [
            {"B": 1, "C": 1},  # missing tag A  1
            {"A": 1, "B": "foo", "C": "bar"},  # non-numeric tag B 2
            {"A": 1, "B": np.nan, "C": "bar"},  # non-numeric tag C 3
            {"A": 2, "B": 1, "C": "bar"},  # A is too high 4
            {"A": 1, "B": -1, "C": "bar"},  # B is too low 5
            {"A": 1, "B": 1, "C": "foo"},  # C is not bar 6
            {"A": 1, "B": 1, "C": "bar"},  # should pass 7
        ]
        counter = 0
        for mol, tag_dict in zip(mols, tags_and_values):
            counter += 1
            for iso in mol.molecules:
                for tag in tag_dict.keys():
                    iso.set_tag(tag, tag_dict[tag])
                    iso.name = f"molecule {counter}"

        rig = TestRig(TagFilter, config=test_config)
        result = rig.setup_run(
            inputs={"inp": [mols]},
            parameters={
                "must_have_tags": ["A", "B"],
                "min_value_tags": {"A": 1, "B": 1},
                "max_value_tags": {"A": 1, "B": 1},
                "exact_value_tags": {"C": "bar"},
            },
        )
        res = result["out"].get()
        assert res is not None
        assert len(res) == 1
        assert res[0].molecules[0].name == "molecule 7"

    def test_RankingFilter(self, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("C"),  # 1
            IsomerCollection.from_smiles("CC"),  # 2
            IsomerCollection.from_smiles("CCCC"),  # 3
            IsomerCollection.from_smiles("CCCCC"),  # 4
            IsomerCollection.from_smiles("CCCCCC"),  # 5
            IsomerCollection.from_smiles("CCCCCCC"),  # 6
            IsomerCollection.from_smiles("CCCCCCCC"),  # 7
        ]
        tags_and_values = [
            {"A": 1, "B": -1},  # 1
            {"A": 2, "B": -2},  # 2
            {"A": 3, "B": -7},  # 3
            {"A": 3, "B": -6},  # 4
            {"A": 3, "B": -5},  # 5
            {"A": 2, "B": 4},  # 6
            {"A": 0, "B": 3},  # 7
        ]
        counter = 0
        for mol, tag_dict in zip(mols, tags_and_values):
            counter += 1
            for iso in mol.molecules:
                for tag in tag_dict.keys():
                    iso.set_tag(tag, tag_dict[tag])
                    iso.name = f"molecule {counter}"
        rig = TestRig(RankingFilter, config=test_config)
        rig = TestRig(RankingFilter, config=test_config)
        result = rig.setup_run(
            inputs={"inp": [mols]},
            parameters={
                "tags_to_rank": [("A", "descending"), ("B", "ascending")],
                "max_output_length": 3,
            },
        )
        res = result["out"].get()
        assert res is not None
        assert all([iso.molecules[0].get_tag("A") == 3 for iso in res])
        assert all([iso.molecules[0].get_tag("B") < -4 for iso in res])
        assert len(res) == 3

    def test_RMSDFilter(self, path_ref: Path, iso_paths: list[Path]) -> None:
        """Test RMSD_Filter"""

        iso_list = [Isomer.from_sdf(path, read_conformers=True) for path in iso_paths]
        for iso in iso_list:
            iso.score_tag = "energy"

        rig = TestRig(RMSDFilter)
        params = {
            "ref_lig": Path(path_ref),
            "reference_charge_type": "ref",
            "strict_target_charge": False,
        }
        res = rig.setup_run(parameters=params, inputs={"inp": [[IsomerCollection(iso_list)]]})
        filtered = res["out"].get()

        assert filtered is not None
        assert filtered[0].molecules[0].scored
        assert filtered[0].molecules[0].n_conformers == 1
        assert len(filtered[0].molecules) == 1

        ref = Isomer.from_sdf(path_ref)
        params = {
            "reference_charge_type": "ref",
            "strict_target_charge": True,
        }
        res = rig.setup_run(
            parameters=params, inputs={"inp": [[IsomerCollection(iso_list)]], "inp_ref": [ref]}
        )
        filtered = res["out"].get()

        assert filtered is not None
        assert len(filtered[0].molecules) == 0

        params = {
            "reference_charge_type": "ref",
            "strict_target_charge": True,
            "isomer_filter": "rmsd",
            "conformer_combo_filter": False,
        }
        res = rig.setup_run(
            parameters=params, inputs={"inp": [[IsomerCollection(iso_list)]], "inp_ref": [ref]}
        )
        filtered = res["out"].get()

        assert filtered is not None
        assert len(filtered[0].molecules) == 0

        params = {
            "reference_charge_type": "ref",
            "strict_target_charge": False,
            "isomer_filter": "rmsd",
            "conformer_combo_filter": False,
        }
        res = rig.setup_run(
            parameters=params, inputs={"inp": [[IsomerCollection(iso_list)]], "inp_ref": [ref]}
        )
        filtered = res["out"].get()

        assert filtered is not None
        assert filtered[0].molecules[0].scored
        assert filtered[0].molecules[0].n_conformers == 1
        assert len(filtered[0].molecules) == 1

        params = {
            "reference_charge_type": "no",
            "strict_target_charge": False,
            "isomer_filter": "rmsd",
            "conformer_combo_filter": True,
        }
        res = rig.setup_run(
            parameters=params, inputs={"inp": [[IsomerCollection(iso_list)]], "inp_ref": [ref]}
        )
        filtered = res["out"].get()

        assert filtered is not None
        assert filtered[0].molecules[0].scored
        assert filtered[0].molecules[0].n_conformers == 1
        assert len(filtered[0].molecules) == 1
