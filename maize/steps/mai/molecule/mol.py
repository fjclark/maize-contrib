"""Molecule handling steps"""

# pylint: disable=import-outside-toplevel, import-error

from collections import defaultdict
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import random
from typing import Annotated, Any, Callable, List, Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix, Flag, MultiInput
from maize.utilities.chem.chem import ChemistryException, ValidRDKitTagType
from maize.utilities.chem import (
    IsomerCollection,
    Isomer,
    dict2lib,
    lib2dict,
    load_sdf_library,
    nested_merge,
)
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


NumericType = int | float | np.number
T = TypeVar("T")


class Smiles2Molecules(Node):
    """
    Converts SMILES codes into a set of molecules with distinct
    isomers and conformers using the RDKit embedding functionality.

    See Also
    --------
    :class:`~maize.steps.mai.molecule.Gypsum` :
        A more advanced procedure for producing different
        isomers and high-energy conformers.

    """

    inp: Input[list[str]] = Input()
    """SMILES input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    n_conformers: Parameter[int] = Parameter(default=1)
    """Number of conformers to generate"""

    n_variants: Parameter[int] = Parameter(default=1)
    """Maximum number of stereoisomers to generate"""

    embed: Flag = Flag(default=True)
    """
    Whether to create embeddings for the molecule. May not be
    required if passing it on to another embedding system.

    """

    def run(self) -> None:
        smiles = self.inp.receive()
        mols: list[IsomerCollection] = []
        n_variants = self.n_variants.value if self.embed.value else 0
        for i, smi in enumerate(smiles):
            self.logger.info("Embedding %s/%s ('%s')", i + 1, len(smiles), smi.strip())
            try:
                mol = IsomerCollection.from_smiles(smi, max_isomers=n_variants)
                if self.embed.value:
                    mol.embed(self.n_conformers.value)
            except ChemistryException as err:
                self.logger.warning("Unable to create '%s' (%s), not sanitizing...", smi, err)
                if "SMILES Parse Error" in err.args[0]:
                    mol = IsomerCollection([])
                    mol.smiles = smi
                else:
                    mol = IsomerCollection.from_smiles(smi, max_isomers=0, sanitize=False)
            mols.append(mol)
        self.out.send(mols)


class Mol2Isomers(Node):
    """Convert a list of `IsomerCollection` to a list of `Isomer`"""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[Isomer]] = Output()
    """Isomer output"""

    def run(self) -> None:
        mols = self.inp.receive()
        self.out.send(list(itertools.chain(*[mol.molecules for mol in mols])))


class Isomers2Mol(Node):

    """Convert a list of `Isomer` to a list of `IsomerCollection`"""

    inp: Input[list[Isomer]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Isomer output"""

    combine: Parameter[bool] = Parameter(default=False)
    """Should all the isomers be combined into a single `IsomerCollection`"""

    def run(self) -> None:
        mols = self.inp.receive()
        if self.combine.value:
            iso_collection = IsomerCollection(mols)
            self.out.send([iso_collection])
        else:
            self.out.send([IsomerCollection([mol]) for mol in mols])


class SaveMolecule(Node):
    """Save a molecule to an SDF file."""

    inp: Input[IsomerCollection] = Input()
    """Molecule input"""

    path: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(exist_required=False)
    """SDF output destination"""

    def run(self) -> None:
        mol = self.inp.receive()
        self.logger.info("Received '%s'", mol)
        mol.to_sdf(self.path.value)


class LoadMolecule(Node):
    """Load a molecule from an SDF or MAE file."""

    out: Output[Isomer] = Output()
    """Isomer output"""

    path: Input[Annotated[Path, Suffix("sdf", "mae")]] = Input(mode="copy")
    """Path to the SDF or MAE file"""

    def run(self) -> None:
        input_path = self.path.receive()
        if input_path.suffix == ".mae":
            mol = IsomerCollection.from_mae(input_path)
        elif input_path.suffix == ".sdf":
            mol = IsomerCollection.from_sdf(input_path)
        else:
            raise ValueError("incorrect filetype %s" % str(input_path.suffix))
        self.out.send(mol.molecules[0])


class LoadSmiles(Node):
    """Load SMILES codes from a ``.smi`` file."""

    path: FileParameter[Annotated[Path, Suffix("smi")]] = FileParameter()
    """SMILES file input"""

    out: Output[list[str]] = Output()
    """SMILES output"""

    sample: Parameter[int] = Parameter(optional=True)
    """Take a sample of SMILES"""

    def run(self) -> None:
        with self.path.filepath.open() as file:
            smiles = [smi.strip("\n") for smi in file.readlines()]
            if self.sample.is_set:
                smiles = random.choices(smiles, k=self.sample.value)
            self.out.send(smiles)


class ExtractTag(Node):
    """Pull a tag from an Isomer"""

    inp: Input[Isomer] = Input()
    """A isomer to extract tag from"""

    out: Output[ValidRDKitTagType] = Output()
    """value of the tag"""

    tag_to_extract: Parameter[str] = Parameter(optional=True)
    """tag to export, will use score_tag by default"""

    def run(self) -> None:
        isom = self.inp.receive()

        if self.tag_to_extract.is_set:
            extract_tag = self.tag_to_extract.value
        else:
            extract_tag = isom.score_tag

        if not extract_tag or not isom.has_tag(extract_tag):
            self.logger.debug("provided isomer does not have tag %s" % extract_tag)
            self.out.send(np.nan)
        else:
            self.out.send(isom.get_tag(extract_tag))


class ToSmiles(Node):
    """transform an isomer or IsomerCollection (or list thereof) to a list of SMILES"""

    inp: Input[Isomer | IsomerCollection | List[IsomerCollection] | List[Isomer]] = Input()
    """SMILES output"""

    out: Output[List[str]] = Output()
    """SMILES output"""

    def run(self) -> None:
        def _liststrip(maybe_list: list[T] | T) -> T:
            # need this uglyness to handle
            # both list of isomer and list of isomer comp
            if isinstance(maybe_list, list):
                return maybe_list[0]
            else:
                return maybe_list

        input_data = self.inp.receive()
        smiles: list[str] | str
        if isinstance(input_data, list):  # need to iteratively build
            smiles = [_liststrip(iso.to_smiles()) for iso in input_data]
        else:
            smiles = input_data.to_smiles()
        if isinstance(smiles, str):  # catch the case where used with single isomer
            smiles = [smiles]
        self.logger.info("sending %i smiles: %s" % (len(smiles), " ".join(smiles)))

        self.out.send(smiles)


class SaveLibrary(Node):
    """Save a list of molecules to multiple SDF files."""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule library input"""

    base_path: FileParameter[Path] = FileParameter(exist_required=False)
    """Base output file path name without a suffix, i.e. /path/to/output"""

    tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out"""

    def run(self) -> None:
        mols = self.inp.receive()
        base = self.base_path.value

        for i, mol in enumerate(mols):
            file = base.with_name(f"{base.name}{i}.sdf")
            if self.tags.is_set:
                # this if statement is better than using mol[0].tags
                # as a default since it would support ragged tags
                mol.to_sdf(file, tags=self.tags.value)
            else:
                mol.to_sdf(file)


class SaveScores(Node):
    """Save VINA Scores to a JSON file."""

    inp: Input[NDArray[np.float32]] = Input()
    """Molecule input"""

    path: FileParameter[Annotated[Path, Suffix("json")]] = FileParameter(exist_required=False)
    """JSON output destination"""

    def run(self) -> None:
        scores = self.inp.receive()
        self.logger.info(f"Received #{len(scores):d} scores")
        with open(self.path.value, "w") as f:
            json.dump(list(scores), f)


class LoadLibrary(Node):
    """Load a small molecule library from an SDF file"""

    path: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter()
    """Input SDF file"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output, each entry in the SDF is parsed as a separate molecule"""

    score_tag: Parameter[str] = Parameter(optional=True)
    """SDF tag used to set scores for the loaded library"""

    def run(self) -> None:
        mols = load_sdf_library(self.path.filepath, split_strategy="none")
        if self.score_tag.is_set:
            for mol in mols:
                mol.set_score_from_tag(self.score_tag.value)
        self.out.send(mols)


class CombineScoredMolecules(Node):
    """Combine molecules scored using different methods together"""

    inp: MultiInput[list[IsomerCollection]] = MultiInput()
    """Multiple scored molecule inputs"""

    out: Output[list[IsomerCollection]] = Output()
    """Combined scores"""

    aggregator: Parameter[Literal["min", "max", "mean"]] = Parameter(default="min")
    """Aggregation function to use for picking the best conformer score"""

    isomer_merging_tag: Parameter[str] = Parameter(optional=True)
    """Tag to use to merge multiple isomers into an `IsomerCollection`"""

    # I believe overloaded function are not currently typable correctly
    AGGREGATORS: dict[str, Callable[[Any], np.float32]] = {
        "min": min,
        "max": max,
        "mean": np.mean,
    }

    @staticmethod
    def _consolidate_isomers(iso_a: Isomer, iso_b: Isomer) -> Isomer:
        for key, tag in iso_a.tags.items():
            if key not in iso_b.tags:
                iso_b.set_tag(key, tag)
        return iso_b

    @staticmethod
    def _add_prefix_tags(iso: Isomer, prefix_tag: str) -> None:
        existing_tag = iso.tags.keys()
        for tag in existing_tag:
            if not tag == iso.score_tag:
                iso.set_tag("-".join([prefix_tag, tag]), iso.get_tag(tag))
                iso.remove_tag(tag)

    def run(self) -> None:
        agg = self.AGGREGATORS[self.aggregator.value]
        mol_colls = []
        self.logger.debug("Ready to receive...")
        for i, inp in enumerate(self.inp):
            self.logger.debug("Attempting to receive batch %s", i)
            mols = inp.receive()
            self.logger.info(
                "Received %s from %s", mols, mols[0].molecules[0].get_tag("origin", "N/A")
            )
            mol_colls.append(mols)

        # Make sure each score entry is unique depending on its origin
        per_collection_best_scores: dict[str, NDArray[np.float32]] = defaultdict(
            lambda: np.full(len(mol_colls), np.nan)
        )

        # In a situation in which we receive the same IsomerCollections from multiple branches,
        # but each branch has done its own filtering, IsomerCollections may contain different
        # Isomers, causing a misclassification. We can instead use a specific tag (if available)
        # to perform this merging correctly for us.
        if self.isomer_merging_tag.is_set:
            merging_tag = self.isomer_merging_tag.value
        else:
            merging_tag = None

        new_score_tags = []

        for i, mols in enumerate(mol_colls):  # for each upstream
            for mol in mols:
                for iso in mol.molecules:
                    iso.uniquify_tags(
                        fallback=f"upstream-node-{i}",
                        exclude=[merging_tag] if merging_tag is not None else None,
                    )
                    new_score_tags.append(iso.score_tag)
                    per_collection_best_scores[iso.inchi][i] = agg(iso.scores)
                    iso.set_tag(f"{iso.score_tag}-best", per_collection_best_scores[iso.inchi][i])

        master = nested_merge(
            *[lib2dict(mols, tag=merging_tag) for mols in mol_colls],
            consolidator=self._consolidate_isomers,
        )

        for isos in master.values():
            for iso in isos.values():
                iso.set_tag("best-scores", per_collection_best_scores[iso.inchi])
                iso.set_tag(
                    iso.score_tag,
                    np.fromiter(
                        (
                            agg(arr)
                            for arr in zip(
                                *[np.atleast_1d(iso.get_tag(c, [np.nan])) for c in new_score_tags]
                            )
                        ),
                        np.float32,
                    ),
                )

        self.out.send(dict2lib(master))


class LibraryFromCSV(Node):
    """
    convert a csv file into an isomer collection with columns added as tags

    """

    inp: Input[Annotated[Path, Suffix("csv")]] = Input()
    """csv file with the molecules as SMILES in a column"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with single isomer and conformer after input"""

    smiles_column: Parameter[str] = Parameter(default="SMILES")
    """Name of column with structures, default is SMILES"""

    name_column: Parameter[str] = Parameter(optional=True)
    """Name of column to use as a name """

    score_column: Parameter[str] = Parameter(optional=True)
    """Name of column to use as top-level scores """

    def run(self) -> None:
        input_file = self.inp.receive()
        smiles_column = self.smiles_column.value
        if self.name_column.is_set:
            name_column = self.name_column.value
        else:
            name_column = None
        if self.score_column.is_set:
            score_column = self.score_column.value
        else:
            score_column = None

        self.logger.info("Reading %s" % str(input_file.as_posix()))

        df = pd.read_csv(input_file.as_posix())

        self.logger.info("Read %i rows and %i columns" % (df.shape[0], df.shape[1]))

        if smiles_column not in df.columns:
            raise ValueError(f"smiles column {smiles_column} not in columns {df.columns}")

        if name_column and name_column not in df.columns:
            raise ValueError(f"name column {name_column} not in columns {df.columns}")

        if score_column and score_column not in df.columns:
            raise ValueError(f"name column {score_column} not in columns {df.columns}")

        skipped_rows = []
        isomer_collections = []

        for row_ind, row in df.iterrows():
            try:
                isom = Isomer.from_smiles(row[smiles_column])

                if name_column in df.columns:
                    isom.name = row[name_column]
                else:
                    isom.name = input_file.stem + "entry-" + str(row_ind)
                for col in df.columns:
                    if score_column and col == score_column:
                        isom.scores = np.array([row[col]])
                    isom.set_tag(col, row[col])
                ic = IsomerCollection([isom])
                isomer_collections.append(ic)

            except (ChemistryException, TypeError):
                self.logger.debug(
                    "failed read at line %i with smiles %s" % (row_ind, str(row[smiles_column]))
                )
                skipped_rows.append(row_ind)

        self.logger.info("loaded %i rows" % len(isomer_collections))
        if len(skipped_rows):
            self.logger.info("skipped %i rows due to read fails" % len(skipped_rows))

        self.out.send(isomer_collections)


class SaveCSV(Node):
    """Save a library to a CSV file"""

    inp: Input[list[IsomerCollection]] = Input()
    """Library input"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV file"""

    tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out"""

    collections_only: Flag = Flag(default=False)
    """
    Whether to write a single `IsomerCollection` per row,
    instead of all contained `Isomer` instances

    """

    def run(self) -> None:
        mols = self.inp.receive()
        tags = (
            set(self.tags.value)
            if self.tags.is_set
            else set(itertools.chain(*(iso.tags.keys() for mol in mols for iso in mol.molecules)))
        )
        with self.file.filepath.open("w") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["smiles", *tags])
            for mol in mols:
                if self.collections_only.value:
                    smiles = mol.smiles or mol.molecules[0].to_smiles(remove_h=True)
                    all_fields: dict[str, Any] = {tag: None for tag in tags}

                    # The first isomer should overwrite all others
                    for iso in reversed(mol.molecules):
                        for tag in tags:
                            val = iso.get_tag(tag, "")
                            if all_fields[tag] is None:
                                all_fields[tag] = val

                    writer.writerow([smiles, *all_fields.values()])
                    continue

                for iso in mol.molecules:
                    fields = (iso.get_tag(tag, "") for tag in tags)
                    writer.writerow(
                        itertools.chain([mol.smiles or iso.to_smiles(remove_h=True)], fields)
                    )


class BatchSaveCSV(Node):
    """Save library batches to a CSV file"""

    inp: Input[list[IsomerCollection]] = Input()
    """Library input"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV file"""

    tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out"""

    n_batches: Parameter[int] = Parameter()
    """Number of batches to expect"""

    @staticmethod
    def _write_batch(writer: Any, mols: list[IsomerCollection], tags: list[str]) -> None:
        for mol in mols:
            for iso in mol.molecules:
                # We use some internal RDKit functionality here for performance reasons, we
                # can get away with it because we don't need any kind of type conversion as
                # we're immediately converting everything to a string anyway
                iso_tag_names = set(iso._molecule.GetPropNames())
                fields = (
                    iso._molecule.GetProp(tag) if tag in iso_tag_names else None for tag in tags
                )
                writer.writerow(
                    itertools.chain([mol.smiles or iso.to_smiles(remove_h=True)], fields)
                )

    def run(self) -> None:
        mols = self.inp.receive()
        if self.tags.is_set:
            tags = self.tags.value
        else:
            tags = list(
                set(itertools.chain(*(iso.tags.keys() for mol in mols for iso in mol.molecules)))
            )

        with self.file.filepath.open("a") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["smiles", *tags])
            self._write_batch(writer, mols, tags)

        for batch_idx in range(self.n_batches.value - 1):
            self.logger.info("Waiting for batch %s", batch_idx + 1)
            mols = self.inp.receive()
            with self.file.filepath.open("a") as out:
                writer = csv.writer(out, delimiter=",")
                self.logger.info("Writing batch %s to %s", batch_idx, self.file.filepath.as_posix())
                self._write_batch(writer, mols, tags)


class TestSuiteMol:
    def test_Smiles2Molecules(self, test_config: Config) -> None:
        rig = TestRig(Smiles2Molecules, config=test_config)
        smiles = ["Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"]
        res = rig.setup_run(
            inputs={"inp": [smiles]}, parameters={"n_conformers": 2, "n_isomers": 2}
        )
        raw = res["out"].get()
        assert raw is not None
        mol = raw[0]
        assert mol.n_isomers <= 2
        assert not mol.scored
        assert mol.molecules[0].n_conformers == 2
        assert mol.molecules[0].charge == 0
        assert mol.molecules[0].n_atoms == 44

    def test_SaveMolecule(self, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveMolecule, config=test_config)
        mol = IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC")
        rig.setup_run(inputs={"inp": mol}, parameters={"path": tmp_path / "file.sdf"})
        assert (tmp_path / "file.sdf").exists()

    def test_LoadSmiles(self, shared_datadir: Path, test_config: Config) -> None:
        rig = TestRig(LoadSmiles, config=test_config)
        res = rig.setup_run(parameters={"path": shared_datadir / "test.smi"})
        mol = res["out"].get()
        assert mol is not None
        assert len(mol) == 1
        assert mol[0] == "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"

    def test_LoadLibrary(self, shared_datadir: Path, test_config: Config) -> None:
        rig = TestRig(LoadLibrary, config=test_config)
        res = rig.setup_run(parameters={"path": shared_datadir / "1UYD_ligands.sdf"})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3

    def test_SaveLibrary(self, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveLibrary, config=test_config)
        mols = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"),
        ]
        base = tmp_path / "mol"
        rig.setup_run(inputs={"inp": [mols]}, parameters={"base_path": base})
        assert base.with_name("mol0.sdf").exists()
        assert base.with_name("mol1.sdf").exists()

    def test_CombineScoredMolecules(self, test_config: Config) -> None:
        rig = TestRig(CombineScoredMolecules, config=test_config)
        mols1 = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"),
        ]
        mols2 = deepcopy(mols1)
        for i, mols in enumerate((mols1, mols2)):
            for mol in mols:
                mol.embed(n_conformers=4)
                for iso in mol.molecules:
                    iso.set_tag("origin", f"node-{i}")
                    iso.set_tag("score", i * np.array([1, 2, 3, 4]))
                    iso.score_tag = "score"

        res = rig.setup_run(inputs={"inp": [[mols1], [mols2]]})
        comb = res["out"].get()
        assert comb is not None
        assert len(comb[0].molecules[0].scores) == 4
        assert np.allclose(comb[0].molecules[0].get_tag("best-scores"), [0.0, 1.0])
        assert np.allclose(comb[0].molecules[0].get_tag("node-0-score-best"), 0.0)
        assert np.allclose(comb[0].molecules[0].get_tag("node-1-score-best"), 1.0)

        rig = TestRig(CombineScoredMolecules, config=test_config)
        mols1 = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"),
        ]
        mols2 = deepcopy(mols1)

        for i, mols in enumerate((mols1, mols2)):
            for mol in mols:
                mol.embed(n_conformers=4)
                for iso in mol.molecules:
                    if not i:
                        iso.set_tag("origin", "node-blue")
                    iso.set_tag("to-remove", "yes")
                    iso.set_tag("score", i * np.array([1, 2, 3, 4]))
                    iso.score_tag = "score"

        res = rig.setup_run(inputs={"inp": [[mols1], [mols2]]}, parameters={"aggregator": "max"})
        comb = res["out"].get()
        assert comb is not None
        assert len(comb[0].molecules[0].scores) == 4
        assert np.allclose(comb[0].molecules[0].get_tag("best-scores"), [0.0, 4.0])
        assert np.allclose(comb[0].molecules[0].scores, [1, 2, 3, 4])
        assert np.allclose(comb[0].molecules[0].get_tag("node-blue-score-best"), 0.0)
        assert np.allclose(comb[1].molecules[0].get_tag("upstream-node-1-score-best"), 4.0)
        assert "to-remove" not in comb[1].molecules[0].tags

    def test_SaveCSV(self, tmp_path: Path, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"),
        ]
        for mol in mols:
            mol.embed(n_conformers=4)
            for iso in mol.molecules:
                iso.set_tag("origin", "node-0")
                iso.set_tag("score", np.array([1, 2, 3, 4]))
                iso.score_tag = "score"

        path = tmp_path / "test.csv"
        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(
            inputs={"inp": [mols]}, parameters={"file": path, "tags": ["origin", "score"]}
        )
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert set(row) == {"smiles", "origin", "score"}
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(inputs={"inp": [mols]}, parameters={"file": path})
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert set(row) == {"smiles", "origin", "score"}
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(
            inputs={"inp": [mols]},
            parameters={"file": path, "collections_only": True, "tags": ["origin", "score"]},
        )
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert set(row) == {"smiles", "origin", "score"}
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

    def test_BatchSaveCSV(self, tmp_path: Path, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OCC)cc3"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OCCC)cc3"),
        ]
        for mol in mols:
            mol.embed(n_conformers=4)
            for iso in mol.molecules:
                iso.set_tag("origin", "node-0")
                iso.set_tag("score", np.array([1, 2, 3, 4]))
                iso.score_tag = "score"

        path = tmp_path / "test.csv"
        rig = TestRig(BatchSaveCSV, config=test_config)
        rig.setup_run(
            inputs={"inp": [mols[:2], mols[2:]]},
            parameters={"file": path, "n_batches": 2, "tags": ["origin", "score"]},
        )
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert row == ["smiles", "origin", "score"]
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert row[1] == "node-0"
            assert row[2] == "[1, 2, 3, 4]"
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert row[1] == "node-0"
            assert row[2] == "[1, 2, 3, 4]"
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert row[1] == "node-0"
            assert row[2] == "[1, 2, 3, 4]"
