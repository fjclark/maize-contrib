"""Nodes for tagging isomers based on molecular properties"""

from pathlib import Path
from random import shuffle
from typing import Any, Callable, Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.chem import IsomerCollection, Isomer, rmsd as chemrmsd
from maize.utilities.testing import TestRig


class TagIndex(Node):
    """Tag each molecule with it's index in the list to allow sorting and re-merging operations"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    tag: Parameter[str] = Parameter(default="idx")
    """The tag to use for indexing"""

    def run(self) -> None:
        mols = self.inp.receive()
        for i, mol in enumerate(mols):
            mol.set_tag(self.tag.value, i)
        self.out.send(mols)


class LogTags(Node):
    """Log the value of a tag for a set of molecules"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules"""

    tag: Parameter[str] = Parameter()
    """The tag to use for logging"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            val = mol.get_tag(self.tag.value, default="")
            self.logger.info("Molecule '%s', %s = %s", mol.smiles, self.tag.value, val)
        self.out.send(mols)


SortableRDKitTagType = bool | int | float | str


class SortByTag(Node):
    """Sort a list of `IsomerCollection` based on a tag"""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be sorted"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule list output"""

    tag: Parameter[str] = Parameter(default="idx")
    """The tag to use for sorting"""

    reverse: Flag = Flag(default=False)
    """Whether to use reverse order"""

    def run(self) -> None:
        mols = self.inp.receive()
        mols.sort(
            key=lambda mol: cast(SortableRDKitTagType, mol.get_tag(self.tag.value)),
            reverse=self.reverse.value,
        )
        self.out.send(mols)


class ExtractTag(Node):
    """
    Extract a specific numeric tag from molecules. The output is guaranteed
    to have the same length and ordering as the input molecules.

    """

    AGG: dict[str, Callable[[Sequence[Any]], float]] = {
        "min": min,
        "max": max,
        "mean": np.mean,
        "first": lambda arr: arr[0],
        "last": lambda arr: arr[-1],
    }

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be sorted"""

    out: Output[NDArray[np.float32]] = Output()
    """Tag output"""

    tag: Parameter[str] = Parameter()
    """The tag to use for sorting"""

    agg: Parameter[Literal["min", "max", "mean", "first", "last"]] = Parameter(default="mean")
    """How to aggregate values across isomers"""

    def run(self) -> None:
        mols = self.inp.receive()
        agg = self.AGG[self.agg.value]
        key = self.tag.value
        outputs: list[float] = []
        for mol in mols:
            vals = [
                float(cast(SortableRDKitTagType, iso.get_tag(key)))
                for iso in mol.molecules
                if iso.has_tag(key)
            ]
            outputs.append(agg(vals) if vals else np.nan)

        self.out.send(np.array(outputs))


class ExtractScores(Node):
    """Extract scores from molecules"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[NDArray[np.float32]] = Output()
    """List of molecules with RMSD tags"""

    def run(self) -> None:
        mols = self.inp.receive()
        scores = np.array([mol.best_score for mol in mols])
        self.out.send(scores)


class RMSD(Node):
    """Calculates RMSDs to a reference molecule"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    inp_ref: Input[Isomer] = Input(cached=True)
    """Reference isomer to compute RMSD to"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with RMSD tags"""

    def run(self) -> None:
        mols = self.inp.receive()
        ref = self.inp_ref.receive()
        for mol in mols:
            if not mol.molecules:
                mol.set_tag("rmsd", np.inf)
            for iso in mol.molecules:
                rmsds = chemrmsd(iso, ref, timeout=5)
                if rmsds is None:
                    continue

                iso.set_tag("rmsd", min(rmsds))
                for rmsd, conf in zip(rmsds, iso.conformers):
                    conf.set_tag("rmsd", rmsd)
        self.out.send(mols)


@pytest.fixture
def tagged_mols() -> list[IsomerCollection]:
    mols = [
        IsomerCollection.from_smiles("CCC"),
        IsomerCollection.from_smiles("CCCC"),
    ]
    for i, mol in enumerate(mols):
        mol.embed()
        for iso in mol.molecules:
            iso.set_tag("score", -i - 0.5)
            iso.score_tag = "score"
    return mols


@pytest.fixture
def indexed_mols() -> list[IsomerCollection]:
    mols = [
        IsomerCollection.from_smiles("CC"),
        IsomerCollection.from_smiles("CCC"),
        IsomerCollection.from_smiles("CCCC"),
    ]
    for i, mol in enumerate(mols):
        mol.embed()
        for iso in mol.molecules:
            iso.set_tag("idx", i)
    return mols


@pytest.fixture
def path_ref(shared_datadir: Path) -> Path:
    return shared_datadir / "rmsd-filter-ref.sdf"


@pytest.fixture
def iso_paths(shared_datadir: Path) -> list[Path]:
    return [shared_datadir / "rmsd-filter-iso1.sdf", shared_datadir / "rmsd-filter-iso2.sdf"]


class TestSuiteTaggers:
    def test_TagIndex(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(TagIndex)
        res = rig.setup_run(inputs={"inp": [tagged_mols]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].get_tag("idx") == 0
        assert mols[1].molecules[0].get_tag("idx") == 1

    def test_SortMolecules(self, indexed_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SortByTag)
        shuffle(indexed_mols)
        res = rig.setup_run(inputs={"inp": [indexed_mols]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3
        for i in range(3):
            assert mols[i].molecules[0].get_tag("idx") == i

    def test_SortMolecules_reverse(self, indexed_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SortByTag)
        shuffle(indexed_mols)
        res = rig.setup_run(inputs={"inp": [indexed_mols]}, parameters={"reverse": True})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3
        for i in range(3):
            assert mols[i].molecules[0].get_tag("idx") == 2 - i

    def test_ExtractTag(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(ExtractScores)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]}, parameters={"tag": "score", "agg": "mean"}
        )
        scores = res["out"].get()
        assert scores is not None
        assert np.allclose(scores, [-0.5, -1.5])

    def test_ExtractScores(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(ExtractScores)
        res = rig.setup_run(inputs={"inp": [tagged_mols]})
        scores = res["out"].get()
        assert scores is not None
        assert np.allclose(scores, [-0.5, -1.5])

    def test_RMSD(self, path_ref: Path, iso_paths: list[Path]) -> None:
        iso_list = [Isomer.from_sdf(path, read_conformers=True) for path in iso_paths]
        ref = Isomer.from_sdf(path_ref)

        rig = TestRig(RMSD)
        res = rig.setup_run(inputs={"inp": [[IsomerCollection(iso_list)]], "inp_ref": [ref]})
        tagged = res["out"].get()

        assert tagged is not None
        assert np.allclose(tagged[0].molecules[0].get_tag("rmsd"), 3.36, 0.01)
        assert np.allclose(tagged[0].molecules[1].get_tag("rmsd"), 3.75, 0.01)
