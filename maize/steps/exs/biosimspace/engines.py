"""An Enum for the supported engines in BioSimSpace."""

from enum import auto

from maize.utilities.utilities import StrEnum


class BSSEngine(StrEnum):
    """Available BioSimSpace Engines."""

    GROMACS = auto()
    SANDER = auto()
    PMEMD = auto()
    PMEMD_CUDA = auto()
    OPENMM = auto()
    SOMD = auto()
    NAMD = auto()

    @property
    def class_name(self) -> str:
        """Get the class name for the engine."""
        return "".join([word.capitalize() for word in self.name.split("_")])


ENGINE_CALLABLES = {
    BSSEngine.GROMACS: ["gmx"],
    BSSEngine.SANDER: ["sander"],
    BSSEngine.PMEMD: ["pmemd"],
    BSSEngine.PMEMD_CUDA: ["pmemd.cuda"],
    BSSEngine.OPENMM: [],
    BSSEngine.SOMD: ["somd"],
    BSSEngine.NAMD: ["namd"],
}
"""A mapping between the BioSimSpace engine and the required callables for each engine."""

# Figure out which engines are available
