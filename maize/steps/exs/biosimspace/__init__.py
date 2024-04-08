"""
biosimspace
^^^^^^^^^

This python package provides Nodes for BioSimSpace (https://biosimspace.openbiosim.org). With 
BioSimSpace is a python framework for engine-agnostic simulation workflows. Nodes are implemented
to allow parameterisation, solvation, minimisation, equilibration, production, and relative and
absolute binding free energy calculations using BioSimSpace.
"""

from .parameterise import Parameterise

__all__ = ["Parameterise"]
