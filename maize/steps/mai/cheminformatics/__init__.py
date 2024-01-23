"""
Cheminformatics
^^^^^^^^^^^^^^^

Various cheminformatics utilities

"""
try:
    from maize.utilities.utilities import deprecated
except ImportError:
    from collections.abc import Callable
    from typing import Any

    def deprecated(_: str) -> Callable[[Any], Any]:  # type: ignore
        def inner(obj: Any) -> Any:
            return obj
        return inner

from .filters import (
    BestIsomerFilter,
    RankingFilter,
    TagFilter,
    SMARTSFilter,
    RMSDFilter,
)
from .sorters import TagSorter
from .taggers import TagIndex, SortByTag, ExtractScores, RMSD, LogTags, ExtractTag

@deprecated("please use 'TagFilter' instead")
class IsomerCollectionTagFilter(TagFilter):
    pass

@deprecated("please use 'RankingFilter' instead")
class IsomerCollectionRankingFilter(RankingFilter):
    pass

@deprecated("please use 'SMARTSFilter' instead")
class IsomerFilter(SMARTSFilter):
    pass

__all__ = [
    "BestIsomerFilter",
    "IsomerCollectionTagFilter",
    "IsomerCollectionRankingFilter",
    "IsomerFilter",
    "TagFilter",
    "RankingFilter",
    "SMARTSFilter",
    "RMSDFilter",
    "TagSorter",
    "TagIndex",
    "LogTags",
    "SortByTag",
    "ExtractScores",
    "ExtractTag",
    "RMSD",
]
