"""A cortical column of primary visual cortex, built from scratch.

Grounded in Kandel, *Principles of Neural Science*, 5th ed., chapter 25:
the laminar scheme of Fig. 25-12, the intrinsic circuitry of Fig. 25-15, the
columnar functional architecture of Fig. 25-11, the horizontal connections of
Fig. 25-16, and the computational module of Fig. 25-13.
"""

from .column import ColumnConfig, CorticalColumn
from .lgn import LGN
from .network import Network, Population
from .neurons import CELL_TYPES, NeuronArray
from .stimuli import (
    Bar,
    Blank,
    Dichoptic,
    DriftingGrating,
    Edge,
    FullFieldFlash,
    MovingBar,
    RetinalGrid,
    StaticGrating,
)

__all__ = [
    "ColumnConfig", "CorticalColumn", "LGN", "Network", "Population",
    "NeuronArray", "CELL_TYPES", "RetinalGrid", "Blank", "DriftingGrating",
    "StaticGrating", "Bar", "MovingBar", "Edge", "FullFieldFlash", "Dichoptic",
]
