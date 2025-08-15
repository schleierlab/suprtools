from typing import Any, Literal, Optional, TypedDict

from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Transform
from matplotlib.typing import ColorType, LineStyleType

PolarizationSpec = Literal[+1, -1]
ModeSpec = tuple[int, PolarizationSpec]


class PlotKwargs(TypedDict, total=False):
    # organized alphabetically per
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
    alpha: Optional[float]
    color: ColorType
    label: Any  # anything that can be str()'d
    linestyle: LineStyleType
    linewidth: float
    marker: None | str | Path | MarkerStyle
    rasterized: bool


class ErrorbarKwargs(PlotKwargs, total=False):
    ecolor: Optional[ColorType]
    elinewidth: Optional[float]
    capsize: float
    capthick: Optional[float]
    barsabove: bool


class AxvspanKwargs(TypedDict, total=False):
    color: ColorType
    alpha: Optional[float]


class PcolorKwargs(TypedDict, total=False):
    cmap: str | Colormap
    norm: str | Normalize
    vmin: float
    vmax: float


class TripcolorKwargs(PcolorKwargs, total=False):
    shading: Literal['flat', 'gouraud']


class PolyCollectionKwargs(TypedDict, total=False):
    color: ColorType


class FillBetweenKwargs(PolyCollectionKwargs, total=False):
    pass


FontSizeSpec = Literal['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']

class TextKwargs(TypedDict, total=False):
    fontsize: float | FontSizeSpec
    transform: Transform
    horizontalalignment: Literal['left', 'center', 'right']
    verticalalignment: Literal['top', 'center_baseline', 'center', 'baseline', 'bottom']


class AnnotateKwargs(TextKwargs, total=False):
    arrowprops: dict
    annotation_clip: Optional[bool]
