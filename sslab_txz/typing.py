from typing import Any, Optional, TypedDict

from matplotlib.typing import ColorType, LineStyleType


class PlotKwargs(TypedDict, total=False):
    # organized alphabetically per
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
    alpha: Optional[float]
    color: ColorType
    label: Any  # anything that can be str()'d
    linestyle: LineStyleType
    linewidth: float


class ErrorbarKwargs(PlotKwargs, total=False):
    ecolor: Optional[ColorType]
    elinewidth: Optional[float]
    capsize: float
    capthick: Optional[float]
    barsabove: bool


class AxvspanKwargs(TypedDict, total=False):
    color: ColorType
    alpha: Optional[float]
