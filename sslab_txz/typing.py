from typing import Any, Optional, TypedDict

from matplotlib.typing import ColorType


class ErrorbarKwargs(TypedDict, total=False):
    color: ColorType
    label: Any  # anything that can be str()'d
    linewidth: Any
    alpha: Optional[float]


class AxvspanKwargs(TypedDict, total=False):
    color: ColorType
    alpha: Optional[float]
