from os import PathLike
from typing import TypeAlias

from uncertainties import UFloat

MaybeUFloat: TypeAlias = float | UFloat
PathSpec: TypeAlias = str | PathLike[str]
