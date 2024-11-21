from os import PathLike
from typing import TypeAlias

from uncertainties import UFloat

MaybeUFloat: TypeAlias = float | UFloat
StrPath: TypeAlias = str | PathLike[str]
