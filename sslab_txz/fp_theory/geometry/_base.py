from abc import ABC

from scipy.constants import c

from sslab_txz._typing import MaybeUFloat


class CavityGeometry(ABC):
    length: MaybeUFloat

    @property
    def fsr(self) -> float:
        return c / (2 * self.length)
