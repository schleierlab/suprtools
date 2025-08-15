from abc import ABC, abstractmethod

from scipy.constants import c
from uncertainties import UFloat


class CavityGeometry(ABC):
    @property
    def length(self):
        return c / (2 * self.length_u)

    @property
    @abstractmethod
    def length_u(self):
        ...

    @property
    def fsr(self) -> float:
        return self.fsr_u.n

    @property
    def fsr_u(self) -> UFloat:
        return c / (2 * self.length_u)
