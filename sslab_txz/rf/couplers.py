from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import epsilon_0, pi


class Probe(ABC):
    line_impedance: float = 50  # ohms
    shunt_capacitance: float = 0  # farads

    @property
    @abstractmethod
    def surface_area(self) -> float:
        ...

    def power_leak(self, electric_field: ArrayLike, frequency: float) -> ArrayLike:
        '''
        Power outcoupling through the probe coupler from a time-harmonic
        electric field, approximated as constant over the surface area
        of the probe.

        This can be calculated
        1/2 (omega eps0 A)^2 field^2 Z * 1/(1 + (omega C Z)^2)

        Parameters
        ----------
        electric_field: array_like, shape (..., 3)
            The (complex time-harmonic) electric field(s) at the probe,
            approximated as constant in magnitude over, and orthogonal
            everywhere to, the entire surface area of the probe.
        frequency: scalar
            The frequency of oscillation (in Hz)

        Returns
        -------
        array_like, shape (...,)
            Power outcoupling, in watts (W).
        '''
        electric_field = np.asarray(electric_field)
        if electric_field.shape[-1] != 3:
            raise ValueError
        omega = 2 * pi * frequency

        # We model the system as an equivalent circuit with a current
        # source I0 attached to a capacitor-resistor parallel structure,
        # where the shunt capacitor C reduces outcoupling and the
        # resistor R represents the impedance of the outcoupling line.
        # The outcoupled power is just 1/2 |I|^2 R where I is the
        # complex current the resistor.

        # The current source I0 comes from the displacement current of
        # the time-harmonic field E through the surface area of the
        # probe, which we approximate as spatially constant in magnitude
        # over the whole area, and everywhere orthogonal to the probe.
        # Then I0 = i \omega \epsilon_0 |E| A

        field_norm_squared = (np.abs(electric_field)**2).sum(axis=-1)
        displacement_current_norm_squared = \
            (omega * epsilon_0 * self.surface_area)**2 * field_norm_squared

        # To get the current through the resistor, we invoke the
        # admittances of the resistor and capacitor:
        # I/I0 = Y_C / (Y_C + Y_R)
        # We compute the norm-squared of this factor.
        shunt_factor = 1 / (1 + (omega * self.shunt_capacitance * self.line_impedance)**2)

        # Then it's just P = 1/2 |I0|^2 |I/I0|^2 R
        return 0.5 * displacement_current_norm_squared * shunt_factor * self.line_impedance

    def resonator_coupling_rate(
            self,
            electric_field_normalized: ArrayLike,
            frequency: float,
    ) -> float | NDArray:
        '''
        Parameters
        ----------
        electric_field_normalized: array_like, shape (..., 3)
            Complex time-harmonic electric field values at various
            points of interest, normalized as E / \\sqrt{\\iiint |E|^2 dV}
            (integral taken over entire resonator volume)
            as to be dimensionless.
        frequency: float
            Electric field oscillation frequency, in Hz

        Returns
        -------
        array_like, shape (...,)
            Outcoupling rates, in Hz.
        '''
        outcoupling_powers = self.power_leak(electric_field_normalized, frequency)

        # in general the total energy is the peak E-field energy,
        # \iiint 1/2 eps0 * |E|^2 dV
        total_energy = 0.5 * epsilon_0

        return outcoupling_powers / total_energy / (2 * pi)


@dataclass
class CylindricalProbe(Probe):
    radius: float
    tip_length: float

    @property
    def surface_area(self) -> float:
        base_area = pi * self.radius**2
        lateral_area = 2 * pi * self.radius * self.tip_length
        return base_area + lateral_area
