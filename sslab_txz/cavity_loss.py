from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.constants
import scipy.integrate
from scipy.constants import hbar
from scipy.constants import k as k_B
from scipy.constants import mu_0, pi

phi_0 = scipy.constants.physical_constants['mag. flux quantum'][0]


@dataclass
class TypeIISuperconductor:
    penetration_depth: float
    coherence_length: float
    gap_temperature: float
    room_temperature_resistivity: float
    residual_resistivity_ratio: float

    @property
    def rrr(self):
        '''
        Alias for residual_resistivity_ratio
        '''
        return self.residual_resistivity_ratio

    @property
    def cryo_normalstate_resistivity(self):
        return self.room_temperature_resistivity / self.residual_resistivity_ratio

    @property
    def cryo_rho_n(self):
        return self.cryo_normalstate_resistivity

    @staticmethod
    def fermi_dirac(beta_times_e: float):
        '''
        The Fermi-Dirac distribution function.

        Parameters
        ----------
        beta_times_e : scalar
            energy times the thermodynamic beta: E / k_B T

        Returns
        -------
        scalar
            The value of the Fermi-Dirac distribution
        '''

        # use exp of negative to avoid numerical instability
        negative_exp = np.exp(-beta_times_e)
        return negative_exp / (1 + negative_exp)

    @classmethod
    @np.vectorize
    def bcs_conductivity_ratio(cls, eta, beta_dimless):
        '''
        Ratio of superconducting to normal-state ac conductivity at given frequency
        and temperature, both specified dimensionlessly, by a numeric evaluation of
        the integral in Gurevich (2017), eq. 15.

        eta : scalar
            normalized frequency of excitation, hbar * omega / Delta
        beta_dimless : scalar
            Delta / k_B T

        Returns
        -------
        scalar
            Ratio sigma1/sigman of superconducting to normal-state (ac) conductivity
        '''

        def integrand(s):
            numerator = (s**2 + eta * s + 1) \
                * (cls.fermi_dirac(s * beta_dimless) - cls.fermi_dirac((s + eta) * beta_dimless))
            denominator = np.sqrt((s**2 - 1) * ((s + eta)**2 - 1))
            return numerator / denominator

        integral, err_est = scipy.integrate.quad(integrand, 1, np.inf, epsabs=0)

        return 2/eta * integral

    BCSMethod = Literal['numeric', '1216', 'sinhlin', 'eq18']

    def bcs_surface_resistance(
            self,
            freq: float,
            temp: float,
            method: BCSMethod = 'numeric',
    ):
        '''
        BCS surface resistance of niobium at specified frequency and temperature.
        according to Gurevich 2017, eqs. 12 and 15, evaluated numerically

        Parameters
        ----------
        freq : scalar
            AC frequency, in Hz
        temp : scalar
            temperature, in K
        method : {'numeric', '1216'}, optional
            Method to use to compute the BCS resistance. Defaults to 'numeric'.
            Should be one of:
                - 'numeric'
                    Numeric calculation using Gurevich 2017, eqs. 12 and 15.
                - '1216'
                    Eqs. 12 and 16 (eq. 18 without approximation 17). Fairly accurate.
                - 'sinhlin'
                    Same as '1216' but with the sinh term linearized.
                    Deviates at low temperature.
                - 'eq18'
                    Gurevich eq. 18. Requires hbar omega / 2 k_B T << 1

        rrr : scalar, optional
            Residual resistivity ratio R(300 K)/R(4 K), for computing normal state
            conductivity.
            Default: 300

        Returns
        -------
        scalar
            The BCS surface resistance at the specified frequency and
            temperature, evaluated numerically.
        '''
        omega = 2 * pi * freq
        eta = hbar * omega / (k_B * self.gap_temperature)
        beta_dimless = self.gap_temperature / temp

        if method == 'numeric':
            prefactor = 1/2 * mu_0**2 * self.penetration_depth**3 / self.cryo_rho_n
            return prefactor * omega**2 * self.bcs_conductivity_ratio(eta, beta_dimless)
        elif method in ['1216', 'sinhlin']:
            half_beta_hbar_omega = 1/2 * hbar * omega / (k_B * temp)
            prefactor = 2 * mu_0**2 * self.penetration_depth**3 \
                * k_B * self.gap_temperature / self.cryo_rho_n

            if method == '1216':
                sinhterm = np.sinh(half_beta_hbar_omega)
            elif method == 'sinhlin':
                sinhterm = half_beta_hbar_omega
            else:
                raise ValueError

            return prefactor * omega / hbar * sinhterm \
                * scipy.special.k0(half_beta_hbar_omega) * np.exp(-self.gap_temperature / temp)
        elif method == 'eq18':
            prefactor = mu_0**2 * self.penetration_depth**3 * self.gap_temperature / self.cryo_rho_n
            c1 = 4 / np.exp(np.euler_gamma)
            return prefactor * omega**2 / temp \
                * np.log(c1 * k_B * temp / (hbar * omega)) \
                * np.exp(-self.gap_temperature / temp)
        else:
            raise ValueError

    @property
    def trapped_vortex_resistance_per_field_highfreqlim(self):
        '''
        '''
        b_critical_2 = phi_0 / (2 * pi * self.coherence_length**2)
        eta = phi_0 * b_critical_2 / self.cryo_rho_n

        return phi_0 / (2 * eta * self.penetration_depth)

    def trapped_vortex_resistance_per_field(
            self,
            freq,
            segment_length,
    ):
        '''
        Compute the surface resistance per unit magnetic field due to
        trapped vortices, according to Gurevich (2017) eq. 30
        (Gurevich (2013) eqs. 14, 16), which assumes vortex segments with
        uniform lengths > penetration depth.

        Parameters
        ----------
        freq : scalar
            RF frequency, in Hz
        segment_length : scalar
            Length of vortex segment from surface to pinning center, in m.

        Returns
        -------
        scalar
            Surface resistance contribution from trapped vortices per unit of
            magnetic flux density
        '''

        omega = 2 * pi * freq

        kappa = self.penetration_depth / self.coherence_length
        g = np.log(kappa) + 0.5

        line_tension_epsilon = phi_0**2 * g / (4 * pi * mu_0 * self.penetration_depth**2)

        b_critical_2 = phi_0 / (2 * pi * self.coherence_length**2)
        eta = phi_0 * b_critical_2 / self.cryo_rho_n

        chi = omega * eta * self.penetration_depth**2 / line_tension_epsilon
        nu = omega * eta * segment_length**2 / line_tension_epsilon

        high_freq_lim_value = self.trapped_vortex_resistance_per_field_highfreqlim
        numerical_factor = chi**2 * (
            (5 + chi**2) / (1 + chi**2)**2
            - 2 / (chi**1.5) * np.imag(
                np.tanh(np.sqrt(nu * 1j)) / (np.sqrt(1j) * (1 - chi * 1j)**2)
            )
        )

        return high_freq_lim_value * numerical_factor


class Niobium(TypeIISuperconductor):
    penetration_depth: float = 36e-9
    coherence_length: float = 40e-9
    gap_temperature: float = 17.67
    room_temperature_resistivity: float = 147e-9
