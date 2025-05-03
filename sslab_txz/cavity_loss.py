from __future__ import annotations

from abc import ABC
from typing import Literal, Optional, Unpack, assert_never, overload

import numpy as np
import scipy.constants
import scipy.integrate
import uncertainties
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from scipy import odr
from scipy.constants import c, elementary_charge, hbar, mu_0, pi
from scipy.constants import k as k_B
from uncertainties import unumpy as unp

from sslab_txz.plotting import expand_range, set_reci_ax
from sslab_txz.typing import ErrorbarKwargs, PlotKwargs

phi_0 = scipy.constants.physical_constants['mag. flux quantum'][0]
geom_factor_f = (pi / 4) * scipy.constants.value('characteristic impedance of vacuum')


class TypeIISuperconductor(ABC):
    penetration_depth: float
    '''The pure material penetration depth.'''

    coherence_length: float
    '''The pure material coherence length.'''

    gap_temperature: float
    room_temperature_resistivity: float
    residual_resistivity_ratio: float
    carrier_density: float

    transition_temperature: float
    '''
    Superconducting Tc. In principle could be deduced from BCS theory
    and the superconducting gap.
    '''

    lambda_temperature_dependence: bool
    '''
    Whether to include the 1 / sqrt(1 - (T/Tc)**4) empirical temperature dependence
    of the penetration depth in computations.
    '''

    lambda_purity_dependence: bool
    '''
    Whether to de-rate the penetration depth in computations given a finite mean-free-path
    '''

    xi_purity_dependence: bool
    '''Whether to de-rate the coherence length in computations given finite mean-free-path.'''

    def __init__(
            self,
            residual_resistivity_ratio: float,
            # can consider rolling these into one config object
            warm_lambda: bool = True,
            impure_lambda: bool = True,
            impure_xi: bool = True,
    ):
        self.residual_resistivity_ratio = residual_resistivity_ratio

        self.lambda_temperature_dependence = warm_lambda
        self.lambda_purity_dependence = impure_lambda
        self.xi_purity_dependence = impure_xi

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

    @classmethod
    def fermi_momentum(cls) -> float:
        return hbar * (3 * pi**2 * cls.carrier_density)**(1/3)

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

    def effective_penetration_depth(
            self,
            temperature: ArrayLike = 0,
    ) -> NDArray:
        R'''
        Compute the penetration depth \lambda.

        Parameters
        ----------
        temperature: array_like, optional
            If supplied, the temperature at which to evaluate the penetration depth.
            Only has nontrivial effect if self.lambda_temperature_dependence is True,
            or if temperature is nonzero.

        Returns
        -------
        NDArray
            Penetration depth, adjusted for material purity if
            `self.lambda_purity_dependence` is True, and for the temperature,
            if supplied and `self.lambda_temperature_dependence` is True.
        '''
        # see Gurevich (2017)

        purity_factor = 1
        if self.lambda_purity_dependence:
            # this is correct; we compare with the -pure- coherence length,
            # not the one shortened by mean free path
            purity_a = pi * self.coherence_length / (2 * np.asarray(self.mean_free_path))
            numerator = np.where(
                purity_a < 1,
                np.arccos(purity_a),
                np.arccosh(purity_a),
            )
            denominator = np.sqrt(np.abs(purity_a**2 - 1))
            purity_factor = np.sqrt(purity_a / (pi/2 - numerator / denominator))

        temperature = np.asarray(temperature)
        temperature_factor = np.ones_like(temperature)
        if self.lambda_temperature_dependence:
            temperature_factor = (1 - (temperature / self.transition_temperature)**4) ** (-1/2)

        return self.penetration_depth * purity_factor * temperature_factor

    @property
    def effective_coherence_length(self):
        if not self.xi_purity_dependence:
            return self.coherence_length

        return 1 / (1 / self.coherence_length + 1 / self.mean_free_path)

    @property
    def upper_critical_field(self):
        '''The upper critical field B_c2, in tesla'''
        return phi_0 / (2 * pi * self.effective_coherence_length**2)

    BCSMethod = Literal['numeric', '1216', 'sinhlin', 'eq18', 'basic']

    @property
    def mean_free_path(self):
        '''
        Compute the mean free path from the normal-state resistivity
        using the Drude formula.
        '''
        return (
            hbar * (3 * pi**2 * self.carrier_density)**(1/3)
            / (self.carrier_density * elementary_charge**2 * self.cryo_normalstate_resistivity)
        )

    def bcs_surface_resistance(
            self,
            freq: ArrayLike,
            temp: ArrayLike,
            method: BCSMethod = 'numeric',
    ):
        '''
        BCS surface resistance of niobium at specified frequency and temperature.
        according to Gurevich 2017, eqs. 12 and 15, evaluated numerically

        Parameters
        ----------
        freq : array-like
            AC frequency, in Hz
        temp : array-like
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
                - 'basic'
                    Gurevich eq. 2 with A = 8.9e+4 nOhm K / GHz^2

        Returns
        -------
        scalar
            The BCS surface resistance at the specified frequency and
            temperature, evaluated numerically.
        '''
        freq = np.asarray(freq)
        temp = np.asarray(temp)
        if method == 'basic':
            a = 8.9e-23
            return a * freq**2 / temp * np.exp(-self.gap_temperature / temp)

        omega = 2 * pi * freq
        eta = hbar * omega / (k_B * self.gap_temperature)
        beta_dimless = self.gap_temperature / temp

        lambda_impure = self.effective_penetration_depth(temp)

        if method == 'numeric':
            prefactor = 1/2 * mu_0**2 * lambda_impure**3 / self.cryo_rho_n
            return prefactor * omega**2 * self.bcs_conductivity_ratio(eta, beta_dimless)
        elif method in ['1216', 'sinhlin']:
            half_beta_hbar_omega = 1/2 * hbar * omega / (k_B * temp)
            prefactor = 2 * mu_0**2 * lambda_impure**3 \
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
            prefactor = mu_0**2 * lambda_impure**3 * self.gap_temperature / self.cryo_rho_n
            c1 = 4 / np.exp(np.euler_gamma)
            return prefactor * omega**2 / temp \
                * np.log(c1 * k_B * temp / (hbar * omega)) \
                * np.exp(-self.gap_temperature / temp)
        else:
            raise ValueError

    def trapped_vortex_resistance_per_field_highfreqlim(self, temperature: float):
        '''
        '''
        eta = phi_0 * self.upper_critical_field / self.cryo_rho_n

        return phi_0 / (2 * eta * self.effective_penetration_depth(temperature))

    def trapped_vortex_resistance_per_field(
            self,
            freq,
            segment_length: ArrayLike,
            temperature: float = 0,
            method: Literal['exact', 'eq32', 'eq33'] = 'exact',
    ):
        R'''
        Compute the surface resistance per unit magnetic field due to
        trapped vortices, assuming vortex segments with uniform lengths
        greater than the penetration depth.

        Parameters
        ----------
        freq : scalar
            RF frequency, in Hz
        segment_length : array_like
            Length of vortex segment from surface to pinning center, in m.
        temperature : scalar
            Temperature in K. Affects the value of the pentration depth used in
            the computation.
        method: {'exact'}
            'exact':
                compute according to Gurevich (2017) eq. 30
                (Gurevich (2013) eqs. 14, 16)
            'eq32'
                Use Gurevich (2017) eq. 32, which assumes the frequency
                is below the upper characteristic frequency \omega_\lambda
            'eq33'
                Use Gurevich (2017) eq. 33 (Gurevich (2013) eq 19),
                which assumes the frequency is between the upper
                characteristic frequency \omega_\lambda and the lower
                characteristic frequency \omega_\ell
            'modified_lowfreq'
                A low-frequency approximation that doesn't assume
                segment_length << london_lambda

        Returns
        -------
        scalar
            Surface resistance contribution from trapped vortices per unit of
            magnetic flux density, in units of ohm/tesla
        '''
        segment_length = np.asarray(segment_length)

        omega = 2 * pi * freq
        lambda_impure = self.effective_penetration_depth(temperature)

        line_tension_epsilon = phi_0**2 * self.g(temperature) / (4 * pi * mu_0 * lambda_impure**2)

        eta = phi_0 * self.upper_critical_field / self.cryo_rho_n

        chi = omega * eta * lambda_impure**2 / line_tension_epsilon
        nu = omega * eta * segment_length**2 / line_tension_epsilon
        root_2_nu = np.sqrt(2 * nu)
        alpha = lambda_impure / segment_length

        high_freq_lim_value = self.trapped_vortex_resistance_per_field_highfreqlim(temperature)

        if method == 'exact':
            numerical_factor = chi**2 * (
                (5 + chi**2) / (1 + chi**2)**2
                - 2 / (chi**1.5) * np.imag(
                    np.tanh(np.sqrt(nu * 1j)) / (np.sqrt(1j) * (1 - chi * 1j)**2)
                )
            )
        elif method == 'eq32':
            numerator = alpha * root_2_nu * (np.sinh(root_2_nu) - np.sin(root_2_nu))
            denominator = (np.cosh(root_2_nu) + np.cos(root_2_nu))
            numerical_factor = numerator / denominator
        elif method == 'eq33':
            b_critical = phi_0 / (2**1.5 * pi * lambda_impure * self.effective_coherence_length)
            return np.sqrt(mu_0 * self.cryo_normalstate_resistivity * omega / (2 * self.g(temperature))) / b_critical
        elif method == 'modified_lowfreq':
            numerical_factor = 2 * nu**2 * alpha**4 * (
                5/2
                + ((np.sinh(root_2_nu) - np.sin(root_2_nu)) - 2*nu * alpha**2 * (np.sinh(root_2_nu) + np.sin(root_2_nu)))
                    / (np.sqrt(2) * alpha**3 * nu**1.5 * (np.cosh(root_2_nu) + np.cos(root_2_nu)))
            )

            return phi_0 / (2 * eta * segment_length * alpha) * numerical_factor
        else:
            assert_never(method)


        return high_freq_lim_value * numerical_factor

    def kappa(self, temperature: ArrayLike):
        return self.effective_penetration_depth(temperature) / self.effective_coherence_length

    def g(self, temperature: ArrayLike):
        return np.log(self.kappa(temperature)) + 0.5

    def trapped_vortex_char_freq_lambda(self, temperature: ArrayLike):
        R'''
        Characteristic frequency \omega_\lambda / 2\pi (Gurevich eq. 29)
        '''
        return (
            self.g(temperature)
            * self.cryo_normalstate_resistivity * self.effective_coherence_length**2
            / (2 * mu_0 * self.effective_penetration_depth(temperature)**4)
            / (2 * pi)
        )

    def trapped_vortex_char_freq_ell(
            self,
            segment_length,
            temperature,
            impure,
    ):
        R'''
        Characteristic frequency \omega_\ell / 2\pi (Gurevich eq. 29)
        '''
        return (
            self.g(temperature, impure)
            * self.cryo_normalstate_resistivity * self.effective_coherence_length**2
            / (2 * mu_0 * self.effective_penetration_depth(temperature)**2 * segment_length**2)
            / (2 * pi)
        )



class Niobium(TypeIISuperconductor):
    penetration_depth: float = 36e-9
    coherence_length: float = 40e-9
    # gap_temperature: float = 17.67
    room_temperature_resistivity: float = 147e-9
    carrier_density = 5.56e+22 * 1e+6  # convert cm(-3) to m(-3)
    transition_temperature = 9.2

    def __init__(
            self,
            residual_resistivity_ratio: float,
            gap_temperature: float = 17.67,
            warm_lambda: bool = True,
            impure_lambda: bool = True,
            impure_xi: bool = True,
    ):
        self.gap_temperature = gap_temperature
        super().__init__(
            residual_resistivity_ratio,
            warm_lambda,
            impure_lambda,
            impure_xi,
        )

    @classmethod
    def from_rrr(cls, rrr):
        return cls(residual_resistivity_ratio=rrr)

    @classmethod
    def from_mean_free_path(
        cls,
        mean_free_path: float,
        # can be made optional, mandatory for easier debugging
        warm_lambda: bool,
        impure_lambda: bool,
        impure_xi: bool,
    ):
        cryo_resistivity = cls.fermi_momentum() /  (cls.carrier_density * elementary_charge**2 * mean_free_path)
        rrr = cls.room_temperature_resistivity / cryo_resistivity
        return cls(
            residual_resistivity_ratio=rrr,
            warm_lambda=warm_lambda,
            impure_lambda=impure_lambda,
            impure_xi=impure_xi,
        )


def roughness_limit_finesse(
        freq: ArrayLike,
        roughness_rms: ArrayLike,
):
    omega = 2 * pi * np.asarray(freq)
    roughness_rms = np.asarray(roughness_rms)
    return pi * c**2 / (2 * omega * roughness_rms)**2


def cavity_finesse(
        freq: ArrayLike,
        temp: ArrayLike,
        limiting_finesse: float,
        superconductor: TypeIISuperconductor,
        bcs_fudge_factor: float = 1,
        method: TypeIISuperconductor.BCSMethod = 'numeric',
) -> float:
    surface_res = superconductor.bcs_surface_resistance(
        freq,
        temp,
        method,
    )
    limiting_surface_res = geom_factor_f / limiting_finesse
    return geom_factor_f / (surface_res * bcs_fudge_factor + limiting_surface_res)


class TemperatureFit[T: TypeIISuperconductor]:
    mode_frequency: float
    model: odr.Model
    data: odr.RealData
    fit_result: odr.Output
    material: type[T]

    impure_lambda: bool
    '''Whether to use the purity-dependence of the penetration depth in computations.'''

    warm_lambda: bool
    '''Whether to include the empirical temperature dependence of the penetration depth.'''

    impure_xi: bool
    '''Whether to use the purity-dependence of coherence length in computations.'''

    @overload
    def __init__(
            self: TemperatureFit[Niobium],
            mode_data,
            material: type[Niobium] = ...,
            method: TypeIISuperconductor.BCSMethod = ...,
    ): ...
    @overload
    def __init__(
        self: TemperatureFit[T],
        mode_data,
        material: type[T],
        method: TypeIISuperconductor.BCSMethod = ...,
    ): ...

    def __init__(
            self,
            mode_data,
            material=Niobium,
            method='numeric',
            impure_lambda=True,
            warm_lambda=True,
            impure_xi=True,
    ):
        self.mode_frequency = mode_data['freq'].mean()
        self.model = odr.Model(self._fitfunc)
        self.data = odr.RealData(
            x=unp.nominal_values(mode_data['temp']),
            y=unp.nominal_values(mode_data['finesse']),
            sx=unp.std_devs(mode_data['temp']),
            sy=unp.std_devs(mode_data['finesse']),
        )
        self.material = material
        self.method = method
        self.impure_lambda = impure_lambda
        self.warm_lambda = warm_lambda
        self.impure_xi = impure_xi

    def _fitfunc(self, params, temp):
        # limit_finesse, scale_fctr = params
        limit_finesse, rrr = params
        return cavity_finesse(
            self.mode_frequency,
            temp,
            limit_finesse,
            self.material(
                residual_resistivity_ratio=rrr,
                warm_lambda=self.warm_lambda,
                impure_lambda=self.impure_lambda,
                impure_xi=self.impure_xi,
            ),
            method=self.method,
        )

    def fit(self, finesse_0=6e+7, rrr_0=300):
        self.odr = odr.ODR(self.data, self.model, beta0=[finesse_0, rrr_0])
        self.fit_result = self.odr.run()
        self.upopt = uncertainties.correlated_values(self.fit_result.beta, self.fit_result.cov_beta)

    def plot(
            self,
            plot_limit_finesse: bool = False,
            ax: Optional[Axes] = None,
            **errorbar_kw: Unpack[ErrorbarKwargs],
    ):
        if ax is None:
            _, ax = plt.subplots()

        errorbar_kw_default = ErrorbarKwargs(
            linestyle='None',
            # markersize=1,
            color='C0',
        )
        ax.errorbar(
            self.data.x,
            self.data.y,
            yerr=self.data.sy,
            xerr=self.data.sx,
            **(errorbar_kw_default | errorbar_kw),
        )

        if plot_limit_finesse:
            fit_limit_finesse = self.upopt[0]
            ax.axhline(
                fit_limit_finesse.n,
                color='red',
                linestyle='dashed',
            )
            ax.annotate(
                f'${fit_limit_finesse:SL}$',
                xy=(1, fit_limit_finesse.n),
                xycoords=ax.get_yaxis_transform(),
                xytext=(-3, -3),
                textcoords='offset points',
                ha='right',
                va='top',
            )

        self.plot_fit(ax=ax, color=(errorbar_kw['color'] or 'C0'))

        ax.set_xlabel('Temperature (K)')

    def plot_fit(
            self,
            limit_finesse: Optional[float] = None,
            ax: Optional[Axes] = None,
            scale: Literal['linear', 'reci', 'reci_r'] = 'linear',
            **kwargs: Unpack[PlotKwargs],
    ):
        """
        limit_finesse: float, optional
            If provided, plot the fitted BCS finesse curve with the
            supplied limiting finesse.
        """
        if ax is None:
            _, ax = plt.subplots()

        if scale == 'linear':
            pass
        elif scale == 'reci':
            set_reci_ax(ax, invert=False)
        elif scale == 'reci_r':
            set_reci_ax(ax, invert=True)
        else:
            assert_never(scale)

        finesse_limit = self.fit_result.beta[0] if limit_finesse is None else limit_finesse

        plot_t_lims = self.plot_t_lims(scale)
        plot_t_range = np.linspace(plot_t_lims[0], plot_t_lims[1], num=200)
        # not *plot_t_lims to satisfy mypy

        ax.plot(
            plot_t_range,
            self._fitfunc([finesse_limit, self.fit_result.beta[1]], plot_t_range),
            **kwargs,
        )

    def plot_t_lims(self, scale: Literal['linear', 'reci', 'reci_r'] = 'linear'):
        if scale == 'linear':
            return expand_range(self.data.x)
        elif scale in ['reci', 'reci_r']:
            return tuple(1/t for t in expand_range(1 / self.data.x))

    def superconductor(self) -> T:
        return self.material(
            residual_resistivity_ratio=self.fit_result.beta[1],
            warm_lambda=self.warm_lambda,
            impure_lambda=self.impure_lambda,
            impure_xi=self.impure_xi,
        )
