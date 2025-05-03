'''
TODO: deprecate/remove unused fitting styles here
'''

# for forward references; needed until python 3.13
from __future__ import annotations

import importlib.resources
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Optional, assert_never, cast, overload

import arc
import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import skimage.restoration
import skrf as rf
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from lmfit.model import ModelResult
from lmfit.models import LinearModel
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from scipy.constants import pi
from tqdm import tqdm
from uncertainties import UFloat, ufloat
from uncertainties import unumpy as unp

import sslab_txz.rydberg as rydtools
from sslab_txz._typing import StrPath
from sslab_txz.fp_theory.geometry._symmetric import SymmetricCavityGeometry
from sslab_txz.plotting import expand_range, sslab_style
from sslab_txz.plotting.style import kwarg_func_factory
from sslab_txz.rf.couplers import Probe
from sslab_txz.rf.cw import CWMeasurement
from sslab_txz.rf.errors import FitFailureError
from sslab_txz.rydberg import RydbergTransitionSeries
from sslab_txz.typing import ErrorbarKwargs, ModeSpec

matplotlib.style.use('default')  # to get rid of ugly arc style


class RingdownSet(CWMeasurement):
    functional_form = (
        R'$\tilde{s}_0\, \exp(-\frac{1}{2}\kappa t-i\Delta\omega t) '
        R'+ \tilde{s}_\infty$'
    )

    @staticmethod
    def ringdown_shape(
            t,
            a0,
            phi0,
            fwhm,
            delta_f,
            offset_re,
            offset_im,
    ):
        '''
        delta_f: f - f0 of the resonance
        '''
        offset_cmplx = offset_re + 1j * offset_im
        a0_cmplx = a0 * np.exp(1j * phi0)

        shifted_pole = -2 * pi * (0.5 * fwhm + 1j * delta_f)
        return offset_cmplx + a0_cmplx * np.exp(shifted_pole * t)

    def mean(self) -> Ringdown:
        return Ringdown(
            self.t, self.s21.mean(axis=0), self.frequency, None, self.stage_positions,
        )

    def _time_mask(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            t: Optional[NDArray] = None,
    ):
        mask: slice | NDArray = slice(None)
        t_arr = self.t if t is None else t
        if xrange is not None:
            x_lo = -np.inf if xrange[0] is None else xrange[0]
            x_hi = +np.inf if xrange[1] is None else xrange[1]
            mask = (x_lo <= t_arr) & (t_arr <= x_hi)

        return mask

    @overload
    def __getitem__(self, key: int) -> Ringdown: ...
    @overload
    def __getitem__(self, key: slice) -> RingdownSet: ...

    def __getitem__(self, key):
        match key:
            case int(i):
                return Ringdown(
                    self.t, self.s21[i], self.frequency, None, self.stage_positions,
                )
            case slice() as s:
                return RingdownSet(
                    self.t, self.s21[s], self.frequency, None, self.stage_positions,
                )

    def __len__(self) -> int:
        return len(self.s21)

    def _repr_html_(self) -> Optional[str]:
        fig, axs = plt.subplots(figsize=(4, 4), nrows=3, ncols=3, layout='constrained')

        # NDArray generic possibilities too limited
        # see https://stackoverflow.com/questions/74633074/how-to-type-hint-a-generic-numpy-array
        axs = cast(NDArray[Any], axs)
        axs_flat = cast(Sequence[Axes], axs.flatten())

        for i, ax in enumerate(axs_flat):
            self[i].plot_cartesian(
                ax=ax,
                alpha=0.8,
                linewidth=0.5,
            )
            # ax.scatter(0, 0, color='red', marker='+')
            ax.axhline(0, linestyle='dotted', color='0.8')
            ax.axvline(0, linestyle='dotted', color='0.8')

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        return fig._repr_html_()

    def _init_params_from_s21(
            self,
            s21: NDArray,
            model: Literal['spiral', 'circularized'] = 'spiral',
            suffix: Optional[str] = None,
            init_params: dict[str, dict] = dict(),
    ) -> Parameters:
        guess_offset = s21[-10:].mean()
        guess_std = np.real(s21[-10:]).std()
        guess_prefactor = s21[0] - guess_offset

        lookahead_ind = np.searchsorted(self.t, 0.1 * self.t[-1])
        lookahead_time = self.t[lookahead_ind]
        lookahead_s21 = s21[lookahead_ind] - guess_offset
        guess_fwhm = np.log(np.abs(guess_prefactor / lookahead_s21)) \
            / lookahead_time / pi

        s21_scale = 0.5 * np.max(np.abs(s21 - s21.mean()))

        full_suffix = '' if suffix is None else f'_{suffix}'

        common_params = dict(
            a0=dict(
                value=np.abs(guess_prefactor),
                min=0.5*np.abs(guess_prefactor),
                max=2*np.abs(guess_prefactor),
            ),
            # require 1/e time to be < 2 * ringdown duration
            fwhm=dict(
                value=guess_fwhm,
                min=1/(2 * self.t[-1])/(2 * pi),
                max=8e+3,
            ),
            offset_re=dict(
                value=np.real(guess_offset),
                min=np.real(guess_offset)-s21_scale,
                max=np.real(guess_offset)+s21_scale,
            ),
            offset_im=dict(
                value=np.imag(guess_offset),
                min=np.imag(guess_offset)-s21_scale,
                max=np.imag(guess_offset)+s21_scale,
            ),
        )
        if model == 'spiral':
            model_specific_params = dict(
                phi0=dict(
                    value=np.angle(guess_prefactor),
                    min=-2*pi,
                    max=2*pi,
                ),
                delta_f=dict(value=0, min=-10e+3, max=+10e+3),
            )
        elif model == 'circularized':
            model_specific_params = dict(
                eps=dict(
                    value=guess_std,
                    min=0,
                    max=10*guess_std,
                )
            )
        elif model == 'abssq':
            model_specific_params = dict(
                a0=dict(
                    value=np.abs(guess_prefactor)**2,
                    min=0.5*np.abs(guess_prefactor)**2,
                    max=2*np.abs(guess_prefactor)**2,
                ),
                eps=dict(
                    value=2*guess_std**2,
                    min=0,
                    max=20*guess_std**2,
                )
            )
        full_init_params = common_params | model_specific_params | init_params

        params = Parameters()
        for param_name, param_spec in full_init_params.items():
            params.add(
                param_name + full_suffix,
                **param_spec,
            )
        return params

    def fit_fwhms(self, init_params=dict()):
        def extract_fwhm(ringdown: Ringdown):
            fit = ringdown.fit_model(model='abssq', init_params=init_params)
            try:
                fit = cast(MinimizerResult, fit)
                return cast(UFloat, fit.uvars['fwhm'])
            except AttributeError:
                return None

        self_iter = tqdm(self) if len(self) > 500 else self
        maybe_fwhms = [extract_fwhm(rd) for rd in self_iter]
        return [maybe_fwhm for maybe_fwhm in maybe_fwhms if maybe_fwhm is not None]

    def fit_model(self, shared_params: Iterable[str] = ['fwhm', 'offset_re', 'offset_im']):
        fit_params: Parameters = sum(
            (self._init_params_from_s21(s21i, suffix=str(i)) for i, s21i in enumerate(self.s21)),
            start=Parameters(),
        )

        # constrain shared params to same value
        for shared_base_param in shared_params:
            for i in range(1, len(self)):
                fit_params[f'{shared_base_param}_{i}'].expr = f'{shared_base_param}_0'

        def multi_objective(params: Parameters, t: NDArray, s21s: NDArray) -> NDArray:
            residual_arr = [
                s21i - self.ringdown_shape(
                    t,
                    params[f'a0_{i}'].value,
                    params[f'phi0_{i}'].value,
                    params[f'fwhm_{i}'].value,
                    params[f'delta_f_{i}'].value,
                    params[f'offset_re_{i}'].value,
                    params[f'offset_im_{i}'].value,
                )
                for i, s21i in enumerate(s21s)
            ]
            return np.array(residual_arr).flatten()

        return lmfit.minimize(multi_objective, fit_params, args=(self.t, self.s21))

    def collective_fit(self) -> RingdownCollectiveFit:
        fit = RingdownCollectiveFit(self)
        fit.fit()
        return fit

    def visualize(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            xscale=1e+6,
            onering: bool = False,
    ):
        raise NotImplementedError


class Ringdown(RingdownSet):
    @staticmethod
    def circularized_gaussian_pseudoresidual(params, t, data=None):
        # unpack parameters: extract .value attribute for each parameter
        parvals = params.valuesdict()
        offset = parvals['offset_re'] + 1j * parvals['offset_im']
        fwhm = parvals['fwhm']
        a0 = parvals['a0']
        eps = parvals['eps']

        model_abs = a0 * np.exp(-pi * fwhm * t)

        if data is None:
            return model_abs

        data_radial = np.abs(data - offset)
        i0_arg = model_abs * data_radial / eps**2
        i0e_val = scipy.special.i0e(i0_arg)  # I_0(x) exp(-|x|)
        neg_log_likelihood = (
            -np.log(data_radial) + 2 * np.log(eps) + (data_radial**2 + model_abs**2) / (2 * eps**2)
            - (np.log(i0e_val) + np.abs(i0_arg))
        )
        return np.sqrt(2 * (1e+2 + neg_log_likelihood))

    @staticmethod
    def abs_residual(params, t, data=None):
        parvals = params.valuesdict()
        offset = parvals['offset_re'] + 1j * parvals['offset_im']
        fwhm = parvals['fwhm']
        a0 = parvals['a0']
        eps = parvals['eps']

        model_abs_sq = a0 * np.exp(-2 * pi * fwhm * t) + eps
        data_radial_sq = np.abs(data - offset)**2
        return data_radial_sq - model_abs_sq

    def fit_scalar(self):
        fit = RingdownScalarFit(self)
        fit.fit()
        fit.fit_phase()
        return fit

    def fit_model(self, model='spiral', init_params=dict()):
        s21 = self.s21[0]
        params = self._init_params_from_s21(s21, init_params=init_params, model=model)

        if model == 'spiral':
            self.model = lmfit.Model(self.ringdown_shape)

            fit_range_trunc_ind = 0
            self.s21_fit = s21[fit_range_trunc_ind:]
            self.t_fit = self.t[fit_range_trunc_ind:]
            self.fit = self.model.fit(
                self.s21_fit,
                params=params,
                t=self.t_fit,
            )
        elif model == 'circularized':
            return lmfit.minimize(
                self.circularized_gaussian_pseudoresidual,
                params,
                args=(self.t,),
                kws=dict(data=s21),
            )
        elif model == 'abssq':
            return lmfit.minimize(
                self.abs_residual,
                params,
                args=(self.t,),
                kws=dict(data=s21),
            )

    def plot_cartesian(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            scale=1,
            ax: Optional[Axes] = None,
            **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots()
            ax = cast(Axes, ax)
            ax.set_aspect(1)
            ax.set_box_aspect(1)
        mask = self._time_mask(xrange)
        ax.plot(
            *rf.complex_2_reim(scale * self.s21[0][mask]),
            **kwargs,
        )

    def plot_polar(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            scale=1,
            ax: Optional[Axes] = None,
            offset: complex = 0,
            **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            ax = cast(Axes, ax)
            ax.set_aspect(1)
            ax.set_box_aspect(1)

        mask = self._time_mask(xrange)
        complex_data = scale * (self.s21[0][mask] - offset)
        ax.plot(
            rf.complex_2_radian(complex_data),
            rf.complex_2_magnitude(complex_data),
            **kwargs,
        )

    def plot_db(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            xscale=1e+6,
            ax: Optional[Axes] = None,
            offset: complex = 0,
            **kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()

        mask = self._time_mask(xrange)
        masked_time = self.t[mask]

        ax.plot(
            xscale * masked_time,
            rf.complex_2_db(self.s21[0, mask] - offset),
            label=fr'$\omega/2\pi$ = {self.frequency / 1e+9:.9f} GHz',
            **kwargs,
        )
        ax.legend(fontsize='x-small')
        ax.set_ylabel('$|S_{21}|$ (dB)')

    def plot_phase(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            xscale=1e+6,
            ax: Optional[Axes] = None,
            offset: complex = 0,
            **kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()

        mask = self._time_mask(xrange)
        masked_time = self.t[mask]
        ax.plot(
            xscale * masked_time,
            np.unwrap(rf.complex_2_degree(self.s21[0, mask] - offset), period=360) / 360,
            **kwargs,
        )
        ax.set_ylabel(R'$\angle S_{21} / 2\pi$')

    def visualize(
            self,
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            xscale=1e+6,
            onering: bool = False,
    ):
        fig, axs = plt.subplot_mosaic(
            [
                ['db', 'reim'],
                ['deg', 'reim'],
            ],
            layout='constrained',
            figsize=(9.6, 4),
        )
        axs['reim'].set_aspect(1)
        axs['reim'].set_box_aspect(1)

        self.plot_db(xrange, xscale, ax=axs['db'])
        self.plot_phase(xrange, xscale, ax=axs['deg'])

        reim_scale = 1e+6
        reim_scale_latex = '10^6'
        self.plot_cartesian(ax=axs['reim'], scale=reim_scale)

        if hasattr(self, 'fit') and hasattr(self.fit, 'uvars'):
            mask_tfit = self._time_mask(xrange, t=self.t_fit)
            masked_tfit = self.t_fit[mask_tfit]
            model_val = self.fit.eval(t=masked_tfit)
            axs['db'].plot(
                xscale * masked_tfit,
                rf.complex_2_db(model_val),
                label=(
                    fr'$\kappa/2\pi = {self.fit.uvars["fwhm"]:SL}$ Hz'
                )
            )
            axs['deg'].plot(
                xscale * masked_tfit,
                np.unwrap(rf.complex_2_degree(model_val), period=360),
                label=(
                    R'$\Delta\omega \equiv \omega - \omega_0 = '
                    fr'{self.fit.uvars["delta_f"]:SL} \times 2\pi$ Hz'
                ),
            )

            axs['reim'].plot(
                *rf.complex_2_reim(reim_scale * model_val),
                label=Ringdown.functional_form,
            )

        axs['deg'].legend(fontsize='x-small')
        axs['reim'].legend(fontsize='x-small')

        axs['deg'].set_xlabel(R'Time ($\mu$s)')
        axs['reim'].set_xlabel(fR'${reim_scale_latex} \times \operatorname{{Re}} S_{{21}}$')
        axs['reim'].set_ylabel(fR'${reim_scale_latex} \times \operatorname{{Im}} S_{{21}}$')

        for ax in axs.values():
            sslab_style(ax)

        if onering:
            # we use a str ('.img') as the anchor here instead of an import `from . import img`
            # because that doesn't work with strict editable installs
            imfile = importlib.resources.files('.img') / 'onering_wikipedia.png'

            # imfile (Traversable) can be passed to open()
            # but typeshed annotation for open() doesn't support Traversable
            with open(imfile, 'rb') as fp:  # type: ignore
                im = Image.open(fp)
                ax_ring = fig.add_axes((0, 0, 1, 1))

                # must stay in because `im` is lazily eval'd
                ax_ring.imshow(
                    im,
                    alpha=0.1,
                    zorder=-3,
                )
                ax_ring.axis('off')

        return fig, axs

    def _repr_html_(self):
        fig, _ = self.visualize()
        return fig._repr_html_()


class RingdownCollectiveFit:
    ringdown_set: RingdownSet
    fit_result: Optional[MinimizerResult] = None

    def __init__(self, ringdown_set: RingdownSet):
        self.ringdown_set = ringdown_set

    @property
    def t(self):
        return self.ringdown_set.t

    def residual(self, params):
        parvals = params.valuesdict()
        offset = parvals['offset_re'] + 1j * parvals['offset_im']
        fwhm = parvals['fwhm']
        a0 = parvals['a0']
        const = parvals['const']

        begin_time = 1e-6
        begin_ind = np.searchsorted(self.t, begin_time)

        model_abs_sq = a0 * np.exp(-2 * pi * fwhm * self.t[begin_ind:]) + const
        data_radial_sq = np.abs(self.ringdown_set.s21[:, begin_ind:] - offset)**2
        return data_radial_sq.mean(axis=0) - model_abs_sq

    def fit(self):
        max_fwhm = 0.1 / (self.t[1] - self.t[0])
        min_fwhm = 0.1 / (self.t[-1] - self.t[0])

        offset_guess = self.ringdown_set.s21[:, -30:].mean()
        params = lmfit.create_params(
            a0=dict(
                value=np.mean(np.abs(self.ringdown_set.s21[:, 0] - offset_guess)**2),
                min=0,
            ),
            fwhm=dict(
                value=np.sqrt(min_fwhm * max_fwhm),
                min=min_fwhm,
                max=max_fwhm,
            ),
            offset_re=np.real(offset_guess),
            offset_im=np.imag(offset_guess),
            const=dict(value=self.ringdown_set.s21[:, -10:].std()**2, min=0),
        )
        self.fit_result = lmfit.minimize(self.residual, params)

    def plot_fit(
            self,
            ax: Optional[Axes] = None,
            xscale=1,
            data_kw=dict(),
            model_kw=dict(),
            legend_kw: Optional[Mapping] = dict(),
            xrange: Optional[tuple[Optional[float], Optional[float]]] = None,
            normalized: bool = False,
            noise_removed: bool = False,
    ):
        '''
        Plot the norm-squared of the offset-subtracted signal together
        with the theoretical exponential-plus-constant form we fitted.

        Parameters
        ----------
        ax: Axes, optional
            If supplied, plot the fit in `ax`.
        xscale: scalar
            Factor by which to multiply the time (x-) axis values, which
            by default are given in seconds (s). For example, passing in
            xscale=1e+3 plots the trace in milliseconds (ms).
        data_kw, model_kw: Mapping
            Keyword arguments for plotting the data and model, resp.
            Passed in each case to Axes.plot.
        legend_kw: Mapping or None, optional
            Keyword arguments for the legend. If None, do not draw a
            legend.
        xrange: 2-tuple of (float or None), optional
            If supplied, only plots the data bounded by the limits. The
            limits should be given as (lower_bound, upper_bound) in
            units of seconds (s) even if xscale != 1.
            A value of None in either limit means not applying a bound.
        normalized: bool, optional
            If True, normalize to the starting value of the fit.
        noise_removed: bool, optional
            If True, subtract out the noise power from the fit.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        if self.fit_result is None:
            raise ValueError
        if not hasattr(self.fit_result, 'uvars'):
            raise ValueError

        mask = self.ringdown_set._time_mask(xrange)
        masked_time = self.t[mask]

        uvars = self.fit_result.uvars
        offset = uvars['offset_re'].n + 1j * uvars['offset_im'].n
        mean_ringdown_power = \
            np.mean(np.abs(self.ringdown_set.s21 - offset)**2, axis=0)[mask]
        fit_values = uvars['const'].n \
            + uvars['a0'].n * np.exp(-2 * pi * uvars['fwhm'].n * masked_time)
        norm_factor = fit_values[0] if normalized else 1

        maybe_noise_offset = uvars['const'].n if noise_removed else 0

        ax.plot(
            xscale * masked_time,
            (mean_ringdown_power - maybe_noise_offset) / norm_factor,
            **data_kw,
        )

        model_kw_default = dict(
            color='red',
            label=Rf'$\kappa/2\pi = {uvars['fwhm']:SL} $ Hz',
        )
        ax.plot(
            xscale * masked_time,
            (fit_values - maybe_noise_offset) / norm_factor,
            **(model_kw_default | model_kw),
        )
        ax.set_yscale('log')
        if legend_kw is not None:
            ax.legend(**legend_kw)

    def _repr_html_(self) -> Optional[str]:
        if self.fit_result is None:
            return None
        if not hasattr(self.fit_result, 'uvars'):
            return f'Unsuccessful collective fit of {repr(self.ringdown_set)}'

        fig, ax = plt.subplots()
        self.plot_fit(ax)
        return fig._repr_html_()

    def fwhm_u(self) -> UFloat:
        if self.fit_result is None:
            raise ValueError
        return self.fit_result.uvars['fwhm']

    def fwhm(self) -> float:
        return self.fwhm_u().n


class RingdownScalarFit(RingdownCollectiveFit):
    ringdown_set: Ringdown
    phase_fit: Optional[ModelResult] = None

    def __init__(self, ringdown: Ringdown):
        self.ringdown_set = ringdown

    @property
    def ringdown(self):
        return self.ringdown_set[0]

    def fit_phase(self):
        if self.fit_result is None:
            raise FitFailureError
        if not hasattr(self.fit_result, 'uvars'):
            raise FitFailureError

        uvars = self.fit_result.uvars
        offset = uvars['offset_re'].n + 1j * uvars['offset_im'].n

        # phase_cyc = np.unwrap(np.angle(self.ringdown.s21[0] - offset)) / (2 * pi)
        phase_rad = np.angle(self.ringdown.s21[0] - offset)
        phase_cyc = skimage.restoration.unwrap_phase(phase_rad) / (2 * pi)

        model = LinearModel()
        self.phase_fit = model.fit(phase_cyc, x=self.t, offset=phase_cyc[0], slope=0)

    def _repr_html_(self) -> Optional[str]:
        if self.fit_result is None:
            return None
        if not hasattr(self.fit_result, 'uvars'):
            return f'Unsuccessful collective fit of {repr(self.ringdown)}'
        uvars = self.fit_result.uvars
        offset = uvars['offset_re'].n + 1j * uvars['offset_im'].n

        if self.phase_fit is None:
            fig, axs = plt.subplots(
                figsize=(6.4, 4.8),
                nrows=2,
                sharex=True,
                layout='constrained',
            )
            ax_cart, ax_db = cast(tuple[Axes, ...], axs)
        else:
            fig, axs = plt.subplot_mosaic(
                [
                    ['cart', 'db'],
                    ['cart', 'phase'],
                ],
                layout='constrained',
                figsize=(9.6, 4),
            )
            axs = cast(Mapping[str, Axes], axs)
            ax_cart = axs['cart']
            ax_db = axs['db']
            ax_phase = axs['phase']
            ax_phase.sharex(ax_db)
            ax_db.xaxis.set_visible(False)

        self.ringdown.plot_cartesian(scale=1e+3, ax=ax_cart)
        ax_cart.scatter(
            1e+3 * np.real(offset),
            1e+3 * np.imag(offset),
            marker='+',
            color='red',
            zorder=2,  # = default Axes.plot zorder, s.t. this is plotted over data
            label=(
                R'$10^3 \times S_{21}(\infty) = '
                fR'{1e+3 * uvars["offset_re"]:SL}'
                fR'+ {1e+3 * uvars["offset_im"]:SL} i$'
            ),
        )
        ax_cart.legend()

        ax_db.plot(
            self.t,
            rf.complex_2_db10(np.mean(np.abs(self.ringdown.s21 - offset)**2, axis=0)),
        )

        model_vals = uvars['a0'].n * np.exp(-2 * pi * uvars['fwhm'].n * self.t) \
            + uvars['const'].n
        ax_db.plot(
            self.t,
            rf.complex_2_db10(model_vals),
            color='red',
            label=Rf'$\kappa/2\pi = {uvars['fwhm']:SL} $ Hz',
        )
        ax_db.legend()

        if self.phase_fit is not None:
            ax_phase.plot(
                self.t,
                self.phase_fit.data,
            )
            ax_phase.plot(
                self.t,
                self.phase_fit.best_fit,
                color='red',
                label=(
                    R'$\omega - \overline{{\omega_0}} = '
                    Rf'{-self.phase_fit.uvars['slope']:SL} \times 2\pi$ Hz'
                ),
            )
            ax_phase.legend()

        fig.suptitle(f'Fit to ringdown at {self.ringdown_set.frequency:,.0f} Hz')
        return fig._repr_html_()


class RingdownSetSweep:
    geometry: SymmetricCavityGeometry
    stage_pos_converter: Callable[[NDArray], NDArray]
    finesse: dict[ModeSpec, NDArray]

    def __init__(
            self,
            ringdowns: Mapping[float, Mapping[tuple[int, int], RingdownSet]],
            geometry: SymmetricCavityGeometry,
            stage_pos_converter: Optional[Callable[[NDArray], NDArray]] = None,
    ):
        self.ringdowns = ringdowns
        self.modelist = tuple(next(iter(self.ringdowns.values())))
        self.stage_positions = sorted(list(self.ringdowns))
        self.geometry = geometry
        # the parens around lambda expression are important
        # otherwise the conditional is part of the lambda
        self.stage_pos_converter = (
            (lambda x: x)
            if stage_pos_converter is None
            else stage_pos_converter
        )

        for single_stage_pos_ringdowns in self.ringdowns.values():
            assert set(single_stage_pos_ringdowns) == set(self.modelist)

        q_values = [q for q, _ in self.modelist]
        self.q_range = (min(q_values), max(q_values))
        self.kwarg_func = kwarg_func_factory(
            q_range=self.q_range,
            label='',
            # markers=('d', '*'),
        )

        self.model = lmfit.Model(
            self.log_finesse_model,
            independent_vars=['probe_r'],
            param_names=['limit_log_fin', 'beam_enlarge_factor', 'probe_loss_factor'],
        )

    @staticmethod
    def load_ringdowns(dir, mainpath, modes, exceptiondict=dict()):
        default_windows = {
            mode_tuple: (
                Path(dir) / mainpath / f'window{i:03d}.h5'
            )
            for i, mode_tuple in enumerate(modes)
        }

        windows = default_windows | exceptiondict
        return {
            mode_tuple: None if path is None else RingdownSet.from_h5(path)
            for mode_tuple, path in tqdm(windows.items())
        }

    @classmethod
    def from_directories(
            cls,
            dir: StrPath,
            datapaths: Mapping[float, StrPath],
            modes,
            geometry: SymmetricCavityGeometry,
            stage_pos_converter: Optional[Callable[[NDArray], NDArray]] = None,
    ):
        ringdowns = {
            stage_pos: cls.load_ringdowns(Path(dir), datapath, modes)
            for stage_pos, datapath in datapaths.items()
        }
        return cls(ringdowns, geometry, stage_pos_converter)

    def fit(self):
        self.collective_fits = {
            key: {
                mode: rdsets_dict[mode].collective_fit()
                for mode in tqdm(rdsets_dict, leave=True)
            }
            for key, rdsets_dict in self.ringdowns.items()
        }

    @staticmethod
    def frac_uncert(uarr):
        return unp.std_devs(uarr) / unp.nominal_values(uarr)

    def extract_finesses(
            self,
            a0_uncert_limit=0.3,
            fwhm_uncert_limit=0.26,
    ):
        stagesweep_rd_finesses = {}
        for q, pol in self.modelist:
            fits = [
                self.collective_fits[stage_pos].get((q, pol))
                for stage_pos in self.stage_positions
            ]
            fwhms = np.array([
                (
                    fit.fwhm_u()
                    if (
                        fit is not None
                        and hasattr(fit.fit_result, 'uvars')
                        and (self.frac_uncert(fit.fit_result.uvars['a0']) < a0_uncert_limit)
                        and (self.frac_uncert(fit.fit_result.uvars['fwhm']) < fwhm_uncert_limit)
                    ) else ufloat(np.nan, np.nan)
                )
                for fit in fits
            ])
            fins = self.geometry.fsr / fwhms
            ringdown_freqs = [fit.ringdown_set.frequency for fit in fits]

            stagesweep_rd_finesses[q, pol] = np.rec.fromarrays(
                [self.stage_positions, fins, ringdown_freqs],
                names=['stage_pos', 'finesse', 'freq'],
            )
        self.finesses = stagesweep_rd_finesses

    def log_finesse_model(
            self,
            probe_r,
            limit_log_fin,
            beam_enlarge_factor,
            probe_loss_factor,
            freq: float,
            probe: Probe,
            probe_z: float,
    ):
        fsr = unp.nominal_values(self.geometry.fsr)
        limit_fwhm = fsr / np.exp(limit_log_fin)

        probe_ufield_scalar_normed = self.geometry.paraxial_scalar_beam_field(
            probe_r / beam_enlarge_factor,
            probe_z,
            freq,
            norm='volume',
        )
        probe_field_vector_normed = \
            unp.nominal_values(probe_ufield_scalar_normed)[..., np.newaxis] * [1, 0, 0]

        single_coupling_rate = probe.resonator_coupling_rate(probe_field_vector_normed, freq)
        return np.log(fsr / (2 * single_coupling_rate * probe_loss_factor + limit_fwhm))

    def plot(
            self,
            ax_fin: Optional[Axes] = None,
            plot_modes: Optional[Sequence[int]] = None,
            probe: Optional[Probe] = None,
            probe_z: Optional[float] = None,
            xerr: float = 0.020,
            extrapolation_r: Optional[float] = None,
            **kwargs,
    ) -> dict[int, tuple[ErrorbarContainer, ...]]:
        if (probe is None) != (probe_z is None):
            raise ValueError('Cannot supply exactly one of probe, probe_z')

        if ax_fin is None:
            _, plot_ax = plt.subplots()
        else:
            plot_ax = ax_fin

        q_vals = (
            range(self.q_range[1], self.q_range[0] - 1, -1)
            if plot_modes is None
            else plot_modes
        )

        def mask_mode_data(mode_data):
            fins = mode_data['finesse']
            fwhms = self.geometry.fsr / fins
            mask = (
                ~np.isclose(unp.nominal_values(fwhms), 1.0)
                & ~np.isnan(unp.nominal_values(fins))
                # we filter out bad fits in extract_finesses()
            )
            return mode_data[mask]

        def plot_single_mode(q, pol) -> ErrorbarContainer:
            mode_data_masked = mask_mode_data(self.finesses[q, pol])
            converted_stage_pos_masked = self.stage_pos_converter(mode_data_masked['stage_pos'])
            frequency = mode_data_masked['freq'].mean()
            errorbar_kw = (
                self.kwarg_func(frequency, q, pol)
                | dict(alpha=1)
                | kwargs
            )
            return plot_ax.errorbar(
                converted_stage_pos_masked,
                unp.nominal_values(mode_data_masked['finesse']),
                unp.std_devs(mode_data_masked['finesse']),
                xerr=xerr,
                **errorbar_kw,
            )

        ebar_containers = dict[int, tuple[ErrorbarContainer, ...]]()
        for q in q_vals:
            ebar_containers[q] = tuple(
                plot_single_mode(q, pol)
                for pol in [+1, -1]
            )

        if probe is not None:
            for q in q_vals:
                bothpol_data = np.concatenate([self.finesses[q, +1], self.finesses[q, -1]])
                bothpol_data_masked = mask_mode_data(bothpol_data)
                log_finesse = unp.log(bothpol_data_masked['finesse'])

                max_log_fin = max(unp.nominal_values(log_finesse))
                params = self.model.make_params(
                    limit_log_fin=dict(value=max_log_fin, min=max_log_fin-0.3, max=max_log_fin+0.3),
                    beam_enlarge_factor=dict(value=1, min=0.8, max=2, vary=True),
                    probe_loss_factor=dict(value=1, min=0.0003, max=3, vary=True),
                )
                frequency = bothpol_data_masked['freq'].mean()

                stage_pos = self.stage_pos_converter(bothpol_data_masked['stage_pos'])
                fit = self.model.fit(
                    unp.nominal_values(log_finesse),
                    params,
                    weights=1/unp.std_devs(log_finesse),
                    probe_r=(stage_pos/1e+3),
                    probe=probe,
                    probe_z=probe_z,
                    freq=frequency,
                )

                # print(
                #     f'{q=}, '
                #     f'beam_enlarge fudge: {fit.uvars['beam_enlarge_factor']:S}, '
                #     f'probe_loss fudge: {fit.uvars['probe_loss_factor']:S}',
                # )

                r_space = 1e-3 * np.linspace(*expand_range(stage_pos, factor=1.1))
                plot_kw = (
                    self.kwarg_func(frequency, q, +1)
                    | dict(alpha=0.7, marker=None)
                    | kwargs
                    | dict(linestyle='solid', label=None)
                )
                plot_ax.plot(
                    1e+3 * r_space,
                    np.exp(fit.eval(probe_r=r_space)),
                    **plot_kw,
                )
                if extrapolation_r is not None:
                    extended_r_space = np.linspace(max(r_space), extrapolation_r)
                    plot_ax.plot(
                        1e+3 * extended_r_space,
                        np.exp(fit.eval(probe_r=extended_r_space)),
                        **(plot_kw | dict(linestyle='dashed')),
                    )

        if ax_fin is None:
            plot_ax.set_yscale('log')
            plot_ax.set_ylabel('Finesse')
            plot_ax.set_xlabel('Probe extension (mm)')
            plot_ax.legend(
                fontsize='x-small',
                ncols=2,
                bbox_to_anchor=(1.01, 0.5),
                loc='center left',
            )
            sslab_style(plot_ax)

        return ebar_containers

    def highest_finesse_values(self, mask=(lambda q, pol: True)):
        records = []
        for (q, pol), data_arr in self.finesses.items():
            if np.all(np.isnan(unp.nominal_values(data_arr['finesse']))):
                continue
            if not mask(q, pol):
                continue
            best_finesse = np.nanmax(data_arr['finesse'])

            records.append((
                q,
                pol,
                data_arr['freq'].mean(),
                best_finesse,
            ))

        return np.rec.fromrecords(records, names=['q', 'pol', 'freq', 'finesse'])

    def plot_highest_finesse(self, ax=None, mask=(lambda q, pol: True), **kwargs):
        if ax is None:
            _, plot_ax = plt.subplots()
        else:
            plot_ax = ax

        highest_finesses = self.highest_finesse_values(mask)
        for pol in [+1, -1]:
            highest_finesses_thispol = highest_finesses[highest_finesses['pol'] == pol]
            plot_ax.errorbar(
                highest_finesses_thispol['freq'] / 1e+9,
                unp.nominal_values(highest_finesses_thispol['finesse']),
                unp.std_devs(highest_finesses_thispol['finesse']),
                **(
                    self.kwarg_func(None, highest_finesses_thispol['q'][0], pol)
                    | dict(alpha=1, color='C0', linestyle='None')
                    | kwargs
                )
            )

        if ax is None:
            plot_ax.set_title(R'All $\mathrm{TEM}_{00}$ modes')
            plot_ax.set_xlabel('Frequency (GHz)')
            plot_ax.set_ylabel('Finesse')
            plot_ax.set_yscale('log')
            sslab_style(plot_ax)

    def cooperativity_plot(
            self,
            series: RydbergTransitionSeries,
            modes: Sequence[int],
            polarizations: Sequence[Literal[1, -1]],
            ax: Optional[Axes] = None,
            freqscale: float = 1e+9,
            finesse_func: Optional[Callable] = None,
    ):
        if ax is None:
            fig, ax_eta = plt.subplots()
        else:
            ax_eta = ax

        power_laws = series.power_law_params()
        def get_cooperativity(freq, get_finesse: Sequence[float] | Callable[[ArrayLike], ArrayLike]):
            # convert units from e a_0 to SI
            d = power_laws['d'](freq) * scipy.constants.elementary_charge * scipy.constants.value('Bohr radius')
            vac_field = self.geometry.waist_vacuum_field_fromfreq(freq)
            vacuum_rabi_freq = 2 * d * vac_field / scipy.constants.Planck / (1 if series.polarization == 0 else np.sqrt(2))

            # 2g, in 2pi Hz
            ugammas = 1 / (2 * pi * power_laws['lifetime'](freq))

            ukappas: NDArray
            if callable(get_finesse):
                ukappas = self.geometry.fsr / np.asarray(get_finesse(freq))
            else:
                ukappas = self.geometry.fsr / np.asarray(get_finesse)

            return vacuum_rabi_freq ** 2 / (ugammas * ukappas)

        for pol_ind, pol in enumerate(polarizations):
            highest_finesses = self.highest_finesse_values(lambda q, p: q in modes and p == pol)

            pol_axis = 'x' if pol == +1 else 'y'
            marker = '.' if pol == +1 else 'x'

            freqs = highest_finesses['freq']
            uetas = get_cooperativity(freqs, highest_finesses['finesse'])

            linestyle_dict = (
                ErrorbarKwargs()
                if finesse_func is None else
                ErrorbarKwargs(linestyle='None')
            )
            ax_eta.errorbar(
                freqs / freqscale,
                unp.nominal_values(uetas),
                unp.std_devs(uetas),
                label=f'{series.label}, ${pol_axis}$-polarized',
                marker=marker,
                **(series.plot_kw | linestyle_dict),
            )

            # TODO clean this duplicative code up... a lot
            if finesse_func is not None:
                freq_space = np.linspace(*expand_range(freqs, factor=1.15))

                ax_eta.plot(
                    freq_space / freqscale,
                    unp.nominal_values(get_cooperativity(freq_space, finesse_func)),
                    **series.plot_kw,
                )

                transition_freqs = np.abs(series.transition_frequencies())
                transition_etas = unp.nominal_values(get_cooperativity(transition_freqs, finesse_func))
                ax_eta.plot(
                    transition_freqs / freqscale,
                    transition_etas,
                    marker='|',
                    color='0.5',
                    linestyle='None',
                )
                for n, freq, eta in zip(series.n_range, transition_freqs, transition_etas):
                    ax_eta.annotate(
                        str(n),  # (f'$n = {n}$' if n == max(series.n_range) else f'${n}$'),
                        xy=(freq / freqscale, eta),
                        xytext=(0, 7),
                        textcoords='offset points',
                        color='0.5',
                        horizontalalignment='center',
                        verticalalignment='baseline',
                        fontsize='x-small',
                    )

        if ax is None:
            unit = {1: 'Hz', 1e+3: 'kHz', 1e+6: 'MHz', 1e+9: 'GHz'}[freqscale]
            ax_eta.set_xlabel(f'Frequency ({unit})')
            ax_eta.set_ylabel('Cooperativity')
            ax_eta.set_yscale('log')

    def cooperativity_summary_plot(
            self,
            transition_spec: arc.AlkaliAtom | Sequence[RydbergTransitionSeries] = arc.Cesium(),
    ):
        serieses: Sequence[RydbergTransitionSeries]
        if isinstance(transition_spec, arc.AlkaliAtom):
            serieses = rydtools.get_common_series(transition_spec)
        elif isinstance(transition_spec, Sequence):
            serieses = transition_spec
        else:
            assert_never(transition_spec)

        fig, axs = plt.subplots(figsize=(15, 8), nrows=3, ncols=3, sharex=True, layout='constrained')

        (ax_d, ax_e, ax_g), (ax_gamma, ax_kappa, ax_fin), (ax_ggamma, ax_gkappa, ax_eta) = axs.T
        ax_d.set_ylabel('Dipole matrix element [$ea_0$]')

        # upper n in transition
        # n_upper = np.arange(38, 50, dtype=np.float64)
        # n_lower = n_upper - 1
        # dipole_mat_elt = n_upper**2 / np.sqrt(2)

        ### these analytical values (from Haroche book) agree with the ARC calculations

        # plot_single_rydberg_line(
        #     (n_lower**(-2) - n_upper**(-2)) * rydfreq,
        #     n_upper**2 / np.sqrt(2),
        #     n_upper,
        #     label=r'$(n-1)C \to nC$, analytic',
        #     ax=ax_d,
        # )

        # plot_single_rydberg_line(
        #     (n_lower**(-2) - n_upper**(-2)) * rydfreq,
        #     3/4 * n_upper**5 * scipy.constants.alpha**(-3) / (2 * pi * scipy.constants.value('Rydberg constant times c in Hz')),
        #     n_upper,
        #     label=r'$nC$, analytic',
        #     ax=ax_gamma,
        # )

        cavity_longi_inds = np.arange(self.q_range[0], self.q_range[1] + 1)

        cavity_freqs = self.geometry.paraxial_frequency(cavity_longi_inds, 0)
        vac_fields = self.geometry.waist_vacuum_field_fromfreq(cavity_freqs)
        ax_e.plot(
            cavity_freqs,
            vac_fields,
            marker='.',
            color='0.5',
        )
        for i in [0, -1]:
            ax_e.annotate(
                f'$q = {int(cavity_longi_inds[i]):d}$', (cavity_freqs[i], vac_fields[i]),
            )

        ax_e.set_title(r'$E_\text{rms} \sim \omega$')
        ax_e.set_ylabel('Vacuum rms field [V/m]')

        for pol in [+1, -1]:
            highest_finesses = self.highest_finesse_values(lambda q, p: p == pol)

            pol_axis = 'x' if pol == +1 else 'y'
            marker = '.' if pol == +1 else 'x'
            freqs = highest_finesses['freq']
            fins_n = unp.nominal_values(highest_finesses['finesse'])
            fins_s = unp.std_devs(highest_finesses['finesse'])
            ufins = unp.uarray(fins_n, fins_s)

            finesse_kw = dict(
                marker=marker,
                color='0.5',
            )

            ax_fin.errorbar(
                freqs,
                fins_n,
                fins_s,
                **finesse_kw,
            )

            ukappas = self.geometry.fsr_u / ufins  # 2pi Hz
            ax_kappa.errorbar(
                freqs,
                unp.nominal_values(ukappas),
                unp.std_devs(ukappas),
                **finesse_kw,
            )

        for series in serieses:
            series.plot_numbers(ax_d, ax_gamma)
            power_laws = series.power_law_params()

            for pol_ind, pol in enumerate([+1, -1]):
                highest_finesses = self.highest_finesse_values(lambda q, p: p == pol)

                pol_axis = 'x' if pol == +1 else 'y'
                marker = '.' if pol == +1 else 'x'
                freqs = highest_finesses['freq']
                fins_n = unp.nominal_values(highest_finesses['finesse'])
                fins_s = unp.std_devs(highest_finesses['finesse'])
                ufins = unp.uarray(fins_n, fins_s)
                ukappas = self.geometry.fsr / ufins  # 2pi * Hz

                # convert units from e a_0 to SI
                d_interp = power_laws['d'](freqs) * scipy.constants.elementary_charge * scipy.constants.value('Bohr radius')
                vac_fields = self.geometry.waist_vacuum_field_fromfreq(freqs)

                # 2g, in 2pi Hz
                vacuum_rabi_freqs = 2 * d_interp * vac_fields / scipy.constants.Planck / (1 if series.polarization == 0 else np.sqrt(2))

                if pol_ind == 0:
                    ax_g.plot(
                        freqs,
                        vacuum_rabi_freqs,
                        marker='.',
                        **series.plot_kw,
                    )

                ugammas = 1 / (2 * pi * power_laws['lifetime'](freqs))  # in 2pi Hz
                uetas = vacuum_rabi_freqs**2 / (ugammas * ukappas)

                figs_of_merit = [uetas, vacuum_rabi_freqs / ugammas, vacuum_rabi_freqs / ukappas]
                axs_fom = [ax_eta, ax_ggamma, ax_gkappa]
                for ax, ufom in zip(axs_fom, figs_of_merit):
                    ax.errorbar(
                        freqs,
                        unp.nominal_values(ufom),
                        unp.std_devs(ufom),
                        label=f'{series.label}, ${pol_axis}$-polarized',
                        marker=marker,
                        **series.plot_kw,
                    )

        ax_d.set_title(R'$d \sim \omega^{-2/3}$')
        ax_d.legend(fontsize='xx-small')

        ax_g.set_ylabel(R'Vacuum Rabi frequency $2g$ [$2\pi \times$ Hz]')
        ax_g.set_title(R'$g \sim \omega^{1/3}$')

        ax_gamma.set_title(R'$\Gamma^{-1}_\text{low-l} \sim \omega^{-1}$, $\Gamma^{-1}_C \sim \omega^{-5/3}$')
        ax_gamma.set_ylabel(R'State lifetimes [s]')

        ax_gamma.legend(ncols=2, fontsize='xx-small')

        ax_kappa.set_title(R'$\kappa$')
        ax_kappa.set_ylabel(R'Cavity linewidth [$2\pi\times$ Hz]')

        ax_fin.set_ylabel('Measured cavity finesse')

        ax_eta.set_ylabel(R'Cooperativity $4g^2/\kappa\Gamma$')
        ax_eta.legend(ncols=2, fontsize='xx-small')

        ax_ggamma.set_title(R'$2g/\Gamma_{\text{low-l}} \sim \omega^{-2/3}$, $2g/\Gamma_C \sim \omega^{-4/3}$')
        ax_ggamma.set_ylabel(R'$2g/\Gamma$')
        ax_gkappa.set_ylabel(R'$2g/\kappa$')

        for ax in axs.flatten():
            ax.set_xscale('log')
            ax.set_yscale('log')

        # axs[-1].set_xscale('log')
        fig.supxlabel('Frequency [Hz]')
