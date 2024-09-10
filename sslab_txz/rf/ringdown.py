# for forward references; needed until python 3.13
from __future__ import annotations

import importlib.resources
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, cast, overload

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import skimage.restoration
import skrf as rf
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from lmfit.model import ModelResult
from lmfit.models import LinearModel
from numpy.typing import NDArray
from PIL import Image
from scipy.constants import pi
from tqdm import tqdm
from uncertainties import UFloat

from sslab_txz.plotting import sslab_style
from sslab_txz.rf.cw import CWMeasurement
from sslab_txz.rf.errors import FitFailureError

from . import img


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
        axs_flat = cast(Sequence[plt.Axes], axs.flatten())

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

    def visualize(self, onering: bool = False):
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

    def plot_cartesian(self, scale=1, ax: Optional[plt.Axes] = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            ax = cast(plt.Axes, ax)

        ax.set_aspect(1)
        ax.set_box_aspect(1)
        ax.plot(
            *rf.complex_2_reim(scale * self.s21[0]),
            **kwargs,
        )

    def visualize(self, onering: bool = False):
        s21 = self.s21[0]

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

        axs['db'].plot(
            1e+6 * self.t,
            rf.complex_2_db(s21),
            label=fr'$\omega/2\pi$ = {self.frequency / 1e+9:.9f} GHz',
        )

        axs['deg'].plot(
            1e+6 * self.t,
            np.unwrap(rf.complex_2_degree(s21), period=360),
        )

        reim_scale = 1e+6
        reim_scale_latex = '10^6'
        self.plot_cartesian(ax=axs['reim'], scale=reim_scale)

        if hasattr(self, 'fit') and hasattr(self.fit, 'uvars'):
            model_val = self.fit.eval(t=self.t_fit)
            axs['db'].plot(
                1e+6 * self.t_fit,
                rf.complex_2_db(model_val),
                label=(
                    fr'$\kappa/2\pi = {self.fit.uvars["fwhm"]:SL}$ Hz'
                )
            )
            axs['deg'].plot(
                1e+6 * self.t_fit,
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

        axs['db'].legend(fontsize='x-small')
        axs['deg'].legend(fontsize='x-small')
        axs['reim'].legend(fontsize='x-small')

        axs['db'].set_ylabel('$|S_{21}|$ [dB]')
        axs['deg'].set_ylabel(R'$\angle S_{21}$ [deg]')
        axs['deg'].set_xlabel(R'Time [$\mu$s]')
        axs['reim'].set_xlabel(fR'${reim_scale_latex} \times \operatorname{{Re}} S_{{21}}$')
        axs['reim'].set_ylabel(fR'${reim_scale_latex} \times \operatorname{{Im}} S_{{21}}$')

        for ax in axs.values():
            sslab_style(ax)

        if onering:
            imfile = importlib.resources.files(img) / 'onering_wikipedia.png'

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

        offset_guess = self.ringdown_set.s21[:, -20:].mean()
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

    def plot_fit(self, ax: Optional[plt.Axes] = None, xscale=1):
        if ax is None:
            fig, ax = plt.subplots()
        if self.fit_result is None:
            raise ValueError
        if not hasattr(self.fit_result, 'uvars'):
            raise ValueError

        uvars = self.fit_result.uvars
        offset = uvars['offset_re'].n + 1j * uvars['offset_im'].n
        ax.plot(
            xscale * self.t,
            np.mean(np.abs(self.ringdown_set.s21 - offset)**2, axis=0)
        )
        ax.plot(
            xscale * self.t,
            uvars['a0'].n * np.exp(-2 * pi * uvars['fwhm'].n * self.t) + uvars['const'].n,
            color='red',
            label=Rf'$\kappa/2\pi = {uvars['fwhm']:SL} $ Hz',
        )
        ax.set_yscale('log')
        ax.legend()

    def _repr_html_(self) -> Optional[str]:
        if self.fit_result is None:
            return None
        if not hasattr(self.fit_result, 'uvars'):
            return f'Unsuccessful collective fit of {repr(self.ringdown_set)}'

        fig, ax = plt.subplots()
        self.plot_fit(ax)
        return fig._repr_html_()


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
            ax_cart, ax_db = cast(tuple[plt.Axes, ...], axs)
        else:
            fig, axs = plt.subplot_mosaic(
                [
                    ['cart', 'db'],
                    ['cart', 'phase'],
                ],
                layout='constrained',
                figsize=(9.6, 4),
            )
            axs = cast(Mapping[str, plt.Axes], axs)
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
