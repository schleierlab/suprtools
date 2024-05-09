# for forward references; needed until python 3.13
from __future__ import annotations

import importlib.resources
from typing import Iterable, Mapping, Optional, Self

import h5py
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from lmfit import Parameters
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from scipy.constants import pi

from sslab_txz.plotting import sslab_style

from . import img


class RingdownSet():
    functional_form = (
        R'$\tilde{s}_0\, \exp(-\frac{1}{2}\kappa t-i\Delta\omega t) '
        R'+ \tilde{s}_\infty$'
    )

    s21: NDArray[np.complex128]
    t: NDArray[np.float64]
    frequency: float

    def __init__(
            self,
            t: ArrayLike,
            s21: ArrayLike,
            frequency: float,
            temperatures: Optional[Mapping[str, float]],
            stage_positions: Optional[Mapping[str, float]],
    ):
        '''
        Parameters
        ----------
        t: arraylike, (m,)
            Time points in the ringdown(s)
        s21: arraylike, (n, m) or (m,)
        frequency: float
        temperatures: Mapping[str, float], optional
            Temperatures, measured as a mapping from measurement name
            to temperature in kelvins.
        stage_positions: Mapping[str, float], optional
            Locations of the stages as a mapping from stage name
            to position in meters.
        '''
        self.t = np.array(t)
        self.s21 = np.array(s21).reshape(-1, len(self.t))

        self.frequency = frequency
        self.temperatures = temperatures
        self.stage_positions = stage_positions

    @classmethod
    def from_h5(cls, h5path: str) -> Self:
        '''
        h5path: str
            Path to the H5 file
        '''
        with h5py.File(h5path, 'r') as f:
            s21: NDArray[np.complex128] = f['data/s_params/s21/real'][:] \
                + 1j * f['data/s_params/s21/imag'][:]
            if len(s21.shape) == 1:
                s21 = s21.reshape(1, -1)
            t = f['data/times'][:]
            frequency = f.attrs['frequency']
            temperatures = {
                key: f.attrs[key]
                for key in [
                    'temperature_still',
                    'temperature_sample',
                    'temperature_still_init',
                    'temperature_sample_init',
                ]
            }

            stage_positions = {
                'aaj': f.attrs['aaj_position'],
                'aak': f.attrs['aak_position'],
            }

        return cls(t, s21, frequency, temperatures, stage_positions)

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

    def __getitem__(self, key: int) -> Ringdown:
        return Ringdown(self.t, self.s21[key], self.frequency, None, self.stage_positions)

    def __len__(self) -> int:
        return len(self.s21)

    def _init_params_from_s21(
            self,
            s21: NDArray,
            suffix: Optional[str] = None,
            init_params: Mapping[str, Mapping] = dict(),
    ) -> Parameters:
        guess_offset = s21[-10:].mean()
        guess_prefactor = s21[0] - guess_offset

        lookahead_ind = np.searchsorted(self.t, 0.1 * self.t[-1])
        lookahead_time = self.t[lookahead_ind]
        lookahead_s21 = s21[lookahead_ind] - guess_offset
        guess_fwhm = np.log(np.abs(guess_prefactor / lookahead_s21)) \
            / lookahead_time / pi

        s21_scale = 0.5 * np.max(np.abs(s21 - s21.mean()))

        full_suffix = '' if suffix is None else f'_{suffix}'
        default_init_params = dict(
            a0=dict(
                value=np.abs(guess_prefactor),
                min=0.5*np.abs(guess_prefactor),
                max=2*np.abs(guess_prefactor),
            ),
            phi0=dict(
                value=np.angle(guess_prefactor),
                min=-2*pi,
                max=2*pi,
            ),
            fwhm=dict(value=guess_fwhm, min=1, max=8e+3),
            delta_f=dict(value=0, min=-5e+3, max=+5e+3),
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
        full_init_params = default_init_params | init_params

        params = Parameters()
        for param_name, param_spec in full_init_params.items():
            params.add(
                param_name + full_suffix,
                **param_spec,
            )
        return params

    def fit_model(self, shared_params: Iterable[str] = ['fwhm', 'offset_re', 'offset_im']):
        fit_params: Parameters = sum(
            (self._init_params_from_s21(s21i, str(i)) for i, s21i in enumerate(self.s21)),
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

    def visualize(self, onering: bool = False):
        raise NotImplementedError


class Ringdown(RingdownSet):
    def fit_model(self, init_params=dict()):
        s21 = self.s21[0]
        self.model = lmfit.Model(self.ringdown_shape)
        params = self._init_params_from_s21(s21, init_params=init_params)

        fit_range_trunc_ind = 0
        self.s21_fit = s21[fit_range_trunc_ind:]
        self.t_fit = self.t[fit_range_trunc_ind:]
        self.fit = self.model.fit(
            self.s21_fit,
            params=params,
            t=self.t_fit,
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
        axs['reim'].plot(
            *rf.complex_2_reim(reim_scale * s21),
        )

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
                    fr'2\pi \times {self.fit.uvars["delta_f"]:SL}$ Hz'
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
