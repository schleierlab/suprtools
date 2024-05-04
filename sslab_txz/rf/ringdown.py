from pathlib import Path

import h5py
import lmfit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from numpy.typing import NDArray
from scipy.constants import pi

from sslab_txz.plotting import sslab_style


class Ringdown():
    functional_form = (
        R'$\tilde{s}_0\, \exp(-\frac{1}{2}\kappa t-i\Delta\omega t) '
        R'+ \tilde{s}_\infty$'
    )

    s21: NDArray[np.complex128]
    t: NDArray[np.float64]
    frequency: float

    def __init__(self, h5path: str):
        with h5py.File(h5path, 'r') as f:
            self.s21 = f['data/s_params/s21/real'][:] \
                + 1j * f['data/s_params/s21/imag'][:]
            self.t = f['data/times'][:]
            self.frequency = f.attrs['frequency']
            self.temperatures = {
                key: f.attrs[key]
                for key in [
                    'temperature_still',
                    'temperature_sample',
                    'temperature_still_init',
                    'temperature_sample_init',
                ]
            }

            self.stage_positions = {
                'aaj': f.attrs['aaj_position'],
                'aak': f.attrs['aak_position'],
            }

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

    def fit_model(self, init_params=dict()):
        self.model = lmfit.Model(self.ringdown_shape)

        guess_offset = self.s21[-10:].mean()
        guess_prefactor = self.s21[0] - guess_offset

        lookahead_ind = np.searchsorted(self.t, 0.1 * self.t[-1])
        lookahead_time = self.t[lookahead_ind]
        lookahead_s21 = self.s21[lookahead_ind] - guess_offset
        guess_fwhm = np.log(np.abs(guess_prefactor / lookahead_s21)) \
            / lookahead_time / pi

        s21_scale = 0.5 * np.max(np.abs(self.s21 - self.s21.mean()))
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
        params = self.model.make_params(**(default_init_params | init_params))

        fit_range_trunc_ind = 0
        self.s21_fit = self.s21[fit_range_trunc_ind:]
        self.t_fit = self.t[fit_range_trunc_ind:]
        self.fit = self.model.fit(
            self.s21_fit,
            params=params,
            t=self.t_fit,
        )

    def visualize(self, onering: bool = False):
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
            rf.complex_2_db(self.s21),
            label=fr'$\omega/2\pi$ = {self.frequency / 1e+9:.9f} GHz',
        )

        axs['deg'].plot(
            1e+6 * self.t,
            np.unwrap(rf.complex_2_degree(self.s21), period=360),
        )

        reim_scale = 1e+6
        reim_scale_latex = '10^6'
        axs['reim'].plot(
            *rf.complex_2_reim(reim_scale * self.s21),
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
            onering_path = Path(__file__).parent.parent / 'img/onering_wikipedia.png'
            with open(onering_path, 'rb') as file:
                im = matplotlib.image.imread(file)
            ax_ring = fig.add_axes((0, 0, 1, 1))
            ax_ring.imshow(
                im,
                alpha=0.1,
                zorder=-3,
            )
            ax_ring.axis('off')

        return fig, axs
