from typing import ClassVar, cast

import numpy as np
import scipy.optimize
import uncertainties
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike
from scipy.constants import c, pi
from uncertainties import unumpy as unp

from sslab_txz.fp_theory.geometry import SymmetricCavityGeometry
from sslab_txz.plotting import sslab_style


class TEM00Fit:
    freq_formula: ClassVar[str] = (
        R'$\frac{\omega}{2\pi} = \frac{c}{2L} ('
            R'q + \frac{\cos^{-1}\overline{g}}{\pi} '  # noqa: E131
            R'- \frac{1 \pm 2\eta\,\cos\phi}{2\pi k\overline{R}}'
            R'+ \frac{\alpha_2}{2\pi(k\overline{R})^2}'
        R')$'
    )

    def __init__(self, qs: ArrayLike, pols: ArrayLike, freqs: ArrayLike):
        self.qs = qs
        self.pols = pols
        self.freqs = np.asarray(freqs)

        freqs_std = unp.std_devs(self.freqs)
        self.freqs_std = None if np.any(freqs_std == 0) else freqs_std

    @classmethod
    def _tem00_fitfunc_nextorder(cls, qpol, length, mean_curv_rad, eta_astig, alpha2):
        q, pol = qpol
        return cls._tem00_freqs(q, pol, length, mean_curv_rad, eta_astig, alpha2)

    @staticmethod
    def _tem00_freqs(q, pol, length, mean_curv_rad, eta_astig, alpha2):
        q = np.asarray(q)
        pol = np.asarray(pol)

        geo = SymmetricCavityGeometry(length, mean_curv_rad, eta_astig)
        parax_contrib = geo.paraxial_frequency(q, n_total=0)  # geo.fsr * (q + unp.arccos(geo.g)/pi)
        k = 2 * pi * parax_contrib / c

        post = - geo.fsr / (2*pi*k*mean_curv_rad)
        vplusa = - geo.fsr * pol * eta_astig / (pi*k*mean_curv_rad)

        second_order = alpha2 * geo.fsr / (2 * pi * (k * mean_curv_rad)**2)

        return parax_contrib + post + vplusa + second_order

    def fit(self, p0):
        popt, pcov = scipy.optimize.curve_fit(
            self._tem00_fitfunc_nextorder,
            [self.qs, self.pols],
            list(unp.nominal_values(self.freqs)),
            sigma=self.freqs_std,
            p0=p0,
            bounds=(
                (30e-3, 40e-3, -0.4, -10),
                (70e-3, 45e-3, +0.4, +10),
            )
        )
        self.upopt = uncertainties.correlated_values(popt, pcov)

    def geometry(self):
        return SymmetricCavityGeometry(*unp.nominal_values(self.upopt[:3]))

    def __call__(self, q, pol):
        return self._tem00_freqs(q, pol, *self.upopt)

    def _fit_info_box(self, loc):
        upopt = self.upopt
        fitstr = '\n'.join([
            f'$L = {upopt[0]*1e3:S}$ mm',
            f'  FSR: ${c/(2*upopt[0])/1e+9:LS}$ GHz',
            f'$\\overline{{R}} = {upopt[1]*1e3:S}$ mm',
            f'  $\\overline{{g}} = {1 - upopt[0]/upopt[1]:S}$',
            f'$\\eta\\,\\cos\\phi = {upopt[2]:S}$',
            '  if $\\phi = 0$:',
            f'  $R_x = {upopt[1]*1e3 / (1 + upopt[2]):S}$ mm',
            f'  $R_y = {upopt[1]*1e3 / (1 - upopt[2]):S}$ mm',
            f'$\\alpha_2 = {upopt[3]:S}$',
        ])

        at = AnchoredText(
            fitstr,
            prop=dict(size='small'),
            frameon=True,
            loc=loc,
            alpha=0.7,
        )
        at.patch.set_boxstyle(
            'round',
            pad=0.,
            rounding_size=0.2,
        )
        at.patch.set(
            edgecolor='0.7',
            alpha=0.7,
        )

        return at

    def plot(self):
        fig, axs = plt.subplots(
            figsize=(6, 4),
            nrows=2,
            sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1]),
            constrained_layout=True,
        )
        axs = cast(tuple[Axes, Axes], axs)
        ax_freq, ax_resid = axs

        plot_q_range = np.arange(min(self.qs), max(self.qs) + 1)
        fit_fsr = self.geometry().fsr

        polstrs = {+1: '$x$', -1: '$y$'}
        markers = {+1: '.', -1: 'x'}
        for pol in [+1, -1]:

            polmask = (self.pols == pol)
            plot_qs = self.qs[polmask]
            plot_freqs = unp.nominal_values(self.freqs[polmask])

            # actual data
            line, _, _ = ax_freq.errorbar(
                plot_qs,
                (plot_freqs - plot_qs * fit_fsr) / 1e+9,
                yerr=self.freqs_std[polmask]/1e+9,
                linestyle='None',
                color='C0',
                marker=markers[pol],
                label=f'{polstrs[pol]}-polarized'
            )

            ax_freq.plot(
                plot_q_range,
                (unp.nominal_values(self(plot_q_range, pol)) - plot_q_range * fit_fsr) / 1e+9,
                linestyle='dashed',
                color=line.get_color(),
            )

            ax_resid.errorbar(
                plot_qs,
                (plot_freqs - unp.nominal_values(self(plot_qs, pol))) / 1e+3,
                yerr=self.freqs_std[polmask]/1e+3,
                capsize=2,
                linestyle='None',
                color=line.get_color(),
                marker=markers[pol],
            )

        fit_format_at = AnchoredText(
            self.freq_formula,
            prop=dict(size='small'),
            frameon=False,
            bbox_to_anchor=(0.2, 0.01),
            bbox_transform=axs[0].transAxes,
            loc='lower left',
            alpha=0.7,
        )

        ax_freq.set_ylabel('Frequency mod FSR [GHz]')
        ax_resid.set_ylabel('Residuals [kHz]')
        axs[-1].set_xlabel('Longitudinal mode index $q$')
        axs[0].add_artist(self._fit_info_box(loc='lower right'))
        axs[0].add_artist(fit_format_at)
        axs[0].legend()
        for ax in axs:
            sslab_style(ax)

        return fig, axs
