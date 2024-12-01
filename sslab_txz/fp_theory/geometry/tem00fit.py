import itertools
from typing import ClassVar, Literal, Optional, TypeAlias, assert_never, cast

import numpy as np
import scipy.optimize
import uncertainties
from lmfit import Model
from lmfit.model import ModelResult
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike
from scipy.constants import c, pi
from uncertainties import unumpy as unp

from sslab_txz.fp_theory.geometry import SymmetricCavityGeometry
from sslab_txz.plotting import sslab_style

InitParam: TypeAlias = float | dict | Literal[False]


class TEM00Fit:
    freq_formula: ClassVar[str] = (
        R'$\frac{\omega}{2\pi} = \frac{c}{2L} ('
            R'q + \frac{\cos^{-1}\overline{g}}{\pi} '  # noqa: E131
            R'- \frac{1 + \tilde{p} \frac{z_1}{\overline{R} - z_1} \pm 2\eta\,\cos\phi}{2\pi k\overline{R}}'  # noqa: E501
            R'+ \frac{\alpha_2}{2\pi(k\overline{R})^2}'
        R')$'
    )

    result: ModelResult

    def __init__(self, qs: ArrayLike, pols: ArrayLike, freqs: ArrayLike):
        self.qs = np.asarray(qs)
        self.pols = np.asarray(pols)
        self.freqs_u = freqs
        self.freqs = unp.nominal_values(self.freqs_u)

        freqs_std = unp.std_devs(self.freqs_u)
        self.freqs_std = None if np.any(freqs_std == 0) else freqs_std

    def _fitted(self):
        return hasattr(self, 'upopt')

    @classmethod
    def _tem00_fitfunc_nextorder(cls, qpol, length, mean_curv_rad, eta_astig, asphere_p, alpha2):
        q, pol = qpol
        return cls._tem00_freqs(q, pol, length, mean_curv_rad, eta_astig, asphere_p, alpha2)

    @staticmethod
    def _tem00_freqs(q, pol, length, mean_curv_rad, eta_astig, asphere_p, alpha2):
        q = np.asarray(q)
        pol = np.asarray(pol)

        geo = SymmetricCavityGeometry(length, mean_curv_rad, eta_astig, asphere_p)
        parax_contrib = geo.paraxial_frequency(q, n_total=0)  # geo.fsr * (q + unp.arccos(geo.g)/pi)
        k = 2 * pi * parax_contrib / c

        asphere_corr_factor = (1 + asphere_p * geo.z1 / (mean_curv_rad - geo.z1))
        post = - geo.fsr * asphere_corr_factor / (2*pi*k*mean_curv_rad)
        vplusa = - geo.fsr * pol * eta_astig / (pi*k*mean_curv_rad)

        second_order = alpha2 * geo.fsr / (2 * pi * (k * mean_curv_rad)**2)

        return parax_contrib + post + vplusa + second_order

    def fit(self, p0):
        popt, pcov = scipy.optimize.curve_fit(
            self._tem00_fitfunc_nextorder,
            [self.qs, self.pols],
            list(self.freqs),
            sigma=self.freqs_std,
            p0=p0,
            bounds=(
                (30e-3, 40e-3, -0.4, -0.4, -10),
                (70e-3, 45e-3, +0.4, +0.4, +10),
            )
        )
        self.upopt = uncertainties.correlated_values(popt, pcov)

    def lmfit(
            self,
            length: InitParam = dict(value=45e-3, min=30e-3, max=70e-3),
            mean_curv_rad: InitParam = dict(value=42e-3, min=40e-3, max=45e-3),
            eta_astig: InitParam = dict(value=0, min=-0.4, max=+0.4),
            asphere_p: InitParam = dict(value=0, min=-0.4, max=+0.4),
            alpha2: InitParam = dict(value=0, min=-10, max=+10)):

        def asdict(x: InitParam):
            match x:
                case float():
                    return dict(value=x)
                case dict():
                    return x
                case False:
                    return dict(value=0, vary=False)
                case _:
                    assert_never(x)

        model = Model(self._tem00_freqs, independent_vars=['q', 'pol'])
        params = model.make_params(
            length=asdict(length),
            mean_curv_rad=asdict(mean_curv_rad),
            eta_astig=asdict(eta_astig),
            asphere_p=asdict(asphere_p),
            alpha2=asdict(alpha2),
        )
        self.result = model.fit(
            self.freqs,
            params, q=self.qs, pol=self.pols,
            weights=(None if self.freqs_std is None else 1/self.freqs_std),
            nan_policy='omit',
        )
        return self.result

    def geometry(self, method='curve_fit'):
        if method == 'curve_fit':
            return SymmetricCavityGeometry(*self.upopt[:4])
        elif method == 'lmfit':
            uvars = self.result.uvars
            return SymmetricCavityGeometry(
                uvars['length'],
                uvars['mean_curv_rad'],
                uvars['eta_astig'],
                uvars['asphere_p'],
            )

    def __call__(self, q, pol, method='curve_fit'):
        if method == 'curve_fit':
            return self._tem00_freqs(q, pol, *self.upopt)

        kwargs = {
            key: self.result.uvars[key]
            for key in ['length', 'mean_curv_rad', 'eta_astig', 'asphere_p', 'alpha2']
        }
        # we don't use ModelResult.eval_uncertainty for uncertainties
        # in order to propagate correlated uncertainties properly
        # into the return value
        return self._tem00_freqs(q, pol, **kwargs)

    def _fit_info_box(self, loc, method: Literal['curve_fit', 'lmfit'] = 'curve_fit'):
        geo = self.geometry(method)
        if method == 'curve_fit':
            alpha2_u = self.upopt[4]
        elif method == 'lmfit':
            alpha2_u = self.result.uvars['alpha2']
        else:
            assert_never(method)

        fitstr = '\n'.join([
            f'$L = {geo.length_u*1e3:S}$ mm',
            f'  FSR: ${geo.fsr_u/1e+9:LS}$ GHz',
            f'$\\overline{{R}} = {geo.mirror_curv_rad_u*1e+3:S}$ mm',
            f'  $\\overline{{g}} = {geo.g_u:S}$',
            f'$\\eta\\,\\cos\\phi = {geo.eta_astig_u:S}$',
            '  if $\\phi = 0$:',
            f'  $R_x = {geo.rx_u*1e+3:S}$ mm',
            f'  $R_y = {geo.ry_u*1e3:S}$ mm',
            f'$\\tilde{{p}} = {geo.asphere_p_u:S}$',
            f'$\\alpha_2 = {alpha2_u:SL}$',
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

    def plot(self, method='curve_fit'):
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
        fit_fsr = self.geometry(method).fsr

        polstrs = {+1: '$x$', -1: '$y$'}
        markers = {+1: '.', -1: 'x'}
        for pol in [+1, -1]:

            polmask = (self.pols == pol)
            plot_qs = self.qs[polmask]
            plot_freqs = self.freqs[polmask]

            # actual data
            line, _, _ = ax_freq.errorbar(
                plot_qs,
                (plot_freqs - plot_qs * fit_fsr) / 1e+9,
                yerr=(self.freqs_std[polmask]/1e+9 if self.freqs_std is not None else None),
                linestyle='None',
                color='C0',
                marker=markers[pol],
                label=f'{polstrs[pol]}-polarized'
            )

            model_uvals_full = self(plot_q_range, pol, method=method)
            ax_freq.plot(
                plot_q_range,
                (unp.nominal_values(model_uvals_full) - plot_q_range * fit_fsr) / 1e+9,
                linestyle='dashed',
                color=line.get_color(),
            )

            model_uvals_predicted = self(plot_qs, pol, method=method)
            ax_resid.errorbar(
                plot_qs,
                (plot_freqs - unp.nominal_values(model_uvals_predicted)) / 1e+3,
                yerr=(self.freqs_std[polmask]/1e+3 if self.freqs_std is not None else None),
                capsize=2,
                linestyle='None',
                color=line.get_color(),
                marker=markers[pol],
            )

            ax_resid.fill_between(
                plot_q_range,
                -unp.std_devs(model_uvals_full) / 1e+3,
                unp.std_devs(model_uvals_full) / 1e+3,
                color='C0',
                alpha=0.3,
                linewidth=0,
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
        axs[0].add_artist(self._fit_info_box(loc='lower right', method=method))
        axs[0].add_artist(fit_format_at)
        axs[0].legend()
        for ax in axs:
            sslab_style(ax)

        return fig, axs

    def _repr_html_(self) -> Optional[str]:
        if not self._fitted():
            return None
        fig, _ = self.plot()
        return fig._repr_html_()

    def scan_segment_file(
            self,
            q_vals,
            pols=[+1, -1],
            span: int = 150000,
            npoints: int = 1001,
            ifbw: int = 100,
            n_avg: int = 1,
    ) -> str:
        header = ['# center span npoints ifbw n_avg']
        lines = (
            f'{self(q, pol).n:15_.0f} {span:_d} {npoints} {ifbw:_d} {n_avg}'
            for q in q_vals for pol in pols
        )

        return '\n'.join(itertools.chain(header, lines))
