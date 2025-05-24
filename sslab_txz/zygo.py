from __future__ import annotations

from typing import Any, Optional, assert_never

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import seaborn as sns
import uncertainties
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.typing import NDArray
from uncertainties import unumpy as unp

from sslab_txz.plotting import mpl_usetex
from sslab_txz.plotting.units import Units


# TODO: maybe deprecate in favor of `prysm` package?
class ZygoProfile:
    def __init__(self, h5: h5py.File):
        self.h5 = h5

    @classmethod
    def from_h5(cls, fname):
        f = h5py.File(fname)
        return cls(f)

    @property
    def lateral_resolution(self):
        return list(self.h5['Attributes'].values())[1].attrs[
            'Surface Data Context.Lateral Resolution'
        ][0][1]

    def crop_data(self, xrange_um: tuple[float, float], yrange_um: tuple[float, float]):
        mirror_profile = np.array(self.h5['Measurement']['Surface'])
        mirror_profile[mirror_profile > 1e100] = 75000

        xrange_um_arr = np.asarray(xrange_um)
        yrange_um_arr = np.asarray(yrange_um)

        x_indrange = np.clip(
            512 + np.round(xrange_um_arr * 1e-6 / self.lateral_resolution).astype(int), 0, 1024
        )
        y_indrange = np.clip(
            512 + np.round(yrange_um_arr * 1e-6 / self.lateral_resolution).astype(int), 0, 1024
        )
        x_minind, x_maxind = x_indrange
        y_minind, y_maxind = y_indrange
        xslice = slice(x_minind, x_maxind)
        yslice = slice(y_minind, y_maxind)

        return xslice, yslice, mirror_profile[yslice, xslice]

    def quadratic_fit(
        self, xrange_um: tuple[float, float], yrange_um: tuple[float, float]
    ) -> QuadraticFit:
        xy_um_full: NDArray = np.arange(-512, 512) * self.lateral_resolution * 1000000

        xslice, yslice, mirror_data_window = self.crop_data(xrange_um, yrange_um)

        x_um, y_um = xy_um_full[xslice], xy_um_full[yslice]
        xs, ys = np.meshgrid(x_um, y_um)
        xs_flat, ys_flat = xs.ravel(), ys.ravel()
        xys_flat = np.vstack([xs_flat, ys_flat]).T

        b = mirror_data_window.ravel()
        popt, pcov = scipy.optimize.curve_fit(QuadraticFit.quadratic_fit_function, xys_flat, b)
        upopt = uncertainties.correlated_values(popt, pcov)

        return QuadraticFit(self, mirror_data_window, upopt, x_um / 1e6, y_um / 1e6)


class QuadraticFit:
    def __init__(
        self, zygo_profile: ZygoProfile, mirror_data_window, upopt, xrange: NDArray, yrange: NDArray
    ):
        self.zygo_profile: ZygoProfile = zygo_profile
        self.xrange = xrange
        self.yrange = yrange
        self.upopt = upopt
        self.mirror_data_window = mirror_data_window

    @property
    def popt(self):
        return unp.nominal_values(self.upopt)

    def plot_raw_data(self, fig: Optional[Figure] = None, ax: Optional[Axes] = None):
        fig, ax, _ = self._validate_figax(fig, ax)

        x_um = self.xrange * 1e6
        y_um = self.yrange * 1e6
        im = ax.pcolormesh(x_um, y_um, self.mirror_data_window / 1000)

        ax.set_aspect('equal')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(Rf'$z$ ({Units.UM.mplstr()})')
        ax.set_title('Raw data')

    def plot_summary(self):
        fig, axs = plt.subplots(figsize=(10, 9), ncols=2, nrows=2, layout='compressed')

        self.plot_raw_data(fig=fig, ax=axs[0, 0])
        self.plot_linear_residuals(fig=fig, ax=axs[0, 1])
        self.plot_residuals(fig=fig, ax=axs[1, 0])
        self.plot_correlations(ax=axs[1, 1])

        a0, a_x, ay, axx, axy, ayy = self.upopt
        curv_rads_mm, _ = self.curvature()
        suptitles = [
            R'$z = a_0 + a_x x + a_y y + \frac{1}{2} (a_{xx} x^2 + 2a_{xy} xy + a_{yy} y^2)$, coefficients in units of nm/$\mu$m$^k$',
            Rf'$\mathbf{{a}} = {a0:SL}, ({a_x:SL}, {ay:SL}), ({axx:SL}, {axy:SL}, {ayy:SL})$',
            f'$R_{{1,2}} = {curv_rads_mm[0]:SL}, {curv_rads_mm[1]:SL}$ mm',
        ]
        fig.suptitle('\n'.join(suptitles))

        axs[1, 1].set_aspect('equal')
        axs[1, 1].set_title('Uncertainty correlations')

        for ax in axs.ravel()[:3]:
            ax.set_xlabel(Rf'$x$ ({Units.UM.mplstr()})')
            ax.set_ylabel(Rf'$y$ ({Units.UM.mplstr()})')

    def curvature(self):
        axx, axy, ayy = self.upopt[3:]

        tr = axx + ayy
        det = axx * ayy - axy**2

        eigvals = (unp.sqrt(tr**2 - 4 * det) * np.array([-1, 1]) + tr) / 2
        _, eigvecs = np.linalg.eigh(unp.nominal_values([[axx, axy], [axy, ayy]]))

        curvature_radii = 1 / eigvals
        return (curvature_radii, eigvecs.T)

    # TODO refactor this out
    @staticmethod
    def _validate_figax(
        fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> tuple[Figure, Axes, bool]:
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
            return fig, ax, True
        elif (fig is not None) and (ax is not None):
            return fig, ax, False
        else:
            raise ValueError('Must pass both or neither `fig` and `ax`')

    def plot_residuals(
        self,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        cmap: str | Colormap = 'BrBG_r',
        norm: Normalize = Normalize(-50, +50),
        hscale=1e6,
        vscale=1e9,
        show_principal_axes: bool = True,
        scale: None | int | float | tuple[int | float, dict[str, Any]] = None,
    ) -> tuple[Figure, Axes, ScalarMappable]:
        """
        scale: float, optional
            If supplied, draw a scale bar with length scale
            (in scaled horizontal units, as specified by hscale)
        """
        fig, ax, is_ax_fresh = self._validate_figax(fig, ax)

        x_um = self.xrange * 1e6
        y_um = self.yrange * 1e6

        quadratic_residuals_nm = self.mirror_data_window - self.quadratic_fit_result(x_um, y_um)
        ax.set_aspect('equal')
        im = ax.pcolormesh(
            self.xrange * hscale,
            self.yrange * hscale,
            quadratic_residuals_nm * vscale / 1e9,
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )

        if scale is not None:
            style: Optional[dict[str, Any]]
            if isinstance(scale, int | float):
                scale_val = scale
                style = None
            elif isinstance(scale, tuple):
                scale_val, style = scale
            else:
                assert_never(scale)

            asb = AnchoredSizeBar(
                ax.transData,
                scale_val,
                (
                    Rf'\SI{{{scale_val}}}{{\micro\meter}}'
                    if mpl_usetex() else
                    Rf'{scale_val} $\mu$m'
                ),
                loc=8,
                pad=0.1,
                borderpad=1,
                sep=5,
                frameon=False,
            )
            if style is not None:
                asb.patch.set(**style)
            ax.add_artist(asb)

        ra = np.abs(quadratic_residuals_nm).mean()
        rq = quadratic_residuals_nm.std()
        roughness_str = f'Ra, Rq = {ra:.1f}, {rq:.1f} {Units.NM.mplstr()}'
        title_lines = [roughness_str]
        if is_ax_fresh:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f'$z - z^{{(2)}}$ ({Units.NM.mplstr()})')
            ax.set_xlabel(Rf'$x$ ({Units.UM.mplstr()})')
            ax.set_ylabel(Rf'$y$ ({Units.UM.mplstr()})')
            title_lines = ['Quadratic fit residuals', roughness_str]
        ax.set_title('\n'.join(title_lines))

        if show_principal_axes:
            _, eigvecs = self.curvature()
            for eigvec in eigvecs:
                min_dim_um = 1e6 * min(
                    self.yrange[-1] - self.yrange[0], self.xrange[-1] - self.xrange[0]
                )
                arrow_origin = 1e+6 * np.array([self.xrange.mean(), self.yrange.mean()])
                ax.arrow(*arrow_origin, *(eigvec * 0.2 * min_dim_um))
                ax.arrow(*arrow_origin, *(eigvec * 0.2 * min_dim_um))

        return fig, ax, im

    def plot_linear_residuals(self, fig: Optional[Figure] = None, ax: Optional[Axes] = None):
        fig, ax, _ = self._validate_figax(fig, ax)

        x_um = self.xrange * 1e6
        y_um = self.yrange * 1e6

        ax.set_aspect('equal')
        im = ax.pcolormesh(x_um, y_um, self.mirror_data_window - self.linear_fit_result(x_um, y_um))
        cbar = fig.colorbar(im, ax=ax)
        ax.set_title('Linear fit residuals')
        cbar.set_label(f'$z - z^{(1)}$ ({Units.NM.mplstr()})')

        ax.set_xlabel(Rf'$x$ ({Units.UM.mplstr()})')
        ax.set_ylabel(Rf'$y$ ({Units.UM.mplstr()})')

    def linear_fit_result(self, x, y):
        a0, ax, ay = self.popt[0:3]
        return a0 + ax * x.reshape(1, -1) + ay * y.reshape(-1, 1)

    def quadratic_fit_result(self, x, y):
        a0, ax, ay, axx, axy, ayy = self.popt
        x_reshape, y_reshape = x.reshape(1, -1), y.reshape(-1, 1)

        return (
            a0
            + ax * x_reshape
            + ay * y_reshape
            + 0.5 * (axx * x_reshape**2 + ayy * y_reshape**2)
            + axy * x_reshape * y_reshape
        )

    @staticmethod
    def quadratic_fit_function(xy, a0, ax, ay, axx, axy, ayy):
        linear_term = xy @ np.array([ax, ay])

        quadratic_form = np.array([[axx, axy], [axy, ayy]])
        quadratic_term = 0.5 * ((xy @ quadratic_form) * xy).sum(axis=1)

        return a0 + linear_term + quadratic_term

    def plot_correlations(self, ax: Optional[Axes] = None):
        if ax is None:
            _, ax = plt.subplots()

        corr_mat = uncertainties.correlation_matrix(self.upopt)
        # stds = np.sqrt(np.diagonal(pcov))
        # corr_mat = pcov / stds.reshape(1, -1) / stds.reshape(-1, 1)

        coeff_names = ['$a_0$', '$a_x$', '$a_y$', '$a_{xx}$', '$a_{xy}$', '$a_{yy}$']
        sns.heatmap(
            corr_mat,
            annot=True,
            xticklabels=coeff_names,
            yticklabels=coeff_names,
            fmt='.2f',
            ax=ax,
            center=0,
        )
