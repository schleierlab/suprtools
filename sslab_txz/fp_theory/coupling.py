from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import matplotlib.projections
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sslab_txz.fp_theory.geometry._base import CavityGeometry
from sslab_txz.fp_theory.modes import VectorModeBasis


@dataclass
class CouplingConfig:
    no_xcoupling: ClassVar[CouplingConfig]
    config_nopolmix: ClassVar[CouplingConfig]
    all_xcoupling: ClassVar[CouplingConfig]
    paraxial: ClassVar[CouplingConfig]

    prop: bool = True
    prop_xcoupling: bool = False
    wave: bool = True
    wave_xcoupling: bool = False
    vec: bool = True
    vec_xcoupling: bool = False
    astig: bool = True
    astig_xcoupling: bool = False
    asphere: bool = True
    asphere_xcoupling: bool = False
    v_plus_a: bool = True


CouplingConfig.no_xcoupling = CouplingConfig()
CouplingConfig.config_nopolmix = CouplingConfig(v_plus_a=False)
CouplingConfig.all_xcoupling = CouplingConfig(
    prop_xcoupling=True,
    wave_xcoupling=True,
    vec_xcoupling=True,
    astig_xcoupling=True,
    asphere_xcoupling=True,
)
CouplingConfig.paraxial = CouplingConfig(
    prop=False,
    wave=False,
    vec=False,
    astig=True,
    asphere=False,
)


class NearConfocalCouplingMatrix():
    cavity_geo: CavityGeometry
    matrix: np.ndarray
    basis: VectorModeBasis
    longi_ind_base: int
    eigvals: np.ndarray
    eigvecs: np.ndarray

    def __init__(
            self,
            cavity_geo: CavityGeometry,
            matrix, basis: VectorModeBasis,
            longi_ind_base,
            compensation_term):
        self.cavity_geo = cavity_geo
        self.matrix = matrix
        self.basis = basis
        self.longi_ind_base = longi_ind_base

        raw_eigvals, eigvecs = np.linalg.eigh(matrix)
        self.eigvals = cavity_geo.fsr * (longi_ind_base + raw_eigvals + compensation_term)
        self.eigvecs = eigvecs.T

    def plot_coupling_matrix(self):
        fig, ax = plt.subplots(
            figsize=(6, 4),
            constrained_layout=True,
        )

        assert np.all(np.imag(self.matrix) == 0)
        cc = ax.imshow(np.real(self.matrix), cmap='coolwarm', vmin=-0.2, vmax=0.2)
        fig.colorbar(cc, ax=ax)

        ax.set_yticks(np.arange(len(self.basis)))
        ax.set_yticklabels(
            (f'${m.latex()}$' for m in self.basis),
            verticalalignment='center',
            fontsize='x-small',
        )

        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(len(self.basis)))
        ax.set_xticklabels(
            (f'${m.latex()}$' for m in self.basis),
            rotation=90,
            horizontalalignment='center',
            fontsize='x-small',
        )

        ax.tick_params(length=0)

        ax.set_xlabel(f'${self.basis.latex()}$')
        ax.set_ylabel(f'${self.basis.latex()}$')

        # TODO improve this
        blockborder_kwargs = dict(color='0.7')
        for i in range(1, 4+1):
            ax.axhline(i**2 - 0.5, **blockborder_kwargs)
            ax.axvline(i**2 - 0.5, **blockborder_kwargs)
        for i in range(1, 4):
            ax.axhline(4**2 + i**2 - 0.5, **blockborder_kwargs)
            ax.axvline(4**2 + i**2 - 0.5, **blockborder_kwargs)

    def plot_eigenvectors(self):
        fig, axs = plt.subplots(figsize=(8, 6), sharey=True, ncols=2, constrained_layout=True)
        d = len(self.matrix) // 2
        assert np.all(self.matrix[:d, d:] == 0)
        assert np.all(self.matrix[d:, :d] == 0)

        for ax, subblock in zip(axs[:2], [(d, None), (0, d)]):
            blockslice = slice(*subblock)
            submatrix = self.matrix[blockslice, blockslice]
            _, subeigvecs = np.linalg.eigh(submatrix)
            ax.imshow(np.abs(subeigvecs.T)**2, cmap='cividis', vmin=0, vmax=1)

            subbasis = list(itertools.islice(self.basis, *subblock))
            ax.set_xticks(np.arange(len(subbasis)))
            ax.set_xticklabels(
                (f'${m.latex()}$' for m in subbasis),
                horizontalalignment='center',
                rotation=90,
            )

        # fig.colorbar(cc, ax=axs[-1])

    def annotate_modes(
            self,
            inds=slice(None),
            offset=0,
            scaling=1e9,
            ax=None,
            label: Optional[float] = None,
            color=(lambda n: f'C{n//2+1}'),
            **kwargs):
        '''
        Parameters
        ----------
        offset: fsr_guess * offset_ind
        label: float, optional
            If specified, the y-position (in axis units) of the annotation labels.
        '''
        if ax is None:
            fig, ax = plt.subplots()

        if callable(color):
            colorfunc = color
        else:
            def colorfunc(_):
                return color

        for eigval, eigvec in zip(self.eigvals[inds], self.eigvecs[inds]):
            max_pop_ind = np.argmax(np.abs(eigvec)**2)
            transverse_ind = self.basis[max_pop_ind].n

            plot_color = colorfunc(transverse_ind)
            # color = f'C{transverse_ind//2 + 1}'
            plot_freq = (eigval - offset) / scaling

            axvline_default_kwargs = dict(linestyle='dashed', color=plot_color, alpha=0.35)
            axvline_kwargs = {**axvline_default_kwargs, **kwargs}
            ax.axvline(plot_freq, **axvline_kwargs)

            if label:
                ax.annotate(
                    f'$N = {transverse_ind}$',
                    (plot_freq, label),
                    (plot_freq, label),
                    rotation=90,
                    xycoords=ax.get_xaxis_transform(),
                    verticalalignment='center',
                    horizontalalignment='right',
                    fontsize='small',
                    color=plot_color,
                    alpha=0.7,
                )

    @staticmethod
    def inset_level_maker(locs, gap):
        '''
        Given sequence `locs` sorted in ascending order, return the unique
        sequence `levels` of non-negative integers of equal length such that
        levels[i] is the smallest non-negative integer that does not occur
        among all levels[j] for indices j < i with locs[i] - locs[j] < gap.

        Parameters
        ----------
        locs : Sequence[float]
            Sequence of inset locations sorted in ascending order
        gap : float

        Returns
        -------
        levels : Sequence[int]
        '''
        levels = []
        j = 0
        for i in range(len(locs)):
            while locs[i] - locs[j] > gap:
                j += 1

            for k in itertools.count():
                if k not in set(levels[j:i]):
                    levels.append(k)
                    break
        return levels

    def plot_mode_insets(
            self,
            inds=slice(None),
            offset=0,
            scaling=1e9,
            gap=0.020e+9,
            inset_size=0.25,
            inset_stagger=0.23,
            inset_pad=0.2,
            projection: Literal['polar', 'rectilinear'] = 'polar',
            fig=None,
            ax=None,
            **kwargs):

        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        elif (fig is None) or (ax is None):
            raise ValueError

        if scaling <= 0:
            raise ValueError
        if gap <= 0:
            raise ValueError

        plot_eigvals = self.eigvals[inds]
        plot_eigvecs = self.eigvecs[inds]
        inset_levels = self.inset_level_maker(plot_eigvals, gap)

        if projection == 'polar':
            axes_class = matplotlib.projections.polar.PolarAxes
        elif projection == 'rectilinear':
            axes_class = None
        else:
            raise ValueError

        for eigval, eigvec, inset_level in zip(plot_eigvals, plot_eigvecs, inset_levels):
            plot_freq = (eigval - offset) / scaling
            inset_ax = inset_axes(
                ax,
                inset_size,
                inset_size,
                loc='lower left',
                bbox_to_anchor=[plot_freq, inset_stagger * inset_level],
                bbox_transform=ax.get_xaxis_transform(),
                axes_class=axes_class,
                borderpad=inset_pad,
            )
            inset_ax.axis('off')
            self.basis.plot_field_intensity(eigvec, projection=projection, ax=inset_ax, **kwargs)
            # ax.indicate_inset_zoom(inset_ax)
