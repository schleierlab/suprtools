from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Sequence
from numbers import Integral
from types import EllipsisType
from typing import ClassVar, Literal, Optional, Protocol, Self, SupportsIndex, cast

import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from matplotlib.axes import Axes
from numpy._typing import _ArrayLikeInt_co
from numpy.typing import ArrayLike, NDArray
from scipy.constants import pi
from suprtools.typing import ErrorbarKwargs
from uncertainties import ufloat


class ModePlotStyler(Protocol):
    def __call__(self, mode_data: NDArray, *xvals: np.number) -> ErrorbarKwargs: ...


RecIndex1D = _ArrayLikeInt_co | slice


class ModeParams:
    xs: tuple[NDArray[np.number], ...]
    params_arr: NDArray
    _FIELD_NAMES: ClassVar[tuple[str, ...]] = ('pole_r', 'pole_i', 'res_r', 'res_i')

    def __init__(self, xs: Sequence[ArrayLike], mode_records):
        '''
        mode_records: array_like, shape (..., 4)
            Parameter order: Re(pole), Im(pole), Re(residue), Im(residue).
            Poles and residues should be specified in angular units (1/s).
        '''
        if isinstance(mode_records, np.recarray):
            if mode_records.dtype.names != self._FIELD_NAMES:
                raise ValueError
            records_shape = mode_records.shape
            self.params_arr = mode_records
        else:
            rec_shape = np.asarray(mode_records).shape
            if rec_shape[-1] != 4:
                raise ValueError
            records_shape = rec_shape[:-1]
            self.params_arr = np.rec.fromrecords(
                mode_records,
                names=self._FIELD_NAMES,
                formats=None,  # workaround for https://github.com/numpy/numpy/issues/26376
            )

        xs_lens = [len(np.asarray(x)) for x in xs]
        if np.any(~np.equal(xs_lens, records_shape)):
            raise ValueError(f'Input record array shape {records_shape} != xs shape {xs_lens}')

        self.xs = tuple([np.array(x) for x in xs])
        for ndarr in self.xs:
            ndarr.flags.writeable = False

    @property
    def ndim(self):
        return len(self.xs)

    def __getitem__(
            self,
            indx: (
                EllipsisType | slice | SupportsIndex  # basic indexing
                | Sequence | np.ndarray  # advanced indexing
                | tuple[SupportsIndex | slice | EllipsisType | Sequence, ...]
            )) -> Self:
        '''
        Return another ModeParams with records given by indexing the
        current records with the given index. The index may be:
            - any allowable index for numpy basic indexing without np.newaxis:
                - integer (or any SupportsIndex)
                - ellipsis
                - slice
                - any tuple of the above (but with no more than two ...)
                    up to the tensor rank (ndim)
            -

        '''
        indx_tuple: tuple[SupportsIndex | slice, ...]
        adv_indexing = False
        if isinstance(indx, tuple):
            sequence_elements = [isinstance(elt, Sequence | np.ndarray) for elt in indx]
            if np.sum(sequence_elements) > 1:
                raise ValueError('>1D advanced indexing not supported')
            # if np.sum(sequence_elements) == 1 and not sequence_elements[0]:
            #     raise ValueError('Advanced indexing only supported for first dimension')

            # can't do np.newaxis in indx because this doesn't work with NDArrays
            # can't use equality (==) because NDArrays override that too
            # we rely on np.newaxis being None and None being a singleton here
            if any(indx_elt is np.newaxis for indx_elt in indx):
                raise ValueError('Cannot insert newaxis into ModeParams')

            # can't np.asarray(indx) or indx_elt == Ellipsis;
            # these fail when there are NDArrays in indx
            ellipsis_occurrences = sum(indx_elt is Ellipsis for indx_elt in indx)

            if ellipsis_occurrences >= 2:
                raise ValueError('Cannot use ... twice')
            elif ellipsis_occurrences == 1:
                ellipsis_position = indx.index(...)

                indx_tuple = indx[:ellipsis_position] \
                    + (self.ndim - len(indx) + 1) * (slice(None),) \
                    + indx[ellipsis_position+1:]
            elif ellipsis_occurrences == 0:
                indx_tuple = cast(tuple[SupportsIndex | slice, ...], indx) \
                    + (slice(None),) * (self.ndim - len(indx))
        else:
            if isinstance(indx, Sequence | np.ndarray):
                adv_indexing = True
            indx_tuple = (indx,) + (slice(None),) * (self.ndim - 1)

        # the copy-if-adv-indexing logic is not really needed
        # because we make self.xs immutable
        # but let's keep this in case we find it useful to make self.xs mutable
        new_xs = tuple(
            (
                copy.copy(x[thisdim_ind])
                if adv_indexing and not isinstance(thisdim_ind, Sequence | np.ndarray)
                else x[thisdim_ind]
            )
            for thisdim_ind, x in zip(indx_tuple, self.xs)
            if (
                isinstance(thisdim_ind, Sequence | np.ndarray)
                or not isinstance(thisdim_ind, SupportsIndex)
            )  # drops dimensions where we take a single index
        )

        retval = copy.copy(self)
        retval.xs = new_xs
        retval.params_arr = self.params_arr[indx]
        return retval

    def transpose(self, axes: Optional[Sequence[Integral]] = None):
        axes_seq: Sequence[Integral]

        if axes is None:
            axes_seq = list(np.arange(self.ndim)[::-1])
        elif np.any(sorted(axes) != np.arange(self.ndim)):
            raise ValueError
        else:
            axes_seq = axes

        retval = copy.copy(self)
        retval.xs = tuple(self.xs[ind] for ind in axes_seq)
        retval.params_arr = np.transpose(self.params_arr, axes)

    @property
    def fwhms(self):
        '''
        Linewidths: in frequency units (Hz, not rad/s)
        '''
        return -2 * self.params_arr['pole_r'] / (2 * pi)

    @property
    def freqs(self):
        return self.params_arr['pole_i'] / (2 * pi)

    @property
    def res_mag(self):
        '''
        Give magnitude of residue in units of cyclic frequency (not angular).
        '''
        return unp.sqrt(self.params_arr['res_r']**2 + self.params_arr['res_i']**2) / (2 * pi)

    @property
    def q_factors(self):
        return self.freqs / self.fwhms

    def errorbar_plot(
            self,
            uvalues,
            axis=-1,
            axes_order: Optional[Sequence[int]] = None,
            ax: Optional[Axes] = None,
            remove_nans=False,
            error_style: Literal['bars', 'fill'] = 'bars',
            kwarg_func: Optional[ModePlotStyler] = None,
            fill_kw_func: Optional[ModePlotStyler] = None,
    ):
        '''
        axis: int
            Axis to plot by.
        axes_order: Sequence[int]
            Order of remaining axes to iterate over.
            Must not include plot axis.

            axes_order[0] is the first axis to iterate over
            axes_order.index(0) is the position of the first storage axis in the iteration
        kwarg_func: callable, optional
            Function that produces plot kwargs given xcombos
        '''
        if ax is None:
            fig, ax = plt.subplots()
            ax = cast(Axes, ax)

        positive_axis = axis % len(self.xs)
        if axes_order is None:
            axes_order = list(range(len(self.xs)))
            axes_order.pop(positive_axis)

        axis_x = self.xs[axis]

        ind_iterator = itertools.product(*[
            range(len(self.xs[i])) for i in axes_order
        ])

        for multi_ind in ind_iterator:
            # multi_ind: axis ordering per desired iteration order

            fullslice = slice(None, None, None)

            # ordered according to storage order
            # type annotation necessary to keep mypy happy
            this_iter_slice: tuple[slice | int, ...] = tuple(
                fullslice if i == positive_axis else multi_ind[axes_order.index(i)]
                for i in range(len(self.xs))
            )
            uvals = uvalues[this_iter_slice]
            full_mode_data = self.params_arr.__getitem__(this_iter_slice)

            mask = np.full_like(axis_x, True, dtype=np.bool_)
            if remove_nans:
                mask = ~np.isnan(unp.nominal_values(uvals))

            xcombo = [
                self.xs[axes_order[i]][ind]
                for i, ind in enumerate(multi_ind)
            ]
            kwargs: ErrorbarKwargs = dict(label=xcombo)
            if kwarg_func is not None:
                kwargs |= kwarg_func(full_mode_data, *xcombo)

            if error_style == 'bars':
                ax.errorbar(
                    axis_x[mask],
                    unp.nominal_values(uvals[mask]),
                    unp.std_devs(uvals[mask]),
                    **kwargs,
                )
                continue

            ebar_container = ax.errorbar(
                axis_x[mask],
                unp.nominal_values(uvals[mask]),
                **kwargs,
            )
            dataline = ebar_container.lines[0]

            fill_kw: ErrorbarKwargs = dict(
                color=dataline.get_color(),
                alpha=0.15,
                linewidth=0,
            )
            if fill_kw_func is not None:
                fill_kw |= fill_kw_func(full_mode_data, *xcombo)

            ax.fill_between(
                axis_x[mask],
                unp.nominal_values(uvals[mask]) - unp.std_devs(uvals[mask]),
                unp.nominal_values(uvals[mask]) + unp.std_devs(uvals[mask]),
                **fill_kw,
            )


class FabryPerotModeParams(ModeParams):
    def __init__(self, xs: Sequence[ArrayLike], mode_records, fsr):
        self.fsr = fsr
        super().__init__(xs, mode_records)

    @property
    def finesses(self):
        return self.fsr / self.fwhms


class ModeMaskingPolicy:
    def __init__(self, checker: Callable[[ModeParams], NDArray[np.bool_]]):
        self.checker = checker

    @classmethod
    def from_limits(cls, fwhm_max=np.inf, freq_s_max=np.inf):
        def checker(mode_info: ModeParams):
            return (
                (unp.std_devs(mode_info.freqs) > freq_s_max)
                | (unp.nominal_values(mode_info.fwhms) > fwhm_max)
            )

        return cls(checker)

    def apply(self, mode_info: ModeParams) -> ModeParams:
        ret_mode_info = copy.deepcopy(mode_info)
        ret_mode_info.params_arr = np.ma.masked_where(
            self.checker(mode_info),
            mode_info.params_arr,
        ).filled(
            fill_value=ufloat(np.nan, np.nan),
        )

        return ret_mode_info
