import copy
import itertools
from collections.abc import Callable, Sequence
from typing import Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from numpy.typing import ArrayLike, NDArray
from scipy.constants import pi
from uncertainties import ufloat


class ModePlotStyler(Protocol):
    def __call__(self, mode_data: np.recarray, *xvals: np.number) -> dict: ...


class ModeParams:
    xs: tuple[NDArray[np.number], ...]

    def __init__(self, xs: Sequence[ArrayLike], mode_records):
        '''
        mode_records: array_like, shape (..., 4)
            Parameter order: Re(pole), Im(pole), Re(residue), Im(residue).
            Poles and residues should be specified in angular units (1/s).
        '''
        rec_shape = np.asarray(mode_records).shape

        if rec_shape[-1] != 4:
            raise ValueError
        if np.any(~np.equal([len(np.asarray(x)) for x in xs], rec_shape[:-1])):
            raise ValueError

        self.xs = tuple([np.array(x) for x in xs])
        self.params_arr = np.rec.fromrecords(
            mode_records,
            names=['pole_r', 'pole_i', 'res_r', 'res_i'],
            formats=None,  # workaround for https://github.com/numpy/numpy/issues/26376
        )

    def __getitem__(self, key):
        retval = copy.deepcopy(self)
        retval.x = self.x[key]
        retval.params_arr = self.params_arr[key]

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
            ax=None,
            remove_nans=False,
            kwarg_func: Optional[ModePlotStyler] = None,
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

        if axes_order is None:
            axes_order = list(range(len(self.xs)))
            axes_order.pop(axis)

        axis_x = self.xs[axis]

        ind_iterator = itertools.product(*[
            range(len(self.xs[i])) for i in axes_order
        ])

        for multi_ind in ind_iterator:
            # multi_ind: axis ordering per desired iteration order

            fullslice = slice(None, None, None)

            # ordered according to storage order
            this_iter_slice = tuple(
                fullslice if i == axis else multi_ind[axes_order.index(i)]
                for i in range(len(self.xs))
            )
            uvals = uvalues[this_iter_slice]
            full_mode_data = self.params_arr[this_iter_slice]

            mask = np.full_like(axis_x, True, dtype=np.bool_)
            if remove_nans:
                mask = ~np.isnan(unp.nominal_values(uvals))

            xcombo = [
                self.xs[axes_order[i]][ind]
                for i, ind in enumerate(multi_ind)
            ]
            kwargs = dict(label=xcombo)
            if kwarg_func is not None:
                kwargs |= kwarg_func(full_mode_data, *xcombo)

            ax.errorbar(
                axis_x[mask],
                unp.nominal_values(uvals[mask]),
                unp.std_devs(uvals[mask]),
                **kwargs,
            )


class FabryPerotModeParams(ModeParams):
    def __init__(self, x, mode_records, fsr):
        self.fsr = fsr
        super().__init__(x, mode_records)

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
