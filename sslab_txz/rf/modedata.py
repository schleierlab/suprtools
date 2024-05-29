import copy
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from numpy.typing import NDArray
from scipy.constants import pi
from uncertainties import ufloat


class ModeParams:
    def __init__(self, x, mode_records):
        '''
        mode_records: array_like, shape (..., 4)
            Parameter order: Re(pole), Im(pole), Re(residue), Im(residue).
            Poles and residues should be specified in angular units (1/s).
        '''
        rec_shape = np.asarray(mode_records).shape

        if rec_shape[-1] != 4:
            raise ValueError
        if np.asarray(x).shape != rec_shape[:-1]:
            raise ValueError

        self.x = np.array(x)
        self.params_arr = np.rec.fromrecords(
            mode_records,
            names=['pole_r', 'pole_i', 'res_r', 'res_i'],
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

    def errorbar_plot(self, uvalues, ax=None, remove_nans=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        mask = np.full(self.x.shape, True)
        if remove_nans:
            mask = ~np.isnan(unp.nominal_values(uvalues))

        ax.errorbar(
            self.x[mask],
            unp.nominal_values(uvalues[mask]),
            unp.std_devs(uvalues[mask]),
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
