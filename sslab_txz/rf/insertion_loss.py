from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage
import skrf as rf
import uncertainties
from numpy.typing import ArrayLike
from uncertainties import unumpy as unp


class LossElement(ABC):
    @abstractmethod
    def loss_db(self, freq: ArrayLike):
        raise NotImplementedError

    def __add__(self, other: Any) -> CascadedLossElement:
        if not isinstance(other, LossElement):
            return NotImplemented
        return CascadedLossElement(self, other)

    def __radd__(self, other: Any) -> CascadedLossElement:
        if not isinstance(other, LossElement):
            return NotImplemented
        return CascadedLossElement(other, self)


@dataclass
class CascadedLossElement(LossElement):
    elt_a: LossElement
    elt_b: LossElement

    def loss_db(self, freq: ArrayLike):
        return self.elt_a.loss_db(freq) + self.elt_b.loss_db(freq)


class MeasuredLossyLine(LossElement):
    measurement: rf.Network
    uncertainty_network: rf.Network

    def __init__(
            self,
            measurement: rf.Network,
            measurement_b: rf.Network,
            uncertainty_bandwidth: float = 20e+9,
    ):
        '''
        measurement: Network
            Single-port network with S21 measurement of loss element.
        measurement_b: Network
            Second measurement of same loss element at slightly
            different conditions. Used to estimate uncertainty
            (as the rms value of `measurement_b / measurement`, smoothed
            over `uncertainty_bandwidth`)
        uncertainty_bandwidth: float
            Bandwidth over which to
        '''
        self.measurement = measurement
        uncertainty_estimator = measurement_b / measurement

        df = measurement.frequency.df[0]
        filter_points = int(uncertainty_bandwidth // df)

        # running rms of [network ratio in dB]
        uncertainty_estimate_db = np.sqrt(scipy.ndimage.uniform_filter1d(
            rf.complex_2_db(uncertainty_estimator.s[:, 0, 0])**2,
            size=filter_points,
        ))

        self.uncertainty_network = rf.Network(
            f=self.measurement.f,
            s=rf.db_2_mag(uncertainty_estimate_db),
        )

    # TODO support arbitrary ndims here instead of flattening in loss_db
    @staticmethod
    def _interpolate_network(network: rf.Network, freq: ArrayLike):
        '''
        freq: array_like, 0- or 1-dimensional
        '''
        freq = np.asarray(freq)
        if freq.ndim >= 2:
            raise ValueError

        # promote `freq` to 1D, and pass sorted version to interpolator
        freq_atleast_1d = np.atleast_1d(freq)
        freq_argsort = np.argsort(freq_atleast_1d)
        interpolated_net = network.interpolate(freq_atleast_1d[freq_argsort], coords='polar')

        # extract interpolated values and unsort to get original freq list
        freq_inverse_argsort = np.argsort(freq_argsort)
        return interpolated_net.s[:, 0, 0][freq_inverse_argsort].reshape(freq.shape)

    def uncertainty_db(self, freq):
        return rf.complex_2_db(
            self._interpolate_network(self.uncertainty_network, freq),
        )

    def loss_db(self, freq: ArrayLike):
        freq_arr = np.asarray(freq)

        freq_flat = freq_arr.flatten()
        nominal_db_flat = rf.complex_2_db(self._interpolate_network(self.measurement, freq_flat))
        uncert_db_flat = self.uncertainty_db(freq_flat)

        uvars_flat = np.array(uncertainties.correlated_values(
            nominal_db_flat,
            np.diag(uncert_db_flat ** 2),
        ))

        if freq_arr.size == 1 and not isinstance(freq, np.ndarray):
            return uvars_flat.item()

        return uvars_flat.reshape(freq_arr.shape)


@dataclass
class RootFrequencyLossElement(LossElement):
    '''
    loss_constant: float
        Loss constant in units of dB / sqrt(Hz) such that insertion loss
        becomes loss_constant * sqrt(frequency)
    '''
    loss_constant: float

    def loss_db(self, freq: ArrayLike):
        return self.loss_constant * unp.sqrt(freq)


# 1 mm f/f ("bullet")
# 6/17 dB @ 50 GHz, 9/17 dB @ 110 GHz
