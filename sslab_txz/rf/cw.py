from typing import Mapping, Optional, Self, cast

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray


class CWMeasurement:
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

    @property
    def f_sample(self):
        return 1 / (self.t[1] - self.t[0])

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

            def getter(key):
                try:
                    return f.attrs[key]
                except KeyError:
                    return None

            frequency = f.attrs['frequency']
            temperatures = {
                key: getter(key)
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

    def plot_power_spectrum(
            self, ax: Optional[plt.Axes] = None, scale: float = 1, **kwargs):
        '''
        Parameters
        ----------
        scale: float
            Scale by which to multiply FFT. This is the square root of
            the scaling factor ultimately on the y-axis, since the power
            spectrum goes as FFT squared. The scale converts from S21
            units U to a desired unit V of choice (and should thus be
            expressed in units of V/U).

        '''
        if ax is None:
            _, ax = plt.subplots(layout='constrained')
            ax = cast(plt.Axes, ax)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power spectral density [U$^2$/Hz]')

        fft = np.fft.fft(self.s21, axis=-1, norm='ortho') / np.sqrt(self.f_sample)

        ax.semilogy(
            np.fft.fftshift(np.fft.fftfreq(len(self.t)) * self.f_sample),
            np.fft.fftshift(np.mean(np.abs(scale * fft)**2, axis=0)),
            **kwargs,
        )
