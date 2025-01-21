from typing import Any, Literal, Mapping, Optional, Self, cast

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray


class CWMeasurement:
    s21: NDArray[np.complex128]
    t: NDArray[np.float64]
    frequency: float
    metadata: dict[str, Any]

    def __init__(
            self,
            t: ArrayLike,
            s21: ArrayLike,
            frequency: float,
            temperatures: Optional[Mapping[str, float]],
            stage_positions: Optional[Mapping[str, float]],
            metadata: Mapping[str, Any] = dict(),
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
        self.metadata = dict(metadata)

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
                    'temperature_plate',
                    'temperature_sample',
                    'temperature_still_init',
                    'temperature_plate_init',
                    'temperature_sample_init',
                ]
            }

            metadata = dict(f.attrs)

            stage_positions = {
                'aaj': f.attrs['aaj_position'],
                'aak': f.attrs['aak_position'],
            }

        return cls(t, s21, frequency, temperatures, stage_positions, metadata)

    def plot_power_spectrum(
            self, ax: Optional[Axes] = None, scale: float = 1, **kwargs):
        return self.plot_spectrum(units='power', ax=ax, scale=scale, **kwargs)

    def plot_spectrum(
            self,
            units: Literal['amplitude', 'power'],
            ax: Optional[Axes] = None,
            scale: float = 1,
            **kwargs):
        '''
        Parameters
        ----------
        scale: float
            Scale by which to multiply FFT. The scale converts from S21
            units U to a desired unit V of choice (and should thus be
            expressed in units of V/U). If plotting power, this is the
            square root of the scaling factor ultimately on the y-axis,
            since the power spectrum goes as FFT squared.

        '''
        if ax is None:
            _, ax = plt.subplots(layout='constrained')
            ax = cast(Axes, ax)
            ax.set_xlabel('Frequency (Hz)')

            if units == 'amplitude':
                ax.set_ylabel(R'Amplitude spectral density (U/$\sqrt{\mathrm{Hz}}$)')
            elif units == 'power':
                ax.set_ylabel('Power spectral density (U$^2$/Hz)')

        fft = np.fft.fft(self.s21, axis=-1, norm='ortho') / np.sqrt(self.f_sample)
        mean_pow_spec = np.mean(np.abs(scale * fft)**2, axis=0)

        fftfreqs = np.fft.fftfreq(len(self.t), d=1/self.f_sample)

        if units == 'amplitude':
            exponent = 0.5
        elif units == 'power':
            exponent = 1

        ax.semilogy(
            np.fft.fftshift(fftfreqs),
            np.fft.fftshift(mean_pow_spec ** exponent),
            **kwargs,
        )

    def plot_integrated_spectrum(
            self,
            units: Literal['amplitude', 'power'],
            ax: Optional[Axes] = None,
            scale: float = 1,
            zero_start: bool = True,
            **kwargs):
        '''
        Parameters
        ----------
        scale: float
            Scale by which to multiply FFT. The scale converts from S21
            units U to a desired unit V of choice (and should thus be
            expressed in units of V/U).
        '''
        if ax is None:
            _, ax = plt.subplots(layout='constrained')
            ax = cast(Axes, ax)
            ax.set_xlabel('Frequency (Hz)')

            if units == 'amplitude':
                ax.set_ylabel('Integrated rms amplitude (U)')
            elif units == 'power':
                ax.set_ylabel('Integrated power (U$^2$)')

        fft = np.fft.fft(self.s21, axis=-1, norm='ortho') / np.sqrt(self.f_sample)
        mean_pow_spec = np.mean(np.abs(scale * fft)**2, axis=0)

        fft_n = len(self.t)
        if fft_n % 2 == 0:
            # when n even, +/- frequency range have unequal size (off by 1)
            # pad the (smaller) positive side with 0 power
            # for the extra freq in the - part
            padded_positive_part = np.pad(
                mean_pow_spec[1:fft_n//2],
                (0, 1),
                mode='constant',
                constant_values=0,
            )
            unsigned_pow_spec = padded_positive_part + mean_pow_spec[::-1][:fft_n//2]
        else:
            unsigned_pow_spec = mean_pow_spec[1:(fft_n+1)//2] + mean_pow_spec[::-1][:(fft_n+1)//2-1]

        # reverse the negative part, since for even n the extra freq is negative
        fftfreq = np.fft.fftfreq(len(self.t), d=1/self.f_sample)
        positive_freqs = -fftfreq[::-1][:fft_n//2]
        integrated_pow_spec = np.cumsum(unsigned_pow_spec) * fftfreq[1]

        def prepend_zero(arr: ArrayLike):
            return np.pad(arr, (1, 0), mode='constant', constant_values=0)

        postprocess_func = prepend_zero if zero_start else lambda x: x

        if units == 'amplitude':
            exponent = 0.5
        elif units == 'power':
            exponent = 1

        ax.plot(
            postprocess_func(positive_freqs),
            postprocess_func(integrated_pow_spec ** exponent),
            **kwargs,
        )
