from typing import Mapping, Optional, Self

import h5py
import numpy as np
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
