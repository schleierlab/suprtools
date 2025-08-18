from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from suprtools.rf.couplers import CylindricalProbe, Probe


@pytest.fixture
def probe():
    return CylindricalProbe(radius=0.287e-3/2, tip_length=2.4e-3)


class TestProbe:
    '''
    Partitions for input loss constants:
        type (float, ufloat)

    Testing partitions for loss_db:

    freq:
        type (scalar, nestedlists, or ndarray)
        ndim (0, 1)
        shape along axis (0, 1, many)
        value (Hz, kHz, MHz, GHz ranges)
    '''
    def test_coupling_broadcasting_scalar(self, probe: Probe):
        retval = probe.resonator_coupling_rate([1, 1, 1], 90e+9)
        assert np.asarray(retval).shape == tuple()

    def test_coupling_broadcasting_ndim1(self, probe: Probe):
        retval = cast(NDArray, probe.resonator_coupling_rate([[1, 1, 1], [2, 2, 2]], 90e+9))
        assert retval.shape == (2,)
