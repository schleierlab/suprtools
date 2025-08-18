import numpy as np
import pytest
import skrf as rf

from suprtools.rf.insertion_loss import MeasuredLossyLine, RootFrequencyLossElement


@pytest.fixture
def root_f_elt():
    return RootFrequencyLossElement(-1)


@pytest.fixture
def measured_elt():
    '''
    Measured element with 20+/-1 dB insertion loss
    '''
    f = np.arange(70, 115, 5) * 1e+9
    s = np.full_like(f, 0.1)
    net = rf.Network(f=f, s=s)
    net_b = rf.Network(f=f, s=(s * 10**(1/20)))

    return MeasuredLossyLine(net, net_b)


class TestRootFrequencyLossElement:
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
    def test_scalar(self, root_f_elt: RootFrequencyLossElement):
        assert root_f_elt.loss_db(25) == -5

    def test_ndim_0(self, root_f_elt: RootFrequencyLossElement):
        assert root_f_elt.loss_db(np.array(36)) == -6

    def test_ndim_1_single(self, root_f_elt: RootFrequencyLossElement):
        assert root_f_elt.loss_db([4]) == -2

    def test_ndim_1(self, root_f_elt: RootFrequencyLossElement):
        np.testing.assert_array_equal(
            root_f_elt.loss_db([1e+4, 1e+6, 1e+8, 1e+10]),
            [-1e+2, -1e+3, -1e+4, -1e+5],
        )

    def test_empty_ndim_1(self, root_f_elt: RootFrequencyLossElement):
        np.testing.assert_array_equal(
            root_f_elt.loss_db([]),
            [],
        )

    def test_ndim_2(self, root_f_elt: RootFrequencyLossElement):
        np.testing.assert_array_equal(
            root_f_elt.loss_db([[144., 196.], [0.81, 0.64]]),
            [[-12., -14.], [-0.9, -0.8]],
        )

    def test_return_ndarray(self, root_f_elt: RootFrequencyLossElement):
        assert isinstance(root_f_elt.loss_db([1, 4]), np.ndarray)


class TestMeasuredLossyLine:
    def test_rms_error(self, measured_elt: MeasuredLossyLine):
        np.testing.assert_almost_equal(
            measured_elt.loss_db(100e+9).s,
            1,
        )


class TestCascadedElement:
    def test_addition(self, measured_elt: MeasuredLossyLine):
        root_f_elt_ghz = RootFrequencyLossElement(-1 / np.sqrt(100e+9))
        sum_elt = root_f_elt_ghz + measured_elt
        assert sum_elt.loss_db(100e+9).n == -(1 + 20)

    def test_multiple_addition(self, measured_elt: MeasuredLossyLine):
        root_f_elt_ghz = RootFrequencyLossElement(-1 / np.sqrt(100e+9))
        sum_elt = root_f_elt_ghz + measured_elt + root_f_elt_ghz
        assert sum_elt.loss_db(100e+9).n == -(2 + 20)
