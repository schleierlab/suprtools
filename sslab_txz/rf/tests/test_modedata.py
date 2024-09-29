import numpy as np
import pytest

from sslab_txz.rf.modedata import FabryPerotModeParams


@pytest.fixture
def modeparams_3d():
    xs = [
        [11, 13, 17, 19],
        [23, 25, 29],
        [31, 37],
    ]

    mode_records = np.arange(4 * 3 * 2 * 4).reshape(4, 3, 2, 4)
    return FabryPerotModeParams(xs, mode_records, fsr=3e+9)


def check_xs_equality(xs1, xs2):
    assert len(xs1) == len(xs2)
    for x1, x2 in zip(xs1, xs2):
        np.testing.assert_array_equal(x1, x2)


class TestModeParams:
    def test_getitem_ellipsis(self, modeparams_3d: FabryPerotModeParams):
        mp = modeparams_3d[...]
        check_xs_equality(mp.xs, ([11, 13, 17, 19], [23, 25, 29], [31, 37]))
        np.testing.assert_array_equal(mp.params_arr, modeparams_3d.params_arr)

    def test_getitem_int(self, modeparams_3d: FabryPerotModeParams):
        mp = modeparams_3d[1]

        check_xs_equality(mp.xs, ([23, 25, 29], [31, 37]))
        np.testing.assert_array_equal(mp.params_arr, modeparams_3d.params_arr[1])

    def test_getitem_slice(self, modeparams_3d: FabryPerotModeParams):
        mp = modeparams_3d[1:2]

        check_xs_equality(mp.xs, [[13], [23, 25, 29], [31, 37]])
        np.testing.assert_array_equal(mp.params_arr, modeparams_3d.params_arr[1:2])

    def test_getitem_tuple_ellipsis(self, modeparams_3d: FabryPerotModeParams):
        mp = modeparams_3d[3, ..., 0]
        check_xs_equality(mp.xs, [[23, 25, 29]])
        np.testing.assert_array_equal(mp.params_arr, modeparams_3d.params_arr[3, ..., 0])

    def test_getitem_tuple_slice(self, modeparams_3d: FabryPerotModeParams):
        mp = modeparams_3d[:2, 1::-1]
        check_xs_equality(mp.xs, [[11, 13], [25, 23], [31, 37]])
        np.testing.assert_array_equal(mp.params_arr, modeparams_3d.params_arr[:2, 1::-1])

    def test_getitem_single_array(self, modeparams_3d: FabryPerotModeParams):
        mp = modeparams_3d[[1, 2]]
        check_xs_equality(mp.xs, [[13, 17], [23, 25, 29], [31, 37]])
        np.testing.assert_array_equal(mp.params_arr, modeparams_3d.params_arr[[1, 2]])
