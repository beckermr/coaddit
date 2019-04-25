import numpy as np
import pytest

from ..lanczos import lanczos_resample


def test_lanczos_resample_interp_grid():
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    for row in range(11):
        for col in range(25):
            val = lanczos_resample(
                im,
                np.array([row], dtype=np.float64),
                np.array([col], dtype=np.float64))
            assert np.allclose(val, im[row, col])


@pytest.mark.parametrize(
    'row, col', [
        # clearly bad
        (-10, 50),

        # one is ok
        (10, 50),
        (-10, 5),

        # just on the edge
        (-3.00001, 5),
        (13, 5),
        (10, -3.00001),
        (10, 27),

        # both on edge
        (13, 27),
        (-3.0001, -3.0001),
        (13, -3.0001),
        (-3.0001, 27)])
def test_lanczos_resample_out_of_bounds(row, col):
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    val = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64))
    assert np.isnan(val)


@pytest.mark.parametrize(
    'row, col', [
        # clearly good
        (10, 5),

        # just inside the edge
        (-3, 5),
        (12, 5),
        (10, -3),
        (10, 26),

        # both inside the edge
        (12, 26),
        (-3, -3),
        (12, -3),
        (-3, 26)])
def test_lanczos_resample_in_bounds(row, col):
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    val = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64))
    assert not np.isnan(val)
