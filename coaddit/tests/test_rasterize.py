import numpy as np
import pytest

from coaddit.rasterize import rasterize_poly


@pytest.mark.parametrize('off', [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_smoke(off):
    # convention here is that last dimension is x then y
    verts = np.zeros((4, 2)) + 0.5 + off
    verts[1, 0] = 1.5 + off
    verts[2, 0] = 1.5 + off
    verts[2, 1] = 1.5 + off
    verts[3, 1] = 1.5 + off
    verts[:, 0] += 2

    arr, start_inds = rasterize_poly(verts, 1)

    assert start_inds[0] == off + 2
    assert start_inds[1] == off
    assert np.all(arr == 0.25), arr


@pytest.mark.parametrize('off', [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_shift(off):
    # convention here is that last dimension is x then y
    verts = np.zeros((4, 2)) + off
    verts[0, 1] += 0.25
    verts[0, 0] += 0.3
    verts[1, 1] += 1.25
    verts[1, 0] += 0.3
    verts[2, 1] += 1.25
    verts[2, 0] += 1.3
    verts[3, 1] += 0.25
    verts[3, 0] += 1.3
    verts[:, 0] += 2

    area = np.array([
        [0.75*0.7, 0.25*0.7],
        [0.75*0.3, 0.25*0.3]])

    arr, start_inds = rasterize_poly(verts, 1)
    assert start_inds[0] == off + 2
    assert start_inds[1] == off
    assert arr.shape[0] == 2
    assert arr.shape[1] == 2
    assert np.allclose(arr, area)


# # @pytest.mark.parametrize('off', [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_shift_dims(off=0):
    # convention here is that last dimension is x then y
    verts = np.zeros((4, 2)) + off
    verts[0, 1] += 0.25
    verts[0, 0] += 0.3
    verts[1, 1] += 2.25
    verts[1, 0] += 0.3
    verts[2, 1] += 2.25
    verts[2, 0] += 1.3
    verts[3, 1] += 0.25
    verts[3, 0] += 1.3
    verts[:, 0] += 2

    area = np.array([
        [0.75*0.7, 1 * 0.7, 0.25*0.7],
        [0.75*0.3, 1 * 0.3, 0.25*0.3]])
    area /= np.sum(area)

    arr, start_inds = rasterize_poly(verts, 1)
    assert start_inds[0] == off + 2
    assert start_inds[1] == off
    assert arr.shape[0] == 2
    assert arr.shape[1] == 3
    assert np.allclose(arr, area)
