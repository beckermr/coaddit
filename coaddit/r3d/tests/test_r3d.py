import numpy as np
import pytest

from .._r3d_interface import _rasterize_quads


@pytest.mark.parametrize('off', [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_smoke(off):
    # convention here is that last dimension is x then y
    verts = np.zeros((1, 4, 2)) + 0.5 + off
    verts[0, 1, 0] = 1.5 + off
    verts[0, 2, 0] = 1.5 + off
    verts[0, 2, 1] = 1.5 + off
    verts[0, 3, 1] = 1.5 + off
    verts[:, :, 0] += 2

    arr, min_x, min_y = _rasterize_quads(verts[:, :, 0], verts[:, :, 1])

    assert min_x == off + 2
    assert min_y == off
    assert np.all(arr == 0.25), arr


# @pytest.mark.parametrize('off', [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_shift(off=0):
    # convention here is that last dimension is x then y
    verts = np.zeros((1, 4, 2)) + off
    verts[0, 0, 0] += 0.25
    verts[0, 0, 1] += 0.3
    verts[0, 1, 0] += 1.25
    verts[0, 1, 1] += 0.3
    verts[0, 2, 0] += 1.25
    verts[0, 2, 1] += 1.3
    verts[0, 3, 0] += 0.25
    verts[0, 3, 1] += 1.3
    verts[:, :, 0] += 2

    area = np.array([
        [0.75*0.7, 0.25*0.7],
        [0.75*0.3, 0.25*0.3]])
    print(np.sum(area))

    arr, min_x, min_y = _rasterize_quads(verts[:, :, 0], verts[:, :, 1])
    print(min_x, min_y, arr, arr.shape, np.sum(arr))
    assert min_x == off + 2
    assert min_y == off
    assert arr.shape[0] == 2
    assert arr.shape[1] == 2
    assert np.allclose(arr, area)


# @pytest.mark.parametrize('off', [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_shift_dims(off=0):
    # convention here is that last dimension is x then y
    verts = np.zeros((1, 4, 2)) + off
    verts[0, 0, 0] += 0.25
    verts[0, 0, 1] += 0.3
    verts[0, 1, 0] += 2.25
    verts[0, 1, 1] += 0.3
    verts[0, 2, 0] += 2.25
    verts[0, 2, 1] += 1.3
    verts[0, 3, 0] += 0.25
    verts[0, 3, 1] += 1.3
    verts[:, :, 0] += 2

    area = np.array([
        [0.75*0.7, 1 * 0.7, 0.25*0.7],
        [0.75*0.3, 1 * 0.3, 0.25*0.3]])
    print(np.sum(area))

    arr, min_x, min_y = _rasterize_quads(verts[:, :, 0], verts[:, :, 1])
    print(min_x, min_y, arr, arr.shape, np.sum(arr))
    assert min_x == off + 2
    assert min_y == off
    assert arr.shape[0] == 2
    assert arr.shape[1] == 3
    assert np.all(arr == area)
