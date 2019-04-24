import numpy as np
from numba import njit

from .polyclip import poly_area, clip_poly


@njit
def rasterize_poly(polygon, delta):
    """Rasterize the input polygon onto a grid of width delta.

    NOTE: The grid lines start at (0, 0) so that the first cell center
    is at (0.5, 0.5).

    Parameters
    ----------
    polygon : np.ndarray, shape (n_verticies, 2)
        The array of verticies listed consecutively. The second
        dimension is ordered (y/row, x/column).
    delta : float
        The grid spacing. Assumed to be the same in both dimensions.

    Returns
    -------
    grid : np.ndarray
        A 2d array with the overlap area of the polygon. Sums to unity.
    start_inds : np.ndarray, shape (2,)
        The starting indices of the grid. The order is (y/row, x/column).
    """
    start_inds = np.zeros(2, dtype=np.int64)
    start_inds[0] = np.floor(np.min(polygon[:, 0]/delta))
    start_inds[1] = np.floor(np.min(polygon[:, 1]/delta))

    end_inds = np.zeros(2, dtype=np.int64)
    end_inds[0] = np.ceil(np.max(polygon[:, 0]/delta))
    end_inds[1] = np.ceil(np.max(polygon[:, 1]/delta))

    n_i = end_inds[0] - start_inds[0] + 1
    n_j = end_inds[1] - start_inds[1] + 1

    orig_cell = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * delta
    cell = np.zeros_like(orig_cell)
    cell[:, 0] = orig_cell[:, 0] + start_inds[0] * delta

    area = poly_area(polygon)
    grid = np.zeros((n_i, n_j), dtype=polygon.dtype)
    for i in range(n_i):
        # reset column
        cell[:, 1] = orig_cell[:, 1] + start_inds[1] * delta
        for j in range(n_j):
            grid[i, j] = poly_area(clip_poly(polygon, cell))
            # increment the column
            cell[:, 1] += delta
        # increment the row
        cell[:, 0] += delta

    grid /= area
    return grid, start_inds
