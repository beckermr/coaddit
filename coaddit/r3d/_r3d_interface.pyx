# cython: language_level=3, infer_types=True
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.float64_t DTYPE_t

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# include the stuff we need from r3d
cdef extern from "r2d.h" nogil:
    ctypedef double r2d_real
    ctypedef int r2d_int

    ctypedef union r2d_rvec2:
        r2d_real xy[2]

    ctypedef union r2d_dvec2:
        r2d_int ij[2]

    ctypedef struct r2d_vertex:
        r2d_int pnbrs[2]
        r2d_rvec2 pos

    ctypedef struct r2d_poly:
        r2d_vertex verts[256]
        r2d_int nverts


cdef extern from "v2d.h" nogil:
    void r2d_rasterize(
        r2d_poly* poly, r2d_dvec2 ibox[2],
        r2d_real* dest_grid, r2d_rvec2 d, r2d_int polyorder)
    void r2d_get_ibox(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_rvec2 d)

cdef extern from "r2d.h" nogil:
    void r2d_init_poly(r2d_poly* poly, r2d_rvec2* vertices, r2d_int numverts)
    void r2d_print(r2d_poly* poly)
    r2d_int r2d_is_good(r2d_poly* poly)
    r2d_real r2d_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _rasterize_quads(double[:, :] verts_x, double[:, :] verts_y):
    # we always assume unit grid spacing with constant values in each input
    # polygon
    cdef r2d_int polyorder = 0
    cdef r2d_rvec2 d
    cdef r2d_rvec2 tmp_verts[4]
    cdef r2d_dvec2 *iboxes = NULL
    cdef r2d_poly *polys = NULL
    cdef r2d_real *grid = NULL
    cdef r2d_real *tmp_grid = NULL
    cdef int pind, vind, min_i, max_i, min_j, max_j, n_i, n_j
    cdef int max_n_i, max_n_j, i, j, tmp_n_j

    assert verts_x.shape[0] == verts_y.shape[0]
    assert verts_x.shape[1] == verts_y.shape[1]
    n_polys = verts_x.shape[0]
    assert n_polys >= 1

    with nogil:
        d.xy[0] = 1.0
        d.xy[1] = 1.0

        polys = <r2d_poly*>malloc(sizeof(r2d_poly) * n_polys)
        if not polys:
            free(polys)
            free(iboxes)
            free(grid)
            free(tmp_grid)
            with gil:
                raise MemoryError()

        iboxes = <r2d_dvec2*>malloc(sizeof(r2d_dvec2) * n_polys * 2)
        if not iboxes:
            free(polys)
            free(iboxes)
            free(grid)
            free(tmp_grid)
            with gil:
                raise MemoryError()

        # first we figure out the grid patches for each quad
        for pind in range(n_polys):
            tmp_verts[0].xy[0] = verts_x[pind, 0]
            tmp_verts[0].xy[1] = verts_y[pind, 0]
            tmp_verts[1].xy[0] = verts_x[pind, 1]
            tmp_verts[1].xy[1] = verts_y[pind, 1]
            tmp_verts[2].xy[0] = verts_x[pind, 2]
            tmp_verts[2].xy[1] = verts_y[pind, 2]
            tmp_verts[3].xy[0] = verts_x[pind, 3]
            tmp_verts[3].xy[1] = verts_y[pind, 3]
            r2d_init_poly(&(polys[pind]), tmp_verts, 4)

            # useful print statements for debugging
            # printf("%d\n", r2d_is_good(&(polys[pind])))
            printf(
                "  vol: %f\n",
                r2d_orient(tmp_verts[0], tmp_verts[1], tmp_verts[2]))
            r2d_print(&(polys[pind]))

            r2d_get_ibox(&(polys[pind]), &(iboxes[2*pind]), d)

        # then we decalare a full grid that is big enough to hold everything
        min_i = iboxes[0].ij[0]
        max_i = iboxes[0].ij[0]
        min_j = iboxes[0].ij[1]
        max_j = iboxes[0].ij[1]
        max_n_i = iboxes[1].ij[0] - iboxes[0].ij[0]
        max_n_j = iboxes[1].ij[1] - iboxes[0].ij[1]
        for pind in range(n_polys):
            for vind in range(2):
                min_i = min(iboxes[2*pind + vind].ij[0], min_i)
                max_i = max(iboxes[2*pind + vind].ij[0], max_i)
                min_j = min(iboxes[2*pind + vind].ij[1], min_j)
                max_j = max(iboxes[2*pind + vind].ij[1], max_j)

            max_n_i = max(
                iboxes[2*pind+1].ij[0] - iboxes[2*pind].ij[0], max_n_i)
            max_n_j = max(
                iboxes[2*pind+1].ij[1] - iboxes[2*pind].ij[1], max_n_j)

        # note that max_i/max_j is the location of the edge of the last
        # cell in the patch - thus the difference (max_i - min_i) is the
        # the actual number of cells in the grid
        n_i = max_i - min_i
        n_j = max_j - min_j
        grid = <r2d_real*>malloc(sizeof(r2d_real) * n_i * n_j)
        if not grid:
            free(polys)
            free(iboxes)
            free(grid)
            free(tmp_grid)
            with gil:
                raise MemoryError()
        for pind in range(n_i * n_j):
            grid[pind] = 0

        tmp_grid = <r2d_real*>malloc(sizeof(r2d_real) * max_n_i * max_n_j)
        if not tmp_grid:
            free(polys)
            free(iboxes)
            free(grid)
            free(tmp_grid)
            with gil:
                raise MemoryError()

        # then we loop for each quad, rasterize it, and assign the values
        # to the big grid
        for pind in range(n_polys):
            for i in range(max_n_i * max_n_j):
                tmp_grid[i] = 0
            r2d_rasterize(
                &(polys[pind]), &(iboxes[2*pind]), tmp_grid, d, polyorder)
            tmp_n_j = iboxes[2*pind+1].ij[1] - iboxes[2*pind].ij[1]
            for i in range(max_n_i):
                for j in range(max_n_j):
                    printf("  val: %d %d %f\n", i, j, tmp_grid[2*i+j])
            for i in range(iboxes[2*pind].ij[0], iboxes[2*pind+1].ij[0]):
                for j in range(iboxes[2*pind].ij[1], iboxes[2*pind+1].ij[1]):
                    grid[n_j*(i - min_i) + (j - min_j)] += tmp_grid[
                        tmp_n_j*(i - iboxes[2*pind].ij[0]) + (j - iboxes[2*pind].ij[1])]

        # free stuff on the way out
        free(polys)
        free(iboxes)
        free(tmp_grid)

    # return the new numpy array
    # see this stackoverflow for dealing with memory management
    # https://stackoverflow.com/questions/25102409/c-malloc-array-pointer-return-in-cython
    cdef void *ptr
    cdef np.npy_intp size[2]
    ptr = <void*>grid
    # fastest index goes on the right
    size[0] = n_i
    size[1] = n_j

    cdef np.ndarray[DTYPE_t, ndim=2] arr = \
        np.PyArray_SimpleNewFromData(2, size, np.NPY_FLOAT64, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    # we ned transpose here since the x (column) data is associated with
    # the slowest index (i) but we need it with the fastest index (j)
    return arr.T[::-1, ::-1], min_i, min_j
