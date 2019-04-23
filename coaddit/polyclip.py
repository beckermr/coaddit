import numpy as np
from numba import njit


@njit
def _ccw(a, b, c):
    """Test if the points a, b, c are in counterclockwise order."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


@njit
def _intersect(a, b, c, d):
    """Test if line segment (a, b) intersects line segement (c, d).

    See this blog post:

        https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    for a description of the algorithm.
    """
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


@njit
def is_simple_poly(polygon):
    """Determine if a set of points forms a simple polygon (i.e. one that
    does not intersect itself).

    NOTE: This code uses an O(n^2) algorithm.

    Parameters
    ----------
    polygon : np.ndarray, shape (n_verticies, 2)
        The array of verticies listed consecutively.

    Returns
    -------
    is_simple : bool
        True if the polygon is simple, False otherwise.
    """
    n_poly = polygon.shape[0]
    for i in range(n_poly):
        ip1 = (i + 1) % n_poly
        a = polygon[i]
        b = polygon[ip1]

        for j in range(i+1, n_poly):
            jp1 = (j + 1) % n_poly
            c = polygon[j]
            d = polygon[jp1]

            if _intersect(a, b, c, d):
                return False
    return True


@njit
def poly_area(polygon):
    """Compute the area of the input polygon.

    Parameters
    ----------
    polygon : np.ndarray, shape (n_verticies, 2)
        The array of verticies listed consecutively.

    Returns
    -------
    area : float
        The area of the polygon.
    """
    n = polygon.shape[0]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    area = np.abs(area) * 0.5
    return area


@njit
def _is_inside(cp1, cp2, p):
    """Returns true if inside"""
    return (
        (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
    )


@njit
def _get_intersection(cp1, cp2, s, e):
    """get intersection"""
    dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
    dp = (s[0] - e[0], s[1] - e[1])
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return ((n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3)


@njit
def clip_poly(subject, clip):
    """Clip a possibly non-convex polygon (the subject) against a convex
    polygon (the clip).

    NOTE: This code implements the Sutherland–Hodgman algorithm.

    Parameters
    ----------
    subject : np.ndarray, shape (nvertices, 2)
        The polygon to be clipped.
    clip : np.ndarray, shape (nverticies, 2)
        The clipping polygon. Must be convex.

    Returns
    -------
    output : np.ndarray, shape (nverticies, 2)
        The clipped polygon.
    """
    nsubject = subject.shape[0]
    outputList = np.zeros((2*nsubject, 2))
    outputList[:nsubject, :] = subject
    inputList = outputList.copy()

    noutput = nsubject
    cp1 = clip[-1]

    nclip = clip.shape[0]
    for i in range(nclip):
        cp2 = clip[i]

        inputList[:] = outputList[:]
        ninput = noutput
        noutput = 0

        s = inputList[ninput-1]

        for j in range(ninput):

            e = inputList[j]

            if _is_inside(cp1, cp2, e):
                if not _is_inside(cp1, cp2, s):
                    outputList[noutput] = _get_intersection(cp1, cp2, s, e)
                    noutput += 1
                outputList[noutput] = e
                noutput += 1
            elif _is_inside(cp1, cp2, s):
                outputList[noutput] = _get_intersection(cp1, cp2, s, e)
                noutput += 1
            s = e
        cp1 = cp2

    outputList = outputList[:noutput, :]
    return outputList
