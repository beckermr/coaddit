import numpy as np
from scipy.spatial import ConvexHull

from coaddit.polyclip import poly_area, clip_poly, is_simple_poly


def test_poly_area_smoke():
    poly = np.zeros((4, 2))
    poly[1, 0] = 0.5
    poly[2, :] = 0.5
    poly[3, 1] = 0.5
    assert poly_area(poly) == 0.25


def _area_heron(a, b, c):
    s = (a + b + c)/2
    return np.sqrt(s * (s - a) * (s - b) * (s-c))


def test_poly_area_triangle():
    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        poly = 10*(rng.uniform(size=(3, 2)) - 0.5)

        a = np.sqrt(
            (poly[0, 0] - poly[1, 0])**2 + (poly[0, 1] - poly[1, 1])**2)
        b = np.sqrt(
            (poly[0, 0] - poly[2, 0])**2 + (poly[0, 1] - poly[2, 1])**2)
        c = np.sqrt(
            (poly[1, 0] - poly[2, 0])**2 + (poly[1, 1] - poly[2, 1])**2)

        assert np.allclose(poly_area(poly), _area_heron(a, b, c))


def test_poly_clip_smoke():
    rng = np.random.RandomState(seed=10)
    subj = rng.uniform(size=(3, 2))
    clip = np.array([[0, 0], [1, 0], [1, 1]])
    clipped = clip_poly(subj, clip)
    assert poly_area(clipped) <= poly_area(subj)


def test_poly_clip_rects():
    subj = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    clip = np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]])
    clipped = clip_poly(subj, clip)
    assert poly_area(clipped) <= poly_area(subj)
    assert np.allclose(poly_area(clipped), 0.5)
    assert clipped.shape[0] == 4

    clipped = set(tuple(c) for c in clipped.tolist())
    assert (0.5, 0) in clipped
    assert (0.5, 1) in clipped
    assert (1, 1) in clipped
    assert (1, 0) in clipped


def test_poly_clip_random():
    clip = np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]])
    rng = np.random.RandomState(seed=10)
    for _ in range(100):
        dim = rng.randint(3, 7)
        pts = rng.uniform(size=(dim, 2)) * 3 - 1.5

        if not is_simple_poly(pts):
            # use convex hull to fix self-intersecting polygons
            hull = ConvexHull(pts)
            loc = 0
            subj = []
            for _ in range(hull.simplices.shape[0]):
                subj.append(pts[hull.simplices[loc, 1], :])
                loc = hull.neighbors[loc, 0]
            subj = np.array(subj)
        else:
            subj = pts

        clipped = clip_poly(subj, clip)
        assert poly_area(clipped) <= poly_area(subj)


def old_ttest(ntrial=1, pngfile=None, show=False):
    import time

    subject = [
        (50, 150),
        (200, 50),
        (350, 150),
        (350, 300),
        (250, 300),
        (200, 250),
        (150, 350),
        (100, 250),
        (100, 200),
    ]

    clip = [
        (100, 100),
        (300, 100),
        (300, 300),
        (100, 300),
    ]

    sa = np.array(subject)
    ca = np.array(clip)

    # run once to compile
    clipped = clip_poly(sa, ca)
    print('original size:', sa.shape[0], 'clipped size:', clipped.shape[0])

    sarea = poly_area(sa)
    carea = poly_area(ca)
    clipped_area = poly_area(clipped)
    print('clip poly area:', carea)
    print('original area:', sarea, 'clipped area:', clipped_area)

    # more for timing
    if ntrial > 1:
        tm = time.time()
        for i in range(ntrial):
            clipped = clip_poly(sa, ca)
            clipped_area = poly_area(clipped)

        tm = time.time()-tm
        print('time for %d: %g  time per: %g' % (ntrial, tm, tm/ntrial))

    if show:
        import biggles
        plt = biggles.FramedPlot()

        sxy = np.array(subject + [subject[0]])
        sx = sxy[:, 0]
        sy = sxy[:, 1]
        plt.add(
            biggles.Curve(sx, sy, color='yellow'),
        )

        cxy = np.array(clip + [clip[0]])
        cx = cxy[:, 0]
        cy = cxy[:, 1]
        plt.add(
            biggles.Curve(cx, cy, color='green'),
        )

        clipped = clip_poly(sa, ca)

        n = clipped.shape[0]
        cpxy = np.zeros((n+1, 2))
        cpxy[0:n, :] = clipped
        cpxy[-1, :] = clipped[0, :]

        cpx = cpxy[:, 0]
        cpy = cpxy[:, 1]
        plt.add(
            biggles.Curve(cpx, cpy, color='red'),
        )

        plt.show()
        if pngfile is not None:
            print('writing plot:', pngfile)
            plt.write_img(800, 800, pngfile)


if __name__ == '__main__':
    old_ttest(ntrial=1000)
