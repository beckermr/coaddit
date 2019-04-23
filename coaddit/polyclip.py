import numpy as np
from numba import njit


@njit
def is_inside(cp1, cp2, p):
    """
    Returns true of inside
    """
    return (
        (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
    )

@njit
def get_intersection(cp1, cp2, s, e):
    """
    get intersection
    """
    dc = ( cp1[0] - cp2[0], cp1[1] - cp2[1] )
    dp = ( s[0] - e[0], s[1] - e[1] )
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0] 
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return ( (n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3 )

@njit
def nclip_poly(subject, clip):#, output):
    """
    Parameters
    ----------
    subject: array
        The polygon to be clipped, shape (nvertices, 2)
    clip: array
        The clipping polygon, shape (nverticies, 2)

    Returns
    -------
    output: array
        The clipped polygon,  shape (nverticies, 2)
    """
    nsubject = subject.shape[0]
    outputList = np.zeros( (2*nsubject, 2) )
    outputList[:nsubject,:] = subject
    inputList = outputList.copy()


    noutput = nsubject
    cp1 = clip[-1]

    nclip = clip.shape[0]
    for i in range(nclip):
        cp2 = clip[i]

        inputList[:] = outputList[:]
        ninput = noutput
        noutput = 0

        s=inputList[ninput-1]

        for j in range(ninput):

            e = inputList[j]

            if is_inside(cp1, cp2, e):
                if not is_inside(cp1, cp2, s):
                    outputList[noutput] = get_intersection(cp1, cp2, s, e)
                    noutput += 1
                outputList[noutput] = e
                noutput += 1
            elif is_inside(cp1, cp2, s):
                outputList[noutput] = get_intersection(cp1, cp2, s, e)
                noutput += 1
            s = e
        cp1 = cp2

    outputList = outputList[:noutput,:]
    return outputList

def test(ntrial=1, pngfile=None, show=False):
    import time

    subject = [
        (50,150),
        (200,50),
        (350,150),
        (350,300),
        (250,300),
        (200,250),
        (150,350),
        (100,250),
        (100,200),
    ]

    clip=[
        (100,100),
        (300,100),
        (300,300),
        (100,300),
    ]


    sa=np.array(subject)
    ca=np.array(clip)

    # run once to compile
    clipped = nclip_poly(sa, ca)
    print('original size:',sa.shape[0],'clipped size:',clipped.shape[0])

    # more for timing
    if ntrial > 1:
        tm = time.time()
        for i in range(ntrial):
            clipped = nclip_poly(sa, ca)
        tm = time.time()-tm
        print('time for %d: %g  time per: %g' % (ntrial,tm, tm/ntrial))

    if show:
        import biggles
        plt=biggles.FramedPlot()

        sxy=np.array(subject + [subject[0]])
        sx = sxy[:,0]
        sy = sxy[:,1]
        plt.add(
            biggles.Curve(sx,sy,color='yellow'),
        )

        cxy=np.array(clip + [clip[0]])
        cx = cxy[:,0]
        cy = cxy[:,1]
        plt.add(
            biggles.Curve(cx,cy,color='green'),
        )

        clipped = nclip_poly(sa, ca)

        n=clipped.shape[0]
        cpxy = np.zeros( (n+1,2) )
        cpxy[0:n,:] = clipped
        cpxy[-1,:] = clipped[0,:]

        cpx = cpxy[:,0]
        cpy = cpxy[:,1]
        plt.add(
            biggles.Curve(cpx,cpy,color='red'),
        )

        plt.show()
        if pngfile is not None:
            print('writing plot:',pngfile)
            plt.write_img(800,800,pngfile)

if __name__=='__main__':
    test()

