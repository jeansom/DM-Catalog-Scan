cdef extern from "math.h":
    long double sqrt(long double xx)
    long double atan2(long double a, double b)

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def appendSpherical(np.ndarray[DTYPE_t,ndim=2] xyz):
    cdef np.ndarray[DTYPE_t,ndim=2] pts = np.empty((xyz.shape[0],3))
    cdef long double XsqPlusYsq
    for i in xrange(xyz.shape[0]):
        XsqPlusYsq = xyz[i,0]**2 + xyz[i,1]**2
        pts[i,2] = atan2(xyz[i,1],xyz[i,0])
        pts[i,1] = atan2(sqrt(XsqPlusYsq),xyz[i,2])
        pts[i,0] = sqrt(XsqPlusYsq + xyz[i,2]**2)
    return pts
