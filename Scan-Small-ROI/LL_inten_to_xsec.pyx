###############################################################################
# LL_inten_to_xsec.pyx
###############################################################################
#
# Convert from intensity LLs to xsec LLs rapidly, and profile over the J
# factor uncertainty
# 
###############################################################################

import numpy as np
from cython.parallel import parallel, prange
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double pow(double x, double y) nogil
    double sqrt(double x) nogil
    double log(double x) nogil

cdef double pi = np.pi
cdef double ln10 = np.log(10)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def construct_xsec_LL(double[::1] xsecs,double[::1] ebins,double [::1] PPnoxsec,double[:,::1] LLs, double [:,::1] intensity, double l10_J, double l10_Jerr):
    """ Given intensity LLs, construct xsec LLs by profiling over the J-factor uncertainty
    """

    cdef double[::1] LL2vals = np.zeros(len(xsecs),dtype=DTYPE)
    cdef double[::1] l10J_proflike = np.linspace(l10_J-3*l10_Jerr,l10_J+3*l10_Jerr,700,dtype=DTYPE)
    cdef double[::1] LL_proflike

    cdef Py_ssize_t xi, Ji, ei, i
    cdef int Nj = len(l10J_proflike)
    cdef int Ne = len(ebins) - 1 #Ne is the number of bin edges, so Ne-1 bins
    cdef int xj = len(xsecs)

    cdef double min_int = 0.0
    cdef double max_int = 0.0
    cdef int Nei = np.shape(LLs)[1]
    cdef double[::1] intensity_ei = np.zeros(Nei, dtype=DTYPE)
    cdef double[::1] LL_ei = np.zeros(Nei, dtype=DTYPE)

    cdef double intval = 0.0

    for xi in range(xj):
        # Need to calculate the likelihood for many J-values and find the one that maximises
        LL_proflike = np.zeros(len(l10J_proflike),dtype=DTYPE)

        for Ji in range(Nj):
            # Loop over J-factors
            for ei in range(Ne):
                # Loop over energy bins
                
                min_int = intensity[ei,0] 
                max_int = intensity[ei,Nei-1]                
                intval = PPnoxsec[ei]*xsecs[xi]*pow(10.,l10J_proflike[Ji])

                if intval <= min_int:
                    # Add minimum value
                    LL_proflike[Ji] += LLs[ei,0]
                elif intval >= max_int:
                    # Add maximum value
                    LL_proflike[Ji] += LLs[ei,Nei-1]
                else:
                    # Determine the value from interpolation
                    LL_proflike[Ji] += interp(intensity[ei], LLs[ei], intval, Nei) 
            
            # Add the J-factor weighting term as a log normal
            LL_proflike[Ji] += -pow(l10J_proflike[Ji]-l10_J,2.)/(2*pow(l10_Jerr,2.)) \
                            - log( sqrt(2*pi)*l10_Jerr*pow(10.,l10J_proflike[Ji])*ln10 )

        # Now find the maximum LL from this list, as that's the profile likelihood method
        LL2vals[xi] = 2*find_max(LL_proflike,Nj) # 2x because for TS

    LL2out = np.array(LL2vals)

    return LL2out


###########################################################
# C functions to do linear interpolation and find maximum #
###########################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double interp(double[::1] x, double[::1] y, double x0, int Nx) nogil:
    """ Manually perform linear interpolation - much faster than python
    """
    cdef Py_ssize_t i
    cdef double y0 = 0.0
    for i in range(Nx-1):
        if x0 >= x[i] and x0 < x[i+1]:
            y0 = y[i] + (x0-x[i])*(y[i+1]-y[i])/(x[i+1]-x[i])
            break
    return y0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double find_max(double[::1] x, int Nx) nogil:
    """ Manually find maximum
    """
    cdef Py_ssize_t i
    cdef double res = x[0]
    for i in range(1,Nx):
        if x[i] > res:
            res = x[i]
    return res
