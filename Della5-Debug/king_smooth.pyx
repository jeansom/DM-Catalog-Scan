###############################################################################
# king_smooth.pyx
###############################################################################
# HISTORY:
#   2017-03-22 - Written - Nick Rodd (MIT)
#   2017-03-24 - Sped up - Ben Safdi (MIT)
###############################################################################

# Accurately smooth a map with a King function 
# To use, initialize with the parameters that define the king function,
# and then call smooth_the_map which returns the smoothed map

import numpy as np
import healpy as hp
import king_params as kp
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange

# Define basic types
DTYPE = np.float
ctypedef np.float_t DTYPE_t

# Define basic math functions
cdef extern from "math.h":
    double fabs(double x) nogil
    double pow(double x, double y) nogil
    double sin(double x) nogil
    double fmin(double x, double y) nogil

cdef double pi = np.pi

# Define cython functions
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double king_fn_base(double x, double sigma, double gamma) nogil:
    """ A basic king function
    """

    return (1/(2*pi*pow(sigma,2))) * \
            (1-1/gamma)*pow((1+pow(x,2)/(2*gamma*pow(sigma,2))),-gamma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double king_fn_full(double r, double fcore, double score, double gcore, 
                         double stail, double gtail, double SpE) nogil:
    """ The combination of two king functions relevant for the Fermi PSF
        Note we multiply by r as the radial distribution is r*PSF(r)
    """

    return r*(fcore*king_fn_base(r/SpE,score,gcore) +
            (1-fcore)*king_fn_base(r/SpE,stail,gtail))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double dist_phi(double phi1, double phi2) nogil:
    """ Determine the shortest distance in phi accounting for the periodicity
    """

    return fmin(fabs(phi1-phi2), 2*pi-fabs((phi1-phi2)))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double dist_sphere(double theta1, double theta2, double phi1, double phi2) nogil:
    """ Calculate the angular distance on a sphere, assuming that the two
        points are close
    """

    return pow( pow(theta1 - theta2,2) +
            pow(dist_phi(phi1,phi2)*(sin((theta1+theta2)/2.)),2) , 0.5)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def smooth_the_map_et3(double[::1] the_map,
                       double[::1] theta_array, double[::1] phi_array,
                       double fcore1, double score1, double gcore1,
                       double stail1, double gtail1, double SpE1,
                       int npix,int threads):
    """ Smooth a map using eventtype 3 (top quartile)
    """

    # NB: must do each king function separately as they are not
    # correctly normalized out of the box, need to do it manually
    cdef double[::1] smoothed_map = np.zeros(npix,dtype=DTYPE)
    cdef double[::1] king_fn1 = np.empty(npix,dtype=DTYPE)
    cdef double theta_center, phi_center, theta_1, phi_1, dist
    cdef Py_ssize_t i, j, pix

    cdef double tmp1,tot1
    cdef double k01, rloc, rtot, kfval
    cdef double dnpix = float(npix)
    cdef double hpixsize = pow(2*pi/dnpix,0.5)

    with nogil:
        k01 = 0.0
        rloc = 0.0
        rtot = 0.0
        for j in range(100):
            tmp1 = king_fn_full(rloc,fcore1,score1,gcore1,
                               stail1,gtail1,SpE1)
            k01 += rloc*tmp1
            rtot += rloc
            rloc += hpixsize/100.

        k01 /= 100.0*rtot
        for pix in prange(npix,num_threads=threads):

            theta_center = theta_array[pix]
            phi_center = phi_array[pix]

            tot1 = 0.0
            for i in range(npix):
                theta_1 = theta_array[i]
                phi_1 = phi_array[i]
                dist = dist_sphere(theta_center,theta_1,phi_center,phi_1)
                if dist == 0.0:
                    king_fn1[i] = k01

                    tot1 += k01
                else:
                    if dist <= 5*SpE1*((score1+stail1)/2.):
                        tmp1 = king_fn_full(dist, fcore1,score1, gcore1, stail1, gtail1, SpE1)
                    else:
                        tmp1 = 0.0
                    king_fn1[i] = tmp1
                    tot1 += tmp1

            for i in range(npix):
                kfval = 0.0
                kfval += king_fn1[i]/tot1
                smoothed_map[i] += kfval * the_map[pix]

    return smoothed_map

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def smooth_the_map_et0(double[::1] the_map,
                       double[::1] theta_array, double[::1] phi_array,
                       double fcore1, double score1, double gcore1,
                       double stail1, double gtail1, double SpE1,
                       double fcore2, double score2, double gcore2,
                       double stail2, double gtail2, double SpE2,
                       double fcore3, double score3, double gcore3,
                       double stail3, double gtail3, double SpE3,
                       double fcore4, double score4, double gcore4,
                       double stail4, double gtail4, double SpE4,
                       int npix,int threads):
    """ Smooth a map using eventtype 0 (all 4 quartiles)
    """

    print("Just entered smooth_the_map_et0")

    # NB: must do each king function separately as they are not
    # correctly normalized out of the box, need to do it manually
    cdef double[::1] smoothed_map = np.zeros(npix,dtype=DTYPE)
    cdef double[::1] king_fn1 = np.empty(npix,dtype=DTYPE)
    cdef double[::1] king_fn2 = np.empty(npix,dtype=DTYPE)
    cdef double[::1] king_fn3 = np.empty(npix,dtype=DTYPE)
    cdef double[::1] king_fn4 = np.empty(npix,dtype=DTYPE)
    cdef double theta_center, phi_center, theta_1, phi_1, dist
    cdef Py_ssize_t i, j, pix

    cdef double tmp1,tmp2, tmp3, tmp4, tot1, tot2, tot3, tot4 
    cdef double k01, k02, k03, k04, rloc, rtot, kfval
    cdef double dnpix = float(npix)
    cdef double hpixsize = pow(2*pi/dnpix,0.5)

    #with nogil:
    if 1==1:
        print("Turned off nogil, TURN BACK ON!")
        k01 = 0.0
        k02 = 0.0
        k03 = 0.0
        k04 = 0.0
        rloc = 0.0
        rtot = 0.0
        print("Entering loop for pixel 0")
        for j in range(100):
            tmp1 = king_fn_full(rloc,fcore1,score1,gcore1,
                               stail1,gtail1,SpE1)
            tmp2 = king_fn_full(rloc,fcore2,score2,gcore2,
                               stail2,gtail2,SpE2)
            tmp3 = king_fn_full(rloc,fcore3,score3,gcore3,
                               stail3,gtail3,SpE3)
            tmp4 = king_fn_full(rloc,fcore4,score4,gcore4,
                               stail4,gtail4,SpE4)
            k01 += rloc*tmp1
            k02 += rloc*tmp2
            k03 += rloc*tmp3
            k04 += rloc*tmp4
            rtot += rloc
            rloc += hpixsize/100.
        
        k01 /= 100.0*rtot
        k02 /= 100.0*rtot
        k03 /= 100.0*rtot
        k04 /= 100.0*rtot

        print("Entering main npix loop")
        for pix in range(npix):
        #for pix in prange(npix,num_threads=threads):
            theta_center = theta_array[pix]
            phi_center = phi_array[pix]
            
            tot1 = 0.0
            tot2 = 0.0
            tot3 = 0.0
            tot4 = 0.0
            for i in range(npix):
                theta_1 = theta_array[i]
                phi_1 = phi_array[i]
                dist = dist_sphere(theta_center,theta_1,phi_center,phi_1)
                if dist == 0.0:
                    king_fn1[i] = k01
                    king_fn2[i] = k02
                    king_fn3[i] = k03
                    king_fn4[i] = k04
                    
                    tot1 += k01
                    tot2 += k02
                    tot3 += k03
                    tot4 += k04
                else:
                    if dist <= 5*SpE1*((score1+stail1)/2.):
                        tmp1 = king_fn_full(dist, fcore1,score1, gcore1, stail1, gtail1, SpE1)
                    else:
                        tmp1 = 0.0
                    king_fn1[i] = tmp1
                    tot1 += tmp1
                    
                    if dist <= 5*SpE2*((score2+stail2)/2.):
                        tmp2 = king_fn_full(dist, fcore2,score2, gcore2, stail2, gtail2, SpE2)
                    else:
                        tmp2 = 0.0
                    king_fn2[i] = tmp2
                    tot2 += tmp2

                    if dist <= 5*SpE3*((score3+stail3)/2.):
                        tmp3 = king_fn_full(dist, fcore3,score3, gcore3, stail3, gtail3, SpE3)
                    else:
                        tmp3 = 0.0
                    king_fn3[i] = tmp3
                    tot3 += tmp3

                    if dist <= 5*SpE4*((score4+stail4)/2.):
                        tmp4 = king_fn_full(dist, fcore4,score4, gcore4, stail4, gtail4, SpE4)
                    else:
                        tmp4 = 0.0
                    king_fn4[i] = tmp4
                    tot4 += tmp4
            
            for i in range(npix):
                kfval = 0.0
                kfval += king_fn1[i]/tot1
                kfval += king_fn2[i]/tot2
                kfval += king_fn3[i]/tot3
                kfval += king_fn4[i]/tot4
                kfval = kfval/4.0
                smoothed_map[i] += kfval * the_map[pix]
        print("Out of main loop") 
    return smoothed_map

# Define a python class
class king_smooth:
    """ Smooth a map using a king function """
    def __init__(self,maps_dir,ebin,eventclass,eventtype,threads=1):

        # Only support a few eventclasses and types right now, so check ok
        assert((eventclass == 2) | (eventclass == 5)), \
            "This eventclass is not currently supported (only Source and UCV)"
        assert((eventtype == 0) | (eventtype == 3)), \
            "This eventtype is not currently supported (only top or all)"

        # Determine the energy, this assumes using 40 log spaced bins from
        # 0.2 to 2000 GeV
        energyvalarray = 2*10**(np.linspace(-1,3,41)+0.05)[0:40]
        energyval = energyvalarray[ebin]

        # First load the king function parameters once and for all
        # either 1 or 4 quartiles
        kparam1 = kp.PSF_king(maps_dir,eventclass,1)
        self.fcore1=kparam1.return_king_params(energyval,'fcore')
        self.score1=kparam1.return_king_params(energyval,'score')
        self.gcore1=kparam1.return_king_params(energyval,'gcore')
        self.stail1=kparam1.return_king_params(energyval,'stail')
        self.gtail1=kparam1.return_king_params(energyval,'gtail')
        self.SpE1=kparam1.rescale_factor(energyval)

        self.threads = threads

        if (eventtype == 0):
            kparam2 = kp.PSF_king(maps_dir,eventclass,2)
            self.fcore2=kparam2.return_king_params(energyval,'fcore')
            self.score2=kparam2.return_king_params(energyval,'score')
            self.gcore2=kparam2.return_king_params(energyval,'gcore')
            self.stail2=kparam2.return_king_params(energyval,'stail')
            self.gtail2=kparam2.return_king_params(energyval,'gtail')
            self.SpE2=kparam2.rescale_factor(energyval)
            
            kparam3 = kp.PSF_king(maps_dir,eventclass,3)
            self.fcore3=kparam3.return_king_params(energyval,'fcore')
            self.score3=kparam3.return_king_params(energyval,'score')
            self.gcore3=kparam3.return_king_params(energyval,'gcore')
            self.stail3=kparam3.return_king_params(energyval,'stail')
            self.gtail3=kparam3.return_king_params(energyval,'gtail')
            self.SpE3=kparam3.rescale_factor(energyval)
            
            kparam4 = kp.PSF_king(maps_dir,eventclass,4)
            self.fcore4=kparam4.return_king_params(energyval,'fcore')
            self.score4=kparam4.return_king_params(energyval,'score')
            self.gcore4=kparam4.return_king_params(energyval,'gcore')
            self.stail4=kparam4.return_king_params(energyval,'stail')
            self.gtail4=kparam4.return_king_params(energyval,'gtail')
            self.SpE4=kparam4.rescale_factor(energyval)

        self.eventtype = eventtype

    def smooth_the_map(self,the_map):
        """ Return a smoothed version of the input map. This is a python 
            wrapper, which passes all the work to cython
        """

        npix = len(the_map)
        nside = hp.npix2nside(npix)
        theta_array, phi_array = hp.pix2ang(nside,np.arange(npix))

        if self.eventtype == 0:
            outmap = smooth_the_map_et0(the_map, theta_array, phi_array,
                                        self.fcore1, self.score1, self.gcore1,
                                        self.stail1, self.gtail1, self.SpE1,
                                        self.fcore2, self.score2, self.gcore2,
                                        self.stail2, self.gtail2, self.SpE2,
                                        self.fcore3, self.score3, self.gcore3,
                                        self.stail3, self.gtail3, self.SpE3,
                                        self.fcore4, self.score4, self.gcore3,
                                        self.stail4, self.gtail4, self.SpE4,
                                        npix,self.threads)
        else:
            outmap = smooth_the_map_et3(the_map, theta_array, phi_array,
                                        self.fcore1, self.score1, self.gcore1,
                                        self.stail1, self.gtail1, self.SpE1,
                                        npix,self.threads)
        
        return np.array(outmap)
