"""
Halo modeling
"""

import numpy as np
from scipy.special import erf, gamma
import scipy.interpolate as ip
import scipy.integrate as integrate
from scipy.optimize import fsolve
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import griddata, RegularGridInterpolator

import CosmologicalDistance as cd
from units import *

class HaloModel:
    """ Halo modeling, from virial mass and redshift to a J-factor. See note for details/refs.
    """
    def __init__(self, boost_model='bartels', M_min_halo=1e-6*M_s, alpha='self-consistent',
                 concentration_model='correa_darksky', data_folder='/group/hepheno/smsharma/Fermi-LSS/AdditionalData/',
                 omega_m = 0.295, omega_lambda = 0.705, h = 0.688):
        """
        Default cosmology from DarkSky-400 values -- Table 1 of 1510.05651.
        """
        self.boost_model = boost_model
        self.concentration_model = concentration_model
        self.M_min_halo = M_min_halo
        self.alpha = alpha

        self.data_folder = data_folder

        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.h = h

        # Get critical density corresponding to chosen parameters
        cosmo = FlatLambdaCDM(H0=100*self.h, Om0=self.omega_m)
        self.rho_c = cosmo.critical_density(0).value*1000*(Kilogram*Meter**-3)

        # For distance calculations
        self.universe = cd.CosmologicalDistance(omega_m=self.omega_m, omega_l=self.omega_lambda, h0=self.h)

        if self.boost_model == 'bartels':
            print "Loading bartels"
            self.load_boost_bartels()  # Load the boost factor files if using Bartels/Ando model

        if self.concentration_model in ['correa_darksky', "diemer_darksky", "prada_darksky",'correa_Planck15','diemer_Planck15']:
            print "Interpolating concentration for DarkSky"
            print "Using model", self.concentration_model
            self.load_concentration_custom() # Load the c-m relation files if using Correa model with custom cosmology

    # def load_concentration_custom(self):
    #     """ Load custom cosmology Correa c-m model
    #     """
    #     correa_ds = pd.read_csv(self.data_folder + "/darksky400_commah.txt", skiprows=5, delimiter=',',header=None)
    #     self.m200_ary = correa_ds[1]*M_s
    #     self.c200_ary = correa_ds[5]
    #     self.c200_correa_ip = ip.interp1d(self.m200_ary, self.c200_ary)

    def load_concentration_custom(self):
        """ Load custom DarkSky cosmology concentrations
        """
        if self.concentration_model == "correa_darksky": self.m_ary, self.z_ary, self.c_ary = np.load(self.data_folder+"m200_correa15.npy"), np.load(self.data_folder+"z_correa15.npy"), np.load(self.data_folder+"c200_correa15.npy")
        if self.concentration_model == "correa_Planck15": self.m_ary, self.z_ary, self.c_ary = np.load(self.data_folder+"m200_correa15_Planck15.npy"), np.load(self.data_folder+"z_correa15_Planck15.npy"), np.load(self.data_folder+"c200_correa15_Planck15.npy")
        if self.concentration_model == "diemer_darksky": self.m_ary, self.z_ary, self.c_ary = np.load(self.data_folder+"mvir_diemer15.npy"), np.load(self.data_folder+"z_diemer15.npy"), np.load(self.data_folder+"cvir_diemer15.npy")
        if self.concentration_model == "diemer_Planck15": self.m_ary, self.z_ary, self.c_ary = np.load(self.data_folder+"mvir_diemer15_Planck15.npy"), np.load(self.data_folder+"z_diemer15_Planck15.npy"), np.load(self.data_folder+"cvir_diemer15_Planck15.npy")
        if self.concentration_model == "prada_darksky": self.m_ary, self.z_ary, self.c_ary = np.load(self.data_folder+"mvir_prada12.npy"), np.load(self.data_folder+"z_prada12.npy"), np.load(self.data_folder+"cvir_prada12.npy")

        self.c_interp = RegularGridInterpolator([np.log10(self.m_ary),self.z_ary], self.c_ary)

    def set_NFW_params(self, M, z):
        """ To just calculate the NFW parameters
        """
        self.cvir = self.rvir = self.rhos = -999.
        self.cvir = self.c_vir(M, z)
        self.rvir = self.r_vir(M, z)
        self.rhos = self.rho_s(M, z)

    """
    ************************************************************
    * Concentration parameter model stuff
    ************************************************************
    """

    def c_vir(self, M, z):
        """ Either get from model or set to input value
        """

        if self.cvir == -999.:

            if self.concentration_model == "darksky_median":
                mmed, bmed = -0.085569827029482584, 1.9966423651976823
                self.cvir = 10**(mmed*np.log10(M/M_s)+bmed)
            if self.concentration_model == 'duffy':
                self.cvir = self.cvir_duffy(M, z)
            if self.concentration_model == 'prada':
                self.cvir = self.cvir_prada(M, z)
            if self.concentration_model == 'sanchez-conde':
                self.cvir = self.cvir_sanchez_conde(M, z)
            if self.concentration_model == 'correa':
                self.cvir = self.cvir_correa(M, z)
            if self.concentration_model in ['prada_darksky', 'diemer_darksky','diemer_Planck15']:
                self.cvir = self.c_interp([np.log10(M/M_s), z])[0]
            if self.concentration_model in ['correa_darksky','correa_Planck15']:
                c200_val = fsolve(lambda c200: c200 - self.c_interp([np.log10(self.M200_cvir(M, z, self.cvir_from_c200(c200, z))/M_s),z]), 10)[0]
                self.cvir = self.cvir_from_c200(c200_val, z)
                # if self.concentration_model == 'correa_darksky':
                #     self.cvir = self.cvir_correa_darksky(M, z)
                # else:
                #     pass
        return self.cvir

    def cvir_correa(self, M, z):
        """ 1502.00391, Equation 19. THIS IS C200 SO NEED TO CONVERT IT TO CVIR FIRST.
        """
        alpha = 1.62774 - 0.2458*(1+z) + 0.01716*(1+z)**2
        beta = 1.66079 + 0.00359*(1+z) - 1.6901*(1+z)**0.00417
        gamma = -0.02049 + 0.0253*(1+z)**-0.1044
        c200_val = fsolve(lambda c200: c200 - 10**(alpha + beta * np.log10(self.M200_cvir(M, z, self.cvir_from_c200(c200, z))/M_s)*(1+gamma*(np.log10(self.M200_cvir(M, z, c200)/M_s))**2)), 10)[0]
        cvir_val = self.cvir_from_c200(c200_val, z)
        return self.cvir_from_c200(c200_val, z)

    def cvir_correa_darksky(self, M, z):
        """ Custom cosmology Correa model (https://github.com/astroduff/commah). THIS IS C200 SO NEED TO CONVERT IT TO CVIR FIRST.
        """
        c200_val = fsolve(lambda c200: c200 - self.c200_correa_ip(self.M200_cvir(M, z, self.cvir_from_c200(c200, z))), 10)[0]
        cvir_val = self.cvir_from_c200(c200_val, z)
        return self.cvir_from_c200(c200_val, z)

    def cvir_sanchez_conde(self, M, z):
        """ 1312.1729, Equation 1. THIS IS C200 SO NEED TO CONVERT IT TO CVIR FIRST.
        """
        self.ci = [37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7]
        c200_val = fsolve(lambda c200: c200 - np.sum([(self.ci[i]*(np.log(self.M200_cvir(M, z, self.cvir_from_c200(c200, z))/(M_s/h)))**i) for i in range(0, 6)]), 10)[0]
        cvir_val = self.cvir_from_c200(c200_val, z)
        return self.cvir_from_c200(c200_val, z)

    def cvir_prada(self, M, z):
        """ 1104.5130 Equation 14. THIS IS C200 SO NEED TO CONVERT IT TO CVIR FIRST.
        """
        xval = (self.omega_lambda/self.omega_m)**(1/3)*(1+z)**-1
        c200_val = fsolve(lambda c200: c200 - self.B0(xval)*self.mathcal_C(self.sigma_prime(self.M200_cvir(M, z, self.cvir_from_c200(c200, z)), z)), 10)[0]
        cvir_val = self.cvir_from_c200(c200_val, z)
        # cvir_val = fsolve(lambda cvir: cvir - c200_val*((200/(self.Delta_c(z)))*M/self.M200_cvir(M,z,cvir))**(1./3.), 10.)
        return cvir_val

    def cvir_duffy(self, M, z):
        """ 0804.2486, Equation 4 and Table 1 with z = [0,2] and full halo sample. This is cvir so don't need to convert anything.
        """
        M_star=2*10**12*(self.h**-1*M_s)
        return 7.85*(M/M_star)**-.081*(1+z)**-0.71

    def delta_c(self, Delta_c, c):
        """ Critical overdensity
        """
        return Delta_c/3*c**3/(np.log(1+c)-c/(1+c))

    def cvir_from_c200(self, c200, z):
        """ Root-find by requiring that the critical overdensity math between the 
            virial and 200 definition (see https://arxiv.org/pdf/1005.0411.pdf, eq. 38)
        """
        return fsolve(lambda cvir: self.delta_c(self.Delta_c(z), cvir) - self.delta_c(200., c200), 10.)[0]

    def c200_from_cvir(self, cvir, z):
        """ Root-find by requiring that the critical overdensity math between the 
            virial and 200 definition (see https://arxiv.org/pdf/1005.0411.pdf, eq. 38)
        """
        return fsolve(lambda c200: self.delta_c(self.Delta_c(z), cvir) - self.delta_c(200., c200), 10.)[0]

    # def cvir_from_c200(self, c200, z):
    #     """ Phenomenological model from https://arxiv.org/pdf/1005.0411.pdf
    #     """
    #     p = -(8.683e-5)*(self.Delta_c(z))**1.82
    #     return c200 + c200**0.9*10**p

    def D(self, z):
        """ 1104.5130 Equation 12
        """
        xval = (self.omega_lambda/self.omega_m)**(1/3)*(1+z)**-1
        D_val = (5/2.)*(self.omega_lambda/self.omega_m)**(1/3)*np.sqrt(1+xval**3)/xval**1.5*integrate.quad(lambda x: x**1.5/(1+x**3)**1.5, 0, xval)[0]
        return D_val

    def sigma(self, M, z):
        """ 1104.5130 Equation 23
        """
        y = (M/(10**12*M_s/self.h))**-1
        sigma_val = self.D(z)*16.9*y**0.41/(1+1.102*y**.20+6.22*y**0.333)
        return sigma_val

    def B0(self, x):
        """ 1104.5130 Equation 18
        """
        return self.cmin(x)/self.cmin(1.393)

    def B1(self, x):
        """ 1104.5130 Equation 18
        """
        return self.sigmamin_inverse(x)/self.sigmamin_inverse(1.393)

    def cmin(self, x):
        """ 1104.5130 Equation 19
        """
        c0, c1, alpha, x0 = 3.681, 5.033, 6.948, 0.424
        return c0 + (c1-c0)*((1/np.sqrt(np.pi))*np.arctan(alpha*(x-x0))+.5)

    def sigmamin_inverse(self, x):
        """ 1104.5130 Equation 20
        """
        sigma0_inverse, sigma1_inverse, beta, x1 = 1.047, 1.646, 7.386, 0.526
        return sigma0_inverse + (sigma1_inverse-sigma0_inverse)*((1/np.sqrt(np.pi))*np.arctan(beta*(x-x1))+.5)

    def sigma_prime(self, M, z):
        """ 1104.5130 Equation 15
        """
        xval = (self.omega_lambda/self.omega_m)**(1/3)*(1+z)**-1
        return self.B1(xval)*self.sigma(M, z)

    def mathcal_C(self, sigmaprime):
        """1104.5130 Equation 16"""
        A, b, c, d = 2.881, 1.257, 1.022, 0.060
        return A*((sigmaprime/b)**c+1)*np.exp(d/sigmaprime**2)

    """
    ************************************************************
    * Boost model stuff
    ************************************************************
    """

    def load_boost_bartels(self):
        """ Load the Bartels/Ando boost model
        """
        a = '%e' % (self.M_min_halo/M_s)
        M_min_halo_str = a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]

        if M_min_halo_str not in ['1e-06', '1e-01', '1e+04']:
            print 'Specified minimum subhalo mass needs to be one of', ['1e-06', '1e-01', '1e+04']

        self.b_bartels_file = np.loadtxt(self.data_folder + 'boost_table/boost_table_'+str(self.alpha) + '_mmin'+M_min_halo_str + '.txt')
        self.Mvals_bartels = np.array([b[0] for b in self.b_bartels_file])*M_s
        self.bvals_bartels = np.array([b[1] for b in self.b_bartels_file])

        a = '%e' % self.M_min_halo
        M_min_halo_str = a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]
        self.bsh_bartels = ip.interp1d(self.Mvals_bartels, self.bvals_bartels)

    def bsh(self, M, z):
        """ Subhalo boost factor
        """
        bsh_val = 0

        if self.boost_model == 'gao':
            """ Boost model of Gao et al [1107.1916], Equation 1
            """
            bsh_val = 110*(self.M200(M, z)/(10**12*M_s))**0.39

        elif self.boost_model == 'gao_bpl':
            """ Boost model of Gao et al [1107.1916], Equation 1 with a broken power law cutoff below self.M_min_halo
            """
            if (M < self.M_min_halo*M_s):
                bsh_val = 110*(self.M200(self.M_min_halo*M_s, z)/(10**12*M_s))**0.39
            else:
                bsh_val = 110*(self.M200(M, z)/(10**12*M_s))**0.39

        elif self.boost_model == 'sanchez-conde_tidal':
            """ Boost model of Sanchez-Conde et al [1603.04057] accounting for tidal stripping, Equation 18
            """
            if self.alpha == 'alpha19':
                self.bi = [-6.8e-2, 9.4e-2, -9.8e-3, 1.05e-3, -3.4e-5, -2.e-7]

            elif self.alpha == 'alpha20':
                self.bi = [-0.186, 0.144, -8.8e-3, 1.13e-3, -3.7e-5, -2.e-7]
            bsh_val = 10**(np.sum([(self.bi[i]*(np.log10(self.M200(M, z)/M_s))**i) for i in range(0, 6)]))

        elif self.boost_model == 'sanchez-conde':
            """ Boost model of Sanchez-Conde et al [1312.1729], Equation 3
            """
            self.bi = [-0.442, 0.0796, -0.0025, 4.77*10**-6, 4.77*10**-6, -9.69*10**-8]
            bsh_val = 10**np.sum([(self.bi[i]*(np.log(self.M200(M, z)/M_s))**i) for i in range(0, 6)])

        elif self.boost_model == 'bartels':
            """ Boost model of Bartels and Ando [1507.08656]
            """
            if M < min(self.Mvals_bartels):
                bsh_val = 0
            elif M > max(self.Mvals_bartels):
                bsh_val = self.bsh_bartels(max(self.Mvals_bartels))
            else:
                bsh_val = self.bsh_bartels(M)

        elif self.boost_model == "none":
            bsh_val = 0

        return bsh_val

    """
    ************************************************************
    * d(), delta_c(), r_vir() relations from  [astro-ph/9710107]
    ************************************************************
    """

    def d(self, z):
        return (self.omega_m*(1+z)**3/(self.omega_m*(1+z)**3+self.omega_lambda))-1

    def Delta_c(self, z):
        return 18*np.pi**2+82*self.d(z)-39*self.d(z)**2

    def r_vir(self, M, z):
        if self.rvir == -999.:
            self.rvir = (3*M/(4*np.pi*self.Delta_c(z)*self.rho_c))**(1./3)
        else:
            pass
        return self.rvir

    """
    ************************************************************
    * NFW Jfactor for a given virial mass and redshift. If no cvir
    * is provided, the models c_vir() and r_vir() above are used.
    ************************************************************
    """

    def Jfactor(self, M, z, cvir=-999., rvir=-999.):

        self.cvir = cvir
        self.rvir = rvir

        dA = self.universe.angular_diameter_distance(z)*Mpc  # Distance to object

        Jf = (1+self.bsh(M, z))*self.a(self.c_vir(M, z))*self.rho_s(M, z)*M/(dA**2)
        return Jf

    def a(self, c): 
        """This is just a function of cvir that tacks on to the Jfactor formula
        """
        return ((1-1./(1+c)**3.)*(np.log(1+c)-c/(1+c))**-1)/3.

    def int_rhosq_dV(self, M, z, cvir=-999., rvir=-999.):

        self.cvir = cvir
        self.rvir = rvir
        return (1+self.bsh(M, z))*self.a(self.c_vir(M, z))*self.rho_s(M, z)*M

    def rho_s(self, M, z):  # NFW scale density using the empirical models above
        self.rhos = (M/(4*np.pi*(self.r_vir(M, z)/self.c_vir(M, z))**3))*(np.log(1+self.c_vir(M, z))-self.c_vir(M, z)/(1+self.c_vir(M, z)))**-1
        return self.rhos

    """
    ************************************************************
    * Burkert Jfactor for a given virial mass and redshift. If no 
    * cB is provided, the models c_vir() and r_vir() above are used.
    * Relations taken from [1507.08656]
    ************************************************************
    """

    def JfactorBurk(self, M, z, cB=-999., rvir=-999.):

        self.rvir = rvir
        self.cB = cB
        if cB == -999.:
            self.cB = self.c_vir(M, z) / 0.666511
        
        dA = self.universe.angular_diameter_distance(z)*Mpc  # Distance to object

        JB = (1+self.bsh(M, z))*self.rho_B(M, z)*M/(dA**2)*(self.cB*(1+self.cB+2*self.cB**2)/(1+self.cB+self.cB**2+self.cB**3)-np.arctan(self.cB))/(np.log((1+self.cB)**2*(1+self.cB**2))-2*np.arctan(self.cB))
        return JB

    def rho_B(self, M, z):
        self.rhoB =  M/(np.pi*(self.r_vir(M, z)/self.cB)**3*(np.log((1+self.cB)**2*(1+self.cB**2))-2*np.arctan(self.cB)))
        return self.rhoB


    """
    ************************************************************
    * Stuff to convert M_vir to M200 assuming NFW profile
    * Following Hu & Kravtsov [astro-ph/0203169]
    * See also http://background.uchicago.edu/mass_convert/
    ************************************************************
    """

    def M200(self, M_vir, z):

        return fsolve(lambda M200: M200 - M_vir*(self.c200_from_cvir(self.c_vir(M_vir, z), z)/self.c_vir(M_vir, z))**3*200./self.Delta_c(z), 1e12*M_s)[0]

    def M200_cvir(self, M_vir, z, cvir):

        return fsolve(lambda M200: M200 - M_vir*(self.c200_from_cvir(cvir, z)/cvir)**3*200./self.Delta_c(z), 1e12*M_s)[0]

    """
    ************************************************************
    * Stuff to convert M_vir to M200 assuming NFW profile
    * Following Hu & Kravtsov [astro-ph/0203169]
    * See also http://background.uchicago.edu/mass_convert/
    ************************************************************
    """

    # def M200_cvir(self, M_vir, z, cvir):

    #     fv = self.hk_f(1/cvir)
    #     f200 = 200.0*fv/self.Delta_c(z)
    #     x = self.hk_x(f200)

    #     c200 = 1.0/x
    #     Rratio = c200/cvir
    #     M200 = M_vir*(f200/fv)*Rratio**3
    #     return M200

    # def M200(self, M_vir, z):

    #     fv = self.hk_f(1/self.c_vir(M_vir, z))
    #     f200 = 200.0*fv/self.Delta_c(z)
    #     x = self.hk_x(f200)

    #     c200 = 1.0/x
    #     Rratio = c200 / self.c_vir(M_vir, z)
    #     M200 = M_vir*(f200/fv)*Rratio**3
    #     return M200

    # def hk_f(self, x):
    #     return x**3*(np.log(1.0 + 1.0/x) - 1.0/(1.0+x))

    # def hk_x(self, f):
    #     hk_a = [0.0, 0.5116, -0.4283, -3.13e-3, -3.52e-5]
    #     logf = np.log(f)
    #     twop = 2*(hk_a[2] + hk_a[3]*logf + hk_a[4]*logf*logf)
    #     return 1.0/np.sqrt(hk_a[1]*f**twop + 0.75**2) + 2*f
