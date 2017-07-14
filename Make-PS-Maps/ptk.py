import numpy as np
import healpy as hp
import smooth_with_king as swk

class ps_template_king:
    # Class to create a point source template at a given location
    # Updated to include up and down binning in nside
    def __init__(self,ell,b,nside,maps_dir,eventclass,eventtype):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.ell = ell
        self.b = b
        self.maps_dir = maps_dir
        self.eventclass = eventclass
        self.eventtype = eventtype

        self.ell_and_b_to_pix()
        self.make_ps_template()

    def ell_and_b_to_pix(self):
        # Make a point source at the origin and rotate it later (get better behaviour near poles)
        self.phi = 0
        self.theta = np.pi/2

        # Upbin the nside depending on the value of the psf
        nsideupbin = 1024
        self.nsideupbin = nsideupbin
        self.npixupbin = hp.nside2npix(self.nsideupbin)
        self.pix_num = hp.ang2pix(self.nsideupbin,self.theta,self.phi)

    def make_ps_template(self):
        the_map = np.zeros(self.npixupbin)
        the_map[self.pix_num] = 1
        self.smooth_ps_map = np.zeros(shape=(40,self.npix))
        for ebin in range(40):
            # Smooth using a king function
            self.skf = swk.smooth_king_psf(self.maps_dir,the_map,ebin,self.eventclass,self.eventtype)
            # Now rotate the map
            thetarot, phirot = hp.pix2ang(self.nsideupbin, np.arange(self.npixupbin))
            r = hp.Rotator(rot = [self.ell,self.b], coord=None, inv=False, deg=False, eulertype='ZYX')
            thetarot, phirot = r(thetarot, phirot)
            pixrot = hp.ang2pix(self.nsideupbin, thetarot, phirot, nest=False)
            smooth_ps_map_rot = self.skf.smooth_the_pixel(self.pix_num)[..., pixrot]
            # Downbin for the final map
            smooth_ps_map = hp.ud_grade(smooth_ps_map_rot,self.nside,power=-2)
            # Normalise back to 1 (gets mucked up by rotation and downbinning)
            self.smooth_ps_map[ebin] = smooth_ps_map/np.sum(smooth_ps_map)
