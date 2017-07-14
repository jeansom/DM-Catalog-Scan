# Code to place a king function at a given location

import numpy as np
import ptk

def ps_king(ell, b, maps_dir, nside, eventclass, eventtype, rescale=np.ones(40)):
    # add in a king function psf, which is the functional form of the Fermi PSF
    psk_inst = ptk.ps_template_king(ell, b, nside, maps_dir, eventclass, eventtype)
    return [rescale[i] * psk_inst.smooth_ps_map[i] for i in range(40)]
