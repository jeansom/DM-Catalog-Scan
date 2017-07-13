import numpy as np
import pyfits
import subprocess
import sys
import pandas as pd
import os
import healpy as hp
from glob import glob
import random
import copy
from tqdm import *
import matplotlib
#matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib as mpl

def mask_lat_band(lat_deg_min, lat_deg_max, nside):
    #make mask of band lat_deg_min < b < lat_deg_max
    mask_none = np.arange(hp.nside2npix(nside))
    return (np.radians(lat_deg_min) <= hp.pix2ang(nside, mask_none)[0]) * \
            (hp.pix2ang(nside, mask_none)[0] <= np.radians(lat_deg_max))

def mask_not_lat_band(lat_deg_min, lat_deg_max, nside):
    #make mask of region outside band lat_deg_min < b < lat_deg_max
    return np.logical_not(mask_lat_band(lat_deg_min, lat_deg_max, nside))

def mask_ring(ring_deg_min, ring_deg_max, center_theta_deg, center_phi_deg, nside):
    #make mask of region outside ring_deg_min < theta < ring_deg_max
    mask_none = np.arange(hp.nside2npix(nside))
    return (np.cos(np.radians(ring_deg_min)) >= np.dot(hp.ang2vec(np.radians(center_theta_deg), np.radians(center_phi_deg)), hp.pix2vec(nside, mask_none))) * \
            (np.dot(hp.ang2vec(np.radians(center_theta_deg), np.radians(center_phi_deg)), hp.pix2vec(nside, mask_none)) >= np.cos(np.radians(ring_deg_max)))

def dot_norm(a,b):
    # Normalized dot product
    return np.ma.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def correlate_random_maps(map1,map2,nrand):
    '''
    Function to produce a distribution of dot products of two maps, 
    each time shuffling the two maps shuffling the first map a given number of times.
    '''
    dp_distrib = []
    for i in tqdm(range(nrand)):
        dp_distrib.append(dot_norm(randomizer(map1),map2))
    return dp_distrib

def plot_corr_distrib(distrib,map1,map2,title):
    '''
    Function to plot distribution of random dot products and compare to
    actual value for the two maps.
    '''
    # Bin size based on number of values in distrib
    iqr = np.subtract(*np.percentile(distrib, [75, 25])) 
    h = 2*iqr*(len(distrib))**(-1./3.)
    nbins = (np.max(distrib)-np.min(distrib))/h

    # nbins=10
    
    plt.rcParams['figure.figsize'] = 14, 6 
    dp84=np.percentile(distrib,84)
    dp50=np.percentile(distrib,50)
    dp16=np.percentile(distrib,16)
    dp=dot_norm(map1,map2)

    plt.hist(distrib,bins=np.linspace(np.min(distrib),np.max(distrib),nbins),color='green',\
             histtype='stepfilled',alpha=.3)
    plt.axvline(np.percentile(distrib,84),linestyle='dashed', linewidth=1.5, color='orange')
    plt.axvline(np.percentile(distrib,16),linestyle='dashed', linewidth=1.5,color='orange')
    plt.axvline(np.percentile(distrib,50),linestyle='dashed', linewidth=1.5, color='red')
    plt.axvline(dot_norm(map1,map2),linewidth=2, color='black')
    plt.title(title,y=1.03)
    print "actual minus median: ", dp - dp50
    print "(actual minus median)/(84%ile - median): ", (dp-dp50)/(dp84-dp50)

def shuffle(array):
    # Shuffle array/indices without changing original array
    random.shuffle(array)
    return array 

def get_unmask_ind(masked_array):
    # Return indices for unmasked pixels
    indxs=[]
    for idx, item in enumerate(masked_array.mask):
        if not(item): indxs.append(idx)
    return indxs

    
def randomizer(masked_array):
    '''
    Function to shuffle unmasked pixels in a map
    '''
    unmask_ind = get_unmask_ind(masked_array)
    unmask_ind_shuffle = shuffle(unmask_ind)
    shuffle_masked_array = copy.copy(masked_array)
    shuffle_masked_array[shuffle(unmask_ind)] = masked_array[unmask_ind_shuffle]

    return shuffle_masked_array

def radec2pix(nside,ra,dec,nest=False):
    '''
    Function to convert right ascension and descent to galactic
    coordinates, then get index corresponding to healpix array
    '''
    l_gal,b_gal = eq2galCoords(ra,dec)
    return hp.ang2pix(nside,np.deg2rad(90-b_gal),np.deg2rad(l_gal),nest=nest)

def lb2pix(nside,l_gal,b_gal,nest=False):
    '''
    Function to get index corresponding to healpix array from galactic l,b
    '''
    return hp.ang2pix(nside,np.deg2rad(90-b_gal),np.deg2rad(l_gal),nest=nest)

def eq2galCoords(RA,DE):

    '''
    Function to convert ecliptical coordinates $(RA,DEC)$ (J2000 epoch) to galactic 
    coordinates $(\ell,b)$ as described in the Hipparcos documentation 
    (http://www.rssd.esa.int/SA-general/Projects/Hipparcos/CATALOGUE_VOL1/sect1_05.pdf):
    '''

    # Transformation matrix between equatorial and galactic coordinates:
    # [x_G  y_G  z_G] = [x_E  y__E  z_E] . A_G

    A_G = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                    [-0.8734370902, -0.4448296300, -0.1980763734],
                    [-0.4838350155, +0.7469822445, +0.4559837762]])

    deg = np.pi/180
    rad = 180./np.pi


    ra = np.array(RA).reshape(-1)*deg
    de = np.array(DE).reshape(-1)*deg
    assert len(ra) == len(de)

    sde,cde = np.sin(de),np.cos(de)
    sra,cra = np.sin(ra),np.cos(ra)
    
    aux0 = A_G[0,0]*cde*cra + A_G[1,0]*cde*sra + A_G[2,0]*sde
    aux1 = A_G[0,1]*cde*cra + A_G[1,1]*cde*sra + A_G[2,1]*sde
    aux2 = A_G[0,2]*cde*cra + A_G[1,2]*cde*sra + A_G[2,2]*sde

    b = np.arcsin(aux2)*rad
    l = np.arctan2(aux1,aux0)*rad
    l[l<0] += 360.

    if len(l) == 1:
        return l[0],b[0]
    else:
        return l,b

def default_parameters():
    return {
        "apodizesigma":"NO",
        "apodizetype":"NO",
        "beam":"NO",
        "beam_file":"beam.csv",
        "beam2":"NO",
        "beam_file2":"beam.csv",
        "corfile":"spiceTEMP_spice_cor.fits",
        "clfile":"spiceTEMP_spice_cl.fits",
        "cl_outmap_file":"NO",
        "cl_inmap_file":"NO",
        "cl_outmask_file":"NO",
        "cl_inmask_file":"NO",
        "covfileout":"NO",
        "decouple":"NO",
        "dry":"YES",
        "extramapfile":"YES",
        "extramapfile2":"YES",
        "fits_out":"YES",
        "kernelsfileout":"NO",
        "mapfile":"YES",
        "mapfile2":"NO",
        "maskfile":"total_mask.fits",
        "maskfile2":"total_mask.fits",
        "maskfilep":"total_mask.fits",
        "maskfilep2":"total_mask.fits",
        "nlmax":"1024",
        "normfac":"NO",
        "npairsthreshold":"NO",
        "noisecorfile":"NO",
        "noiseclfile":"NO",
        "overwrite":"YES",
        "polarization":"YES",
        "pixelfile":"YES",
        "subav":"NO",
        "subdipole":"NO",
        "symmetric_cl":"NO",
        "tf_file":"NO",
        "thetamax":"NO",
        "verbosity":"1",
        "weightfile":"NO",
        "weightfilep":"YES",
        "weightfile2":"NO",
        "weightfilep2":"YES",
        "weightpower":"1.00000000000000",
        "weightpower2":"1.00000000000000",
        "weightpowerp":"1.00000000000000",
        "weightpowerp2":"1.00000000000000",
        "windowfilein":"NO",
        "windowfileout":"NO"
}

def write_params(params, filename="spiceTEMP_spice_params.txt"):
    with open(filename, "w") as f:
        for k in sorted(params.keys()):
            f.write("%s = %s\n" % (k, params[k]))

def read_cl(filename="./spiceTEMP_spice_cl.fits"):
    cl_file = pyfits.open(filename)
    cl_file[1].verify("fix")
    cl = pd.DataFrame(np.array(cl_file[1].data))
    # convert from string (?!) to float
    cl[cl.keys()] = cl[cl.keys()].astype(np.float32)
    cl.index.name = "ell"
    return cl

def read_cor(filename="./spiceTEMP_spice_cor.fits"):
    cor_file = pyfits.open(filename)
    cor_file[1].verify("fix")
    cor = pd.DataFrame(np.array(cor_file[1].data))
    # convert from string (?!) to float
    cor[cor.keys()] = cor[cor.keys()].astype(np.float32)
    cor.index.name = "idx"
    return cor    

def compute_llp1(ell):
    return ell * (ell+1) / 2. / np.pi

def compute_power(ell):
    return (2*ell+1) / 4. / np.pi # power

def weighted_average(g):
    weights = compute_power(g.index)
    return g.mul(weights, axis="index").sum()/weights.sum() 

def bin_cl(cl):
    ctp_binning = pd.read_csv("planck_ctp_bin.csv")
    #ctp_binning = ctp_binning[0:ctp_binning["last l"].searchsorted(1024)+1]
    bins = pd.cut(cl.index, bins=np.concatenate([[1], ctp_binning["last l"]]))
    cl["ell"] = np.float64(cl.index)
    binned_cl = cl.groupby(bins).apply(weighted_average)
    binned_cl = binned_cl.set_index("ell")
    return binned_cl

def spice(bin=True, norm=True, **kwargs):
    """Run spice (needs to be in PATH)

    Parameters
    ----------
    bin : bool
        apply C_ell binning (planck_ctp)
    norm : bool
        return C_ell * ell(ell+1)/2pi [muK^2]
    other_arguments:
        all spice arguments, see `default_params` in the module source
        for example:
        mapfile is the input map for spectra, assumed to be in [K]
        mapfile2 is the second map for cross-spectra

    Returns
    -------
    cl : pd.Series
        pandas Series of C_ell[K^2] (if norm is False) or C_ell * ell(ell+1)/2pi [muK^2]
        (if norm is True)
    """
    params = default_parameters()

    # write arrays maps to disk
    for key in ["mapfile", "mapfile2"]:
        if kwargs.has_key(key):
            if not isinstance(kwargs[key], str):
                temp_map_file = "spiceTEMP_" + key + ".fits"
                hp.write_map(temp_map_file, kwargs[key])
                kwargs[key] = temp_map_file

    params.update(kwargs)
    write_params(params)
    try:
        subprocess.check_output(["/Users/siddharth/PolSpice_v03-00-03/src/spice", "-optinfile", "./spiceTEMP_spice_params.txt"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error in spice")
        print(e.output)
        sys.exit(1)
    cl = read_cl()
    cor = read_cor()
    # Remove created fits file
    for f in glob("spiceTEMP*"):
        os.remove(f)
    # Remove created mask file
    for f in glob("*mask*"):
        os.remove(f)        
    lmax = cl.index.max()
    imax = cor.index.max()

    # Binning and normalizaton in ell; need to implement binning in theta
    if bin:
        cl = bin_cl(cl)
    if norm:
        cl = cl.mul(1e12 * compute_llp1(np.array(cl.index)), axis="index")
    return cl[cl.index < lmax], cor[cor.index < imax]