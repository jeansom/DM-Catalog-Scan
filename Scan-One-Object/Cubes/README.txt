Data based on the catalog presented in 1708.09385 by Mariangela Lisanti, 
Siddharth Mishra-Sharma, Nicholas L. Rodd, and Benjamin R. Safdi

This folder contains 100 cubes corresponding to the top 100 objects in the catalog
available here: https://github.com/bsafdi/DMCat

Each cube is of shape: 40 x 5 x pixels in ROI:
  - 40: 40 energy bins, log spaced from 200 MeV to 2 TeV
  - 5: index the template type:
    - 0: location in an nside=128 map of this pixel 
    - 1: diffuse template
    - 2: isotropic template
    - 3: point source model template
    - 4: DM emission for this object template
  - pixels in ROI: index of the pixels in the 10 degree ROI
    we form around each object

The inclusion of the location at index 0 is done so we don't have to send the whole
healpix map.
