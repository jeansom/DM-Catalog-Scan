DM-Catalog-Scan
===============

**2MASS2Furious or using a Catalog of objects as a filter for astrophsical scans for Dark Matter**

AUTHORS
-------

*  Siddharth Mishra-Sharma; smsharma at princeton dot edu
*  Nicholas Rodd; nrodd at mit dot edu
*  Benjamin Safdi; bsafdi at mit dot edu
*  Mariangela Lisanti; mlisanti at princeton dot edu


Code Contents
-------------

The various folders within the code contain the following:

* **DataFiles:** data used by various parts of the code. Includes the following subfolders:

    - Catalogs: the catalog files to be used as filter for our scans

    - DM-Maps: full sky DM maps created from catalogs

    - MonteCarlo: MC ready for analysis

    - PP-Factor: files associated with calculating the particle physics factor

    - PS-Maps: individual and collective point source maps

* **Make-Catalogs:** generate catalogs of objects, including lists of their DM properties and associated uncertainties 

* **Make-DM-Map:** generate sky maps of DM halos

* **Make-Fermi-MC:** generate MC designed to mock the Fermi-LAT data

* **Scan-FullSky:** scans for the entire sky at once

* **Scan-Small-ROI:** scans for small regions of interests, aka stacking analysis

* **Smooth-Maps:** rapidly smooth maps using the Fermi PSF, also useful for generating point source models

To Do
-----

* Add the notebooks we use into the **Make-Catalogs** folder (DarkSky, 2MASS Tully/Lu, others?)

* Tidy up and complete the code in **Scan-Small-ROI**
