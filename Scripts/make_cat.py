# @@ PYTHON SCRIPT TO CREATE A CUSTOM QUASAR CATALOG FROM DR12Q

from collections import OrderedDict
from PyAstronomy import pyasl

import numpy as np
import config_read as cfg
import fitsio
import timeit
import os

def main():
	# 1. Reading the Paris DR12Q catalog
	X = cfg.ConfigMap('MakeCat')
	if (X == 'FAIL'):
		sys.exit("Errors! Couldn't locate section in .ini file")

	data = fitsio.read(X['drqcat'], ext = int(X['drq_dataext']))

	plate, mjd, fiberid = data['PLATE'], data['MJD'], data['FIBERID'] 

	# Which redshift to use?
	z_use = data[cfg.drq_z]

	# Getting the extinction
	g_ext = data['EXTINCTION'][:,1]
	ebv = g_ext/3.793  # ref:  Schlegel, Finkbeiner & Davis 1998 (ApJ 500, 525, 1998; SFD98)

	print "The total number of objects in the catalog are %d" %len(ebv)
	# -------------------------------------------------------------------------------------

	# BAL flagged from visual inspection 
	# Ref: http://www.sdss.org/dr12/algorithms/boss-dr12-quasar-catalog/
	balFlag = data['BAL_FLAG_VI']

	# Ref: http://www.sdss.org/dr12/algorithms/bitmasks/#BOSSTILE_STATUS
	# targetFlag = data['BOSS_TARGET1'] ----- currently using everything
	ind = np.where((balFlag == 0) & (data['ZWARNING'] == 0) & (z_use > float(cfg.zmin)))[0]
	#---------------------------------------------------------------------------------------

	# 2b. Remove DLA's - Ref: (Noterdaeme et al. 2012c) http://adsabs.harvard.edu/abs/2012A%26A...547L...1N
	dla_data = np.loadtxt(X['dla_file'], skiprows=2, usecols=([1]), dtype='str').T
	dla_list = np.array([dla_data[i].split('-') for i in range(len(dla_data))])

	myinfo = np.array([mjd[ind], plate[ind], fiberid[ind]]).T
	target = dict((tuple(k),i) for i, k in enumerate(myinfo))
	candidate = [ tuple(i) for i in dla_list.astype(int)]

	# Way faster than before - http://stackoverflow.com/questions/1388818/how-can-i-compare-two-lists-in-python-and-return-matches
	inter = set(target).intersection(candidate) 
	indices = [target[x] for x in inter]

	ind[indices] = -1

	print "The total number of DLAs removed are %d" % len(indices)
	ind = ind[find(ind != -1)]

	# In case a test run is needed
	if cfg.run != 0:
		ind = np.random.choice(ind, size = run, replace=False)

	#---------------------------------------------------------------------------------------
	# Load in the sky-lines file - There are a lot of skylines!!!
	spec_dir = X['spec_dir']
	skylines = np.loadtxt(X['skyline_file']).T

	# These will be written to the final catalogue
	calFlux, calInvar = np.zeros((len(ind),NPIX)), np.zeros((len(ind),NPIX))

	nObj = len(ind)
	print 'Processing data. The total number of objects in the final catalog are %d' %nObj
	#------------------------------------------------------------------------------------
	start_time = timeit.default_timer()

	for k in range(nObj):
		x = ind[k]

		# Fetch the spectra
		SpecName = os.path.join(spec_dir, str(plate[x]), 'spec-%04d-%5d-%04d.fits' %(plate[x],mjd[x],fiberid[x]))
		try:
			a = fitsio.read(SpecName, ext=1)
		except IOError:
			continue
		else:		
			corr_flux, corr_ivar, loglam = a['flux'],a['ivar'],a['loglam']
			
			# Deredden the flux vector using the Fitzpatrick (1999) parameterization
			corr_flux, corr_ivar = pyasl.unred(10**loglam, corr_flux, corr_ivar, ebv=ebv[x])

			# 6. Shift to rest-frame of the quasar
			log2rest = loglam - np.log10(1 + z_use[x])

			# Nearest pixel assignment for flux and invar
			s = np.floor((log2rest - cfg.COEFF0)/cfg.COEFF1).astype(int)
			temp1 = where((s >= 0) & (s < cfg.NPIX))[0]
			calFlux[k,s[temp1]], calInvar[k,s[temp1]] = corr_flux[temp1], corr_ivar[temp1]

			# More efficient code for masking of skylines - halves the time as used before		
			minSky, maxSky = np.log10(skylines - 2) - np.log10(1 + z_use[x]), np.log10(skylines + 2)  - np.log10(1 + z_use[x])
			minT, maxT = np.floor((minSky - cfg.COEFF0)/cfg.COEFF1).astype(int), np.ceil((maxSky - COEFF0)/COEFF1).astype(int)
			for t in range(len(minT)): calInvar[k, minT[t]:maxT[t]] = 0
			
	elapsed_time = timeit.default_timer() - start_time
	print elapsed_time

	# 7. Write everything to a fits file
	if os.path.exists(cat_file):os.remove(cat_file) 

	fits = fitsio.FITS(cat_file,'rw')

	mydict = OrderedDict()
	keys = X['paramstowrite'].split(', ')
	for key in keys:mydict[key] = data[key][ind]

	# Write images and table to fits file
	hdict = {'COEFF0': cfg.COEFF0, 'COEFF1': cfg.COEFF1, 'NPIX': cfg.NPIX, 'AUTHOR': X['author']}
	fits.write(calFlux, header=hdict)
	fits.write(calInvar)
	fits.write(mydict)
	fits.close()
	print "Creation of catalogue complete. The file is %s." %cat_file
	#----------------------------------------------------------------------------------------

if __name__=="__main__":
	main()