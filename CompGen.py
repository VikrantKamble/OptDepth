from __future__ import division
import config_read as cfg
import numpy as np
import matplotlib.pyplot as plt
import fitsio
import calibrate2

reload(calibrate2)

# HELPER FUNCTION TO SET THE REDSHIFT BINS <- Refine this
def adapt_hist(z, p, n_chop):
	low, up, z_diff = min(z), max(z), 0.2

	# The edges in the parameter space - bins will be n_chop - 1
	Chop1 = np.linspace(min(p[0]), max(p[0]), n_chop[0])
	Chop2 = np.linspace(min(p[1]), max(p[1]), n_chop[1])

	while True:
		z_edges = np.arange(low, up, z_diff)

		n_zbins = len(z_edges) - 1 

		# Make a 3D DataCube
		mCube = np.zeros((n_zbins, n_chop[0]-1, n_chop[1]-1))

		## MAKING HISTOGRAM OVER ALL BINS
		for i in range(n_zbins):
			# Find the ones that belong in a certain zbin
			ind = np.where((z > z_edges[i]) & (z <= z_edges[i+1]))[0]

			# Make the appropriate histogram of the parameters
			mCube[i] = np.histogram2d(p[0][ind], p[1][ind], bins=(Chop1,Chop2))[0]

		## TRIM BINS WITH NO OBJECTS 
		# Outer - parameter; Inner - redshift
		for i in range(n_chop[0]-1):
			for j in range(n_chop[1]-1):
				# Sets all bins to 0 if any one bin has no objects in it
				if (0 in mCube[:,i,j]):mCube[:,i,j]=0 

		check = np.ravel(mCube[0])

		foo = len(check[check == 0])

		if foo < 6:
			break
		up -= z_diff

	print 'Number of redshift bins obtained is %d, and the number of bins in parameter space rejected are %d' %(n_zbins, foo)

	for i in range(n_zbins):
		ind = np.where((z > z_edges[i]) & (z <= z_edges[i+1]))[0]
		print len(ind)

	return z_edges

def CompCompute(ext, spec, ivar, z, p, z_edges, n_chop, outfile, boot=False):

	# Total number of bins in redshift space 
	n_zbins = len(z_edges) - 1

	# Reshuffling for bootstrap and file nomenclature -------------------------------------------
	if boot==True:
		outfile += 'boot' +'_' + str(ext) + '.fits'

		bucket = []
		for i in range(n_zbins):
			ind = np.where((z > z_edges[i]) & (z <= z_edges[i+1]))[0]

			# Resample the objects in each resdhift bin
			bootInd = np.random.choice(len(ind), size=len(ind), replace=True)

			# Collect all the resampled indices
			bucket += list(ind[bootInd])

		spec, ivar, z, p = spec[bucket], ivar[bucket], z[bucket], p[:,bucket]
	else:
		outfile += '_' + str(ext) + '.fits'
	# -------------------------------------------------------------------------------------------

	NPIX = cfg.NPIX

	Chop1 = np.linspace(min(p[0]), max(p[0]), n_chop[0])
	Chop2 = np.linspace(min(p[1]), max(p[1]), n_chop[1])

	# Make a 3D DataCube
	mCube = np.zeros((n_zbins, n_chop[0]-1, n_chop[1]-1))

	## MAKING HISTOGRAM OVER ALL BINS
	for i in range(n_zbins):
		# Find the ones that belong in a certain zbin
		ind = np.where((z > z_edges[i]) & (z <= z_edges[i+1]))[0]

		# Make the appropriate histogram of the parameters
		mCube[i] = np.histogram2d(p[0][ind], p[1][ind], bins=(Chop1,Chop2))[0]

	## TRIM BINS WITH NO OBJECTS 
	# Outer - parameter; Inner - redshift
	for i in range(n_chop[0]-1):
		for j in range(n_chop[1]-1):
			# Sets all bins to 0 if any one bin has no objects in it
			if (0 in mCube[:,i,j]):mCube[:,i,j]=0 

	mCount = np.sum(mCube, axis=0) 

	# A. NORMALIZED WEIGHTS ACROSS ALL REDSHIFTS 
	p0_bins, p1_bins = Chop1, Chop2   

	p0_bins[-1] += 0.001   # <-- Required since histogram2d and digitize have different
	p1_bins[-1] += 0.001   #     binning scheme

	foo = np.digitize(p[0], p0_bins)
	blah = np.digitize(p[1], p1_bins)

	Z_Ind = np.digitize(z, z_edges, right=True) 

	F = mCount / mCube
	F[np.isnan(F)] = 0

	# To obtain consistent weights across all redshifts
	F = F / np.linalg.norm(F, axis=(1,2))[:,None,None]

	hist_weights = F[Z_Ind-1 , foo-1, blah-1]

	# 1. PERFORM CALIBRATION IF SPECIFIED
	if X['calib'] == True:
		# The rest-frame range used for calibrating
		rest_range = [[1280,1290],[1320,1330],[1345,1360],[1440,1500]]

		# Range used for normalizing the calibration vector
		obs_min, obs_max = 4100, 4200

		# Obtain the required mask that spans the calibration redshifts
		target = [Z_Ind < np.digitize(X['calib_max'], z_edges, right=True)]

		calibrate2.calibrate(spec[target], ivar[target], z[target], hist_weights[target], rest_range, obs_min, obs_max)

	# 2. COMPOSITE GENERATION IF SPECIFIED
	if X['comp'] == True:	
		target = np.digitize(X['comp_min'], z_edges, right=True)

		#Arrays that will store the composites and respective errors
		COMP, IVAR = np.zeros((n_zbins, NPIX)), np.zeros((n_zbins, NPIX))
		RED        = np.zeros(n_zbins)

		for i in range(target, n_zbins):
			ind = np.where(Z_Ind == (i+1))[0]

			loc_spec, loc_ivar, lever = spec[ind], ivar[ind], hist_weights[ind]

			for j in range(cfg.NPIX):
				temp = np.where(loc_ivar[:,j] * lever > 0)[0]

				if (len(temp) > 0): # to prevent estimates that can't be normalized
					vals, w2, w1 = loc_spec[:,j][temp], loc_ivar[:,j][temp], lever[temp]

					# Trimming outliers with 5 percent rejection
					outInd = np.where((vals > np.percentile(vals, 5)) & (vals < np.percentile(vals, 95)))[0]

					if (len(outInd) > 5):
						# SIMPLE WEIGHTED AVERAGE
						COMP[i,j] = np.average(vals[outInd], weights=w2[outInd]*w1[outInd])
						IVAR[i,j] = np.average(w2[outInd], weights=w1[outInd])*len(w2[outInd])
				
			# The redshift of the composite:
			RED[i] = np.average(z[ind], weights=lever)

		# ----------------------------------------------------------------------------
		fits = fitsio.FITS(outfile,'rw')

		# Writing operation
		fits.write(COMP)
		fits.write(IVAR)
		fits.write([RED], names=['REDSHIFT'])

		fits.close()
		print "Writing of the composite to fits file complete. Filename: %s" %outfile
