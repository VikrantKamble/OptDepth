from __future__ import division
import config_read as cfg
import numpy as np
import os
import json
import fitsio
import shutil
import sys

# HELPER FUNCTION TO SET THE REDSHIFT BINS
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

def CompCompute(spec, ivar, z, p, z_edges, n_chop, outfile, boot=False):

	n_zbins = len(z_edges) - 1

	if boot==True:
		bucket = []
		for i in range(n_zbins):
			ind = np.where((z > z_edges[i]) & (z <= z_edges[i+1]))[0]

			# Resample the objects in each resdhift bin
			bootInd = np.random.choice(len(ind), size=len(ind), replace=True)

			# Collect all the resampled indices
			bucket += list(ind[bootInd])

		spec, ivar, z, p = spec[bucket], ivar[bucket], z[bucket], p[:,bucket]

	NPIX = len(spec[0])

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

	mCount = np.sum(mCube, axis=0) 

	## TRIM BINS WITH NO OBJECTS 
	# Outer - parameter; Inner - redshift
	for i in range(n_chop[0]-1):
		for j in range(n_chop[1]-1):
			# Sets all bins to 0 if any one bin has no objects in it
			if (0 in mCube[:,i,j]):mCube[:,i,j]=0 

	## DISTRIBUTE WEIGHTS TO OBJECTS AND CREATE COMPOSITE
	# digitize and histogram have different binning logic - this takes care of that
	Chop_dig1, Chop_dig2 = Chop1, Chop2   
	Chop_dig1[-1] += 0.001
	Chop_dig2[-1] += 0.001

	# Get the histogram bins where each spectra falls
	foo = np.digitize(p[0], Chop_dig1)
	blah = np.digitize(p[1], Chop_dig2)

	mCount = np.sum(mCube, axis=0) 

	# Arrays that will store the composites and respective errors
	COMP, IVAR = np.zeros((n_zbins, NPIX)), np.zeros((n_zbins, NPIX))
	RED        = np.zeros(n_zbins)

	for i in range(n_zbins):
		ind = np.where((z > z_edges[i]) & (z <= z_edges[i+1]))[0]

		loc_spec, loc_ivar = spec[ind], ivar[ind]

		# Weight assigned as per the ratio
		lever = np.where(mCount[foo[ind] - 1, blah[ind] - 1] == 0, 0 , mCount[foo[ind] - 1, blah[ind] - 1]/mCube[i, foo[ind] - 1, blah[ind]- 1])

		for j in range(NPIX):
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

	#OVERWRITE THE FILE IF ALREADY PRESENT
	if os.path.exists(outfile):os.rmdir(outfile) 

	fits = fitsio.FITS(outfile,'rw')

	hdict = {'COEFF0':cfg.COEFF0, 'COEFF1': cfg.COEFF1, 'NPIX': cfg.NPIX, 'AUTHOR': cfg.author}

	# Writing operation
	fits.write(COMP, header=hdict)
	fits.write(IVAR)
	fits.write([RED], names=['REDSHIFT'])

	fits.close()
	print "Writing of the composite to fits file complete. Filename: %s" %outfile
	

# GENERATES COMPOSITES FOR A GIVEN CATALOG AND PARAMETER CUTS
def CompGen(spec, ivar, tb, sn):

	z = tb[cfg.drq_z]

	X = cfg.ConfigMap('CompGen')
	if (X == 'FAIL'):sys.exit("Errors! Couldn't locate section in .ini file")

	# Load in the basis parameters with their rough span
	pSpace = X['param_space'].split(',')
	pSpan = json.loads(X['param_span']) 

	# Overall cuts and specific bin selection
	pBin = json.loads(X['param_nbins'])
	pSel = json.loads(X['param_sel'])

	# Grid for histogram rebinning
	n_chop = json.loads(X['n_chop'])

	# Spectra with good fits
	chi_min, chi_max, snt = float(X['chi_min']), float(X['chi_max']), float(X['sn_threshold'])

	p1, p2 = tb[pSpace[0]], tb[pSpace[1]]

	# Get chi_square on parameter fits
	chir_p1 = tb['CHISQ_' + pSpace[0]] / tb['DOF_' + pSpace[0]]
	chir_p2 = tb['CHISQ_' + pSpace[1]] / tb['DOF_' + pSpace[1]]

	x_edges = np.linspace(pSpan[0][0], pSpan[0][1], pBin[0])
	y_edges = np.linspace(pSpan[1][0], pSpan[1][1], pBin[1])

	# Get the parameter working bin
	print '%s = %.2f, %.2f \n%s = %.2f, %.2f \n' %(pSpace[0], x_edges[pSel[0]-1], x_edges[pSel[0]], pSpace[1], y_edges[pSel[1]-1], y_edges[pSel[1]]) 

	t = np.where((z > 2.3) & (sn > snt) & (chir_p1 > chi_min) & (chir_p1 <= chi_max) &  (chir_p2 > chi_min) & (chir_p2 <= chi_max) & (p1 > x_edges[pSel[0]-1]) & (p1 <= x_edges[pSel[0]]) & (p2 > y_edges[pSel[1]-1]) & (p2 <= y_edges[pSel[1]]))[0]

	print 'Total number of objects in this bin are: %d \n' %len(t)

	myp = np.array([p1[t],p2[t]])

	# Normalization Range
	wlInd = np.where((cfg.wl > 1445) & (cfg.wl < 1455))[0]

	# Normalize the spectra and the variance 
	scale   = np.median(spec[t][:,wlInd], axis=1)
	myspec  = spec[t] / scale[:, None]
	myivar  = ivar[t] * (scale[:,None])**2
	myz = z[t]

	print 'min(z) = %.2f, max(z) = %.2f' %(min(myz), max(myz))

	# Defining the redshift bins
	z_edges = adapt_hist(myz, myp, n_chop)

	print 'Redshift bins obtained are:', z_edges

	# Name of the output file, tagged for boot
	outfile = 'comp_' + X['comp_ver'] + '_' + str(pBin[0]) + str(pBin[1]) + '_' + str(pSel[0]) + str(pSel[1]) + '_'  + X['comp_suffix'] 

	# Run suffiicient boot samples to get Covaraince Matrix that is PSD
	nboot = int(X['nboot'])

	# CREATE A NEW DIRECTORY AND PUT COMPOSITE AND ITS BOOT COMPOSITES THERE -------------------
	dir_file = os.environ['OPT_COMPS'] + outfile

	if os.path.exists(dir_file) == True:shutil.rmtree(dir_file)
	
	# Make directory and cd to it
	os.makedirs(dir_file)
	mydir = os.getcwd()
	os.chdir(dir_file)

	try:
		CompCompute(myspec, myivar, myz, myp, z_edges, n_chop, outfile+'.fits')

		for k in range(nboot):
			loc_file = outfile + '_boot' +'_' + str(k)+'.fits'
			CompCompute(myspec, myivar, myz, myp, z_edges, n_chop, loc_file, True)
	except Exception as e:
		print e.__doc__
		print e.message
		os.chdir(mydir)

	# Return to the previous working directory
	os.chdir(mydir)

	return outfile
	