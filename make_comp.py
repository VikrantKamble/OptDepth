"""
1. Generate Composites Given Input Spectra 
2. Implements Selection Cuts And Histogram Rebinning 
3. Uses Multiprocessing Library To Generate Bootstrap Realizations

Note: Removes all the files if the folder already exists
"""

import numpy as np
import sys
import json
import os
import shutil


import config_read as cfg
import CompGen

def CompGen(spec, ivar, tb, sn):

	z = tb[cfg.drq_z]

	# LOAD IN FILE SPECIFIC PARAMETERS -----------------------------------------------------
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
	# SN calculated over the range same as that used for spectral index
	chi_min, chi_max, snt = float(X['chi_min']), float(X['chi_max']), float(X['sn_threshold'])

	# Load in the basis parameters
	p1, p2 = tb[pSpace[0]], tb[pSpace[1]]

	# Get chi_square on parameter fits
	chir_p1 = tb['CHISQ_' + pSpace[0]] / tb['DOF_' + pSpace[0]]
	chir_p2 = tb['CHISQ_' + pSpace[1]] / tb['DOF_' + pSpace[1]]

	p0_bins = np.linspace(pSpan[0][0], pSpan[0][1], pBin[0])
	p1_bins = np.linspace(pSpan[1][0], pSpan[1][1], pBin[1])

	print '%s = %.2f, %.2f \n%s = %.2f, %.2f \n' %(pSpace[0], p0_bins[pSel[0]-1], p0_bins[pSel[0]], pSpace[1], p1_bins[pSel[1]-1], p1_bins[pSel[1]]) 

	# ---------------------------------------------------------------------------------------

	# Indices of the spectra 'used' - Send all z's and then chose according to composite creation or calibration
	# Add necessary parameters to the configuration file

	t = np.where((sn > snt) & (chir_p1 > chi_min) & (chir_p1 <= chi_max) &  (chir_p2 > chi_min) & (chir_p2 <= chi_max) & (p1 > x_edges[pSel[0]-1]) & (p1 <= x_edges[pSel[0]]) & (p2 > y_edges[pSel[1]-1]) & (p2 <= y_edges[pSel[1]]))[0]

	print 'Total number of objects in this bin are: %d \n' %len(t)

	# Normalization Range
	wlInd = np.where((cfg.wl > 1445) & (cfg.wl < 1455))[0]

	# Normalize the spectra and the variance #####################
	scale   = np.median(spec[t][:,wlInd], axis=1)

	myspec  = spec[t] / scale[:, None]
	myivar  = ivar[t] * (scale[:,None])**2
	##############################################################
	
	myz = z[t]
	myp = np.array([p1[t],p2[t]])

	# Defining the redshift bins
	z_edges = adapt_hist(myz, myp, n_chop)
	print 'Redshift bins obtained are:', z_edges

	# Name of the output file
	outfile = 'comp_' + X['comp_ver'] + '_' + str(pBin[0]) + str(pBin[1]) + '_' + str(pSel[0]) + str(pSel[1]) + '_'  + X['comp_suffix'] 

	# Run sufficient boot samples to get Covaraince Matrix that is PSD
	nboot = int(X['nboot'])

	# CREATE A NEW DIRECTORY AND PUT COMPOSITE AND ITS BOOT COMPOSITES THERE -------------------
	dir_file = os.environ['OPT_COMPS'] + outfile

	if os.path.exists(dir_file) == True:shutil.rmtree(dir_file)
	
	# Make directory and cd to it
	os.makedirs(dir_file)

	mydir = os.getcwd()

	os.chdir(dir_file)

	# -------------------------------------------------------------------------------------------
	# FUNCTION TO ALLOW POOL TO ACCEPT ADDITIONAL ARGUMENTS
	myfunc = partial(CompGen.CompCompute, spec = myspec, ivar = myivar, z = myz, p = myp, z_edges = z_edges, n_chop = n_chop, outfile = outfile, boot = True)

	try:
		# Main composite creation
		CompGen.CompCompute(0, myspec, myivar, myz, myp, z_edges, n_chop, outfile)

		if nboot > 0:
			pool = mp.Pool()
			pool.map(myfunc, np.arange(nboot) + 1)

			pool.close()
			pool.join()

	except Exception as e:
		print e.__doc__
		print e.message
		os.chdir(mydir)

	# Return to the previous working directory
	os.chdir(mydir)

	return outfile