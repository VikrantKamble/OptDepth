from __future__ import division
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel
from scipy.optimize import minimize
from scipy.integrate import quad
from func_defs import *
import numpy as np
import fitsio
import timeit
import os
import json

#Load  in the data from the FITS file - BIG!!!
A = fitsio.FITS(cat_file, 'rw')
spec, ivar, tb = A[0].read(), A[1].read(), A[2].read()
z = tb[drq_z]

# Read in local configuration parameters
X = ConfigMap('Estimate')
if (X == 'FAIL'):sys.exit("Errors! Couldn't locate section in .ini file")

# Read ranges as lists from config.ini
lin_range = json.loads(X['lin_range'])
gauss_range = json.loads(X['gauss_range'])
wav_range = json.loads(X['wav_range'])
per_value = float(X['per_value'])

nObj = len(z)

	
# 1. Calculation of the spectral index
if X['alpha_flag'] == 'Yes':
	ver = X['alpha_ver']
	AMP, ALPHA, CHISQ, DOF = np.zeros((4, nObj))

	start_time = timeit.default_timer()

	for i in range(nObj):
		AMP[i], ALPHA[i], CHISQ[i], DOF[i] = compute_alpha(wl, spec[i], ivar[i], wav_range, per_value)

	A[-1].insert_column('AMP_' + ver, AMP)
	A[-1].insert_column('ALPHA_' + ver, ALPHA)
	A[-1].insert_column('CHISQ_ALPHA_' + ver, CHISQ)
	A[-1].insert_column('DOF_ALPHA_' + ver, DOF)

	print(timeit.default_timer() - start_time)
	print "Writing of alpha, dof and chisq complete!"
# -----------------------------------------------------------------------------------

# 2. Calculation of EW of CIV
if X['ew_flag'] == 'Yes':
	ver = X['ew_ver']

	EW, CHISQ, DOF = np.zeros((3, nObj))
	LIN_PARAM, GAUSS_PARAM = np.zeros((nObj, 2)), np.zeros((nObj,6))

	start_time = timeit.default_timer()
	for i in range(nObj):
		EW[i], LIN_PARAM[i], GAUSS_PARAM[i], CHISQ[i], DOF[i] = compute_ew(wl, spec[i], ivar[i], lin_range, gauss_range, per_value)

	A[-1].insert_column('EW_' + ver, EW)
	A[-1].insert_column('CHISQ_EW_' + ver, CHISQ)
	A[-1].insert_column('DOF_EW_' + ver, DOF)
	A[-1].insert_column('LIN_PARAM_' + ver, LIN_PARAM)
	A[-1].insert_column('GAUSS_PARAM_' + ver, GAUSS_PARAM)

	print(timeit.default_timer() - start_time)
	print "Writing of EW complete"

# 3. Calculation of Luminosty
Omega_m, Omega_lam = 0.3, 0.7
invQ = lambda x: 1.0/np.sqrt(Omega_m*(1+x)**3 + Omega_lam)

def lum_dist(z):
	c, H0 = 2.99792 * 10**5, 70
	return (1+z)*(c/H0)*quad(invQ, 0, z)[0]

if X['lum_flag'] == 'Yes':
	ver = X['lum_ver']

	# Bolometric corrections applied
	lum_ref_wav, lum_ref_bc = float(X['lum_ref_wav']), float(X['lum_ref_bc'])

	LBOL = np.zeros(nObj)
	
	start_time = timeit.default_timer()

	bolInd = np.where((wl > (lum_ref_wav - 5)) & (wl < (lum_ref_wav + 5)))[0]

	# Flux in units of 10**-17 ergs/s/ang/cm2 and conversion of Mpc2 to cm2
	offset = np.log10((10**6 * 3.086 * 10**18)**2 * 10**-17)

	for i in range(nObj):
		lum = lum_ref_bc * np.median(spec[i][bolInd]) * 4 * np.pi * lum_dist(z[i])**2
		if lum > 0:LBOL[i] = np.log10(lum) + offset

	A[-1].insert_column('LBOL_' + ver, LBOL)

	print(timeit.default_timer() - start_time)
	print "Writing of bolometric luminosity complete!"
# -----------------------------------------------------------------------------------
# 4. CIV_FWHM - currently from Paris et. al