from __future__ import division
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel
from scipy.optimize import minimize
from scipy.integrate import quad
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

# HELPER FUNCTION DEFINATIONS 
def myfunct(wave, a , b):return a*(wave/1450)**b

# Linear model for local continuum
def lin_funct(wave, a , b):return a*wave + b

# Double gaussian to model the CIV line
def gauss_funct(x, *theta):
	A1, A2, S1, S2, C1, C2,  = theta
	return  A1*np.exp(-(x - C1)**2/S1**2) + A2*np.exp(-(x - C2)**2/S2**2)

# Helper function for calculating EW of CIV
def f_ratio(x, *theta):
	temp = theta[2:8]
	return gauss_funct(x, *temp)/lin_funct(x,theta[0],theta[1]) 
# -----------------------------------------------------------------------------------

# MAIN FUNCTION DEFINATIONS
def compute_alpha(wl, spec, ivar, wav_range, per_value, plotit=0):
		wavelength, spectra, invar = np.array([]), np.array([]), np.array([])

		for j in range(len(wav_range)):
			temp = np.where((wl > wav_range[j][0]) & (wl < wav_range[j][1]))[0]
			tempspec, tempivar  = spec[temp], ivar[temp]
			
			#Mask out metal absorption lines
			cut = np.percentile(tempspec, per_value)
			blah = np.where((tempspec > cut) & (tempivar > 0))[0]
			wave = wl[temp][blah]

			# Create vectors for model fitting
			wavelength = np.concatenate((wavelength, wave))
			spectra = np.concatenate((spectra, tempspec[blah]))
			invar = np.concatenate((invar, tempivar[blah]))
		
		try:
			popt, pcov = curve_fit(myfunct, wavelength, spectra, sigma=1.0/np.sqrt(invar))
		except (RuntimeError, TypeError):
			return 0, 0, -1, 0
		else:
			AMP, ALPHA = popt[0], popt[1]
			CHISQ = np.sum(invar * (spectra - myfunct(wavelength, popt[0], popt[1]))**2)
			# DOF = N - n  , n = 2
			DOF = len(spectra) - 2
			if (plotit == 1):
				plot(wl, spec, linewidth=0.4, alpha=0.7, color='black')
				plot(wl, myfunct(wl, popt[0], popt[1]), color='red')
			return AMP, ALPHA, CHISQ, DOF		

def compute_ew(wl, spec, ivar, lin_range, gauss_range, per_value, plotit=0):
	wavelength, spectra, invar = np.array([]), np.array([]), np.array([])
	for j in range(len(lin_range)):
		temp = np.where((wl > lin_range[j][0]) & (wl < lin_range[j][1]))[0]
		tempspec, tempivar  = spec[temp], ivar[temp]
		
		#Mask out metal absorption lines
		cut = np.percentile(tempspec, per_value)
		blah = np.where((tempspec > cut) & (tempivar > 0))[0]
		wave = wl[temp][blah]

		wavelength = np.concatenate((wavelength, wave))
		spectra = np.concatenate((spectra, tempspec[blah]))
		invar = np.concatenate((invar, tempivar[blah]))

	try:
		# Fit a linear local continuum
		popt = curve_fit(lin_funct, wavelength, spectra, sigma=1.0/np.sqrt(invar))[0]

		ind = np.where((wl > gauss_range[0]) & (wl < gauss_range[1]))[0]
		
		mywl, myivar = wl[ind], ivar[ind]
		myspec = spec[ind] - lin_funct(mywl, popt[0], popt[1])

		s = np.where(myivar > 0)[0]
	
		# Smoothing to remove narrow absoprtion troughs, more than 3 sigma away 
		smooth = convolve(myspec[s], Box1DKernel(20))
		t = s[np.where(np.sqrt(myivar[s])*(smooth - myspec[s]) < 3)[0]]

		W, S, I  = mywl[t], myspec[t], myivar[t]
	
		# Lambda function to specify the minimzation format
		err = lambda p: np.sum(I*(gauss_funct(W,*p) - S)**2) 

		# Initial guess for the parameters
		A1, A2 = max(smooth), max(smooth)/3.0
		C1 = C2 = mywl[s[np.argmax(smooth)]]
		S1 = np.sqrt(np.sum((mywl[s]-C1)**2 * smooth)/np.sum(smooth))
		S2 = S1*2.0

		# Initial guess for the optimiztion module
		p_init = [A1,A2,S1,S2,C1,C2]

		# Bounded optimization - Removes contimation from nearby lines especially He II
		bounds=[(0,None),(0,None),(5,30),(0,50),(1520,1580),(1520,1580)]

		chi_init = np.inf

		# k indicates the number of recursive iterations to perform
		for k in range(3):
			p_opt = minimize(err, p_init, bounds = bounds,	method="L-BFGS-B").x

			if (err(p_opt) < chi_init):
				chi_init = err(p_opt)
				dummy = np.sqrt(I)*(S - gauss_funct(W,*p_opt)) > -2
				W, S, I = W[dummy], S[dummy], I[dummy]
			else: break

	except (RuntimeError, ValueError, TypeError): 
		return -1, np.zeros(2), np.zeros(6), 0, 0
	else:
		lin_param, gauss_param = popt, p_opt
		chisq, dof = err(p_opt), len(S) - 6

		ew = quad(f_ratio, 1460, 1640, args = tuple([popt[0], popt[1]] + list(p_opt)))[0]

		if (plotit == 1):
			plot(wl, spec, linewidth=0.4, color='k')
			plot(wl, lin_funct(wl, popt[0], popt[1]) + gauss_funct(wl, *p_opt))
		return ew, lin_param, gauss_param, chisq, dof

#------------------------------------------------------------------------------------	
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