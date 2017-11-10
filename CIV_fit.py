"""
- Estimate equivalent width of C IV 
- Model as a sum of two Gaussians
- MP implementation for faster computation

NOTE: THIS CODE HAS BEEN HEABILY OPTIMZED AND TUNED TO WORK WITH C IV LINE OF BOSS QUASARS
- DO NOT ATTEMPT TO TRIVIALLY MODIFY IT FOR FITTING OF ANY OTHER LINE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel
from scipy import optimize as op

from scipy.integrate import quad
from multiprocessing import Pool

wl = 10**(2.73 + np.arange(8140) * 10 ** -4)

# We are going to model the local continuum by a linear fit
def lin_funct(wave, a, b): return a*wave + b 

# Model the CIV feature as a sum of two Gaussians?
def gauss_funct(x, *theta):
	A1, A2, S1, S2, C1, C2 = theta
	model = A1*np.exp(-(x - C1)**2 / (2*S1**2)) + A2*np.exp(-(x - C2)**2 / (2*S2**2))
	return model

def f_ratio(x, *theta):
	model = gauss_funct(x, *theta[2:8]) / lin_funct(x, theta[0], theta[1])
	return model

def mychisq(theta, X, y, yinv):
	model = gauss_funct(X, *theta)
	return np.sum(yinv * (y - model) ** 2, axis=0)


plotInd = np.where((wl > 1400) & (wl < 1700))[0]

# Ranges used for fitting
lin_range = [[1450, 1465], [1685, 1700]]
gauss_range = [1500, 1580]

def fit_EW(S, plotit = False):
	flux, invvar, redshift = S[0], S[1], S[2]

	# Linear Local continuum Fit

	wave, spectra, error = np.array([]), np.array([]), np.array([])

	for j in range(2):
		foo = np.where((wl > lin_range[j][0]) & (wl < lin_range[j][1]))[0]

		# Remove one-sided 3 percentile points to mitigate narrow absorption lines
		blah = np.where(flux[foo] > np.percentile(flux[foo], 3))[0]

		wave = np.hstack((wave, wl[foo][blah]))
		spectra = np.hstack((spectra, flux[foo][blah]))
		error = np.hstack((error, 1.0/np.sqrt(invvar[foo][blah])))

	try:
		popt, pcov = curve_fit(lin_funct, wave, spectra, sigma=error)

		# Double Gaussian Fit
		ind = np.where((wl > gauss_range[0]) & (wl < gauss_range[1]))

		l_wave, l_ivar = wl[ind], invvar[ind]
		l_spec = flux[ind] - lin_funct(l_wave, popt[0], popt[1])

		# Select point that provide information
		s = np.where(l_ivar > 0)[0]

		# Remove absorption troughs more than 5 sigma away
		smooth = convolve(l_spec[s], Box1DKernel(30))
		t = np.where((smooth - l_spec[s]) * np.sqrt(l_ivar[s]) < 5)[0]

		# Final vectors used for fitting
		W, S, I = l_wave[s][t], l_spec[s][t], l_ivar[s][t]

		# INITIAL GUESSES
		A1, A2 = max(smooth), max(smooth)/3.0
		C1 = C2 = l_wave[s][np.argmax(smooth)]
		S1 = np.sqrt(np.sum(np.abs(smooth[t]) * (C1 - W)**2) / np.sum(np.abs(smooth[t])))
		S2 = 2 * S1

		p_init = [A1, A2, S1, S2, C1, C2]

		# Bounded optimization limits - Removes contamination from He II
		bounds = [(0, 5*A1), (0, 3*A1), (4, 30), (5, 50), (1530, 1570), (1530, 1570)]

		# Changes on each iteratin
		fit_W, fit_S, fit_I = W, S, I
		chisq_r0, p0 = np.inf, np.zeros(6)

		for k in range(5):
			# Minimize the chi-square function
			result = op.minimize(mychisq, p_init, args=(fit_W, fit_S, fit_I), method="L-BFGS-B", bounds=bounds)
			chisq_r = result['fun'] / (len(fit_W) - 6)

			if chisq_r < chisq_r0:
				chisq_r0, p0 = chisq_r, result['x']

				# Remove points that are 3 sigma away from the fit
				outlier = (S - gauss_funct(W, *result['x'])) * np.sqrt(I) > -3
				fit_W, fit_S, fit_I = W[outlier], S[outlier], I[outlier]
			else:
				break

		# Calculate equivalent width using flux ratios
		ew = quad(f_ratio, 1460, 1640, args = tuple([popt[0], popt[1]] + list(p0)))[0]

		# Calculate FWHM - sum of red HWHM and blue HWHM
		loc = op.minimize(lambda x: -gauss_funct(x, *p0), 1550).x[0]

		r_loc = op.minimize(lambda x: np.abs(gauss_funct(x, *p0) - gauss_funct(loc, *p0)/2.), loc + 2. , method='COBYLA', bounds = (loc, None)).x 
		b_loc = op.minimize(lambda x: np.abs(gauss_funct(x, *p0) - gauss_funct(loc, *p0)/2.), loc - 2. , method='COBYLA', bounds = (None, loc-0.001)).x 

		fwhm = r_loc - b_loc

		if plotit == True:
			plt.figure()
			plt.plot(wl[plotInd], flux[plotInd], alpha=0.6, color='gray', label=r'$z = %.2f$' %redshift)
			plt.plot(wl[plotInd], lin_funct(wl[plotInd], popt[0], popt[1]) + gauss_funct(wl[plotInd], *p0), color='r', label=r'$EW = %.2f, FWHM = %.2f, \chi^2_r = %.2f$' %(ew, fwhm, chisq_r0))
			plt.legend(loc = 'lower right')

			plt.plot(wl[plotInd], lin_funct(wl[plotInd], popt[0], popt[1]) + p0[0] * np.exp(-(wl[plotInd] - p0[4])**2 / (2 * p0[2] ** 2)), color='g')
			plt.plot(wl[plotInd], lin_funct(wl[plotInd], popt[0], popt[1]) + p0[1] * np.exp(-(wl[plotInd] - p0[5])**2 / (2 * p0[3] ** 2)), color='b')

			plt.axvline(r_loc, color='k')
			plt.axvline(b_loc, color='k')
			plt.axvline(loc, color='k')
			plt.axvline(1549.48, linestyle='dashed', color='k')
			plt.show()

		return ew, fwhm, popt, p_init, chisq_r0

	except(RuntimeError, ValueError, TypeError):
		return -1, -1,  np.zeros(2), np.zeros(6), -1

# Local implementation to find CIV line parameters for the full sample using mp library
def process_all(spec, ivar, z):
     from multiprocessing import Pool

     pool = Pool()
     res = pool.map(fit_EW, zip(spec, ivar, z), chunksize=5)

     pool.close()
     pool.join()
     return res
