"""
1. MCMC analysis script to fit model to the COMPOSITES data
2. Feel free to add improved models 

3. Returns the complete chain - to be used for estimation or for GetDist
"""
#! /bin/python

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import emcee
import fitsio

def Gauss(x, amp, cen, scale):
	return amp * np.exp(-0.5 * (x - cen) ** 2 / scale**2)

def flux_model4(theta, wave, z_comp):
	tau0, gamma, f0, alpha = theta
	z_obs = (1 + z_comp) * wave / 1215.67 - 1

	model = f0 * (wave / 1450.) ** alpha  * np.exp(-tau0 * (1 + z_obs) ** gamma)
	return model

def lnprob4(theta, X, y, yivar):
	tau0, gamma, f0, alpha = theta
	
	if 1e-4 < tau0 < 1e-2 and 2 < gamma < 5 and 0.8 < f0 < 1.2 and -3 < alpha < -1:
		wave, z_comp = X.T
		model = flux_model4(theta, wave, z_comp)
		return -0.5 * np.sum(yivar * (y - model) ** 2)
	
	return -np.inf

def flux_model6(theta, wave, z_comp):
	tau0, gamma, f0, alpha, A1, S1 = theta
	z_obs = (1 + z_comp) * wave / 1215.67 - 1
		
	# 2 - Continuum + 2 - Optical depth
	C = f0 * (wave / 1450.) ** alpha + Gauss(wave, A1, 1123, S1)
	model =   C * np.exp(-tau0 * (1 + z_obs) ** gamma)
	return model

def lnprob6(theta, X, y, yivar):
	tau0, gamma, f0, alpha, A1, S1 = theta

	if 1e-4 < tau0 < 1e-2 and 2 < gamma < 5 and 0.8 < f0 < 1.2 and -3 < alpha < -1:
		wave, z_comp = X.T
		model = flux_model6(theta, wave, z_comp)
		return -0.5 * np.sum(yivar * (y - model) ** 2)

	return -np.inf

def flux_model8(theta, wave, z_comp):
	tau0, gamma, f0, alpha, A1, S1, A2, S2 = theta
	z_obs = (1 + z_comp) * wave / 1215.67 - 1
		
	# 2 - Continuum + 2 - Optical depth
	C = f0 * (wave / 1450.) ** alpha + Gauss(wave, A1, 1123, S1) + Gauss(wave, A2, 1069, S2)
	model =   C * np.exp(-tau0 * (1 + z_obs) ** gamma)
	return model

def lnprob8(theta, X, y, yivar):
	tau0, gamma, f0, alpha, A1, S1, A2, S2 = theta

	if 1e-4 < tau0 < 1e-2 and 2 < gamma < 5 and 0.8 < f0 < 1.2 and -3 < alpha < -1:
		wave, z_comp = X.T
		model = flux_model8(theta, wave, z_comp)
		return -0.5 * np.sum(yivar * (y - model) ** 2)

	return -np.inf

def flux_model10(theta, wave, z_comp):
	tau0, gamma, f0, alpha, A1, S1, A2, S2, A3, S3 = theta
	z_obs = (1 + z_comp) * wave / 1215.67 - 1
		
	# 2 - Continuum + 2 - Optical depth
	C = f0 * (wave / 1450.) ** alpha + Gauss(wave, A1, 1123, S1) + Gauss(wave, A2, 1069, S2) + Gauss(wave, A3, 1215.67, S3)
	model =   C * np.exp(-tau0 * (1 + z_obs) ** gamma)
	return model

def lnprob10(theta, X, y, yivar):
	tau0, gamma, f0, alpha, A1, S1, A2, S2, A3, S3 = theta

	if 1e-4 < tau0 < 1e-2 and 2 < gamma < 6 and 0.2 < f0 < 2 and -6 < alpha < 4:
		wave, z_comp = X.T
		model = flux_model10(theta, wave, z_comp)
		return -0.5 * np.sum(yivar * (y - model) ** 2)

	return -np.inf

def flux_model12(theta, wave, z_comp):
	tau0, gamma, f0, alpha, A1, S1, A2, S2, A3, S3, A4, S4 = theta
	z_obs = (1 + z_comp) * wave / 1215.67 - 1
		
	# 2 - Continuum + 2 - Optical depth
	C = f0 * (wave / 1450.) ** alpha + Gauss(wave, A1, 1123, S1) + Gauss(wave, A2, 1069, S2) + Gauss(wave, A3, 1215.67, S3) + Gauss(wave, A4, 1034,S4)
	model =   C * np.exp(-tau0 * (1 + z_obs) ** gamma)
	return model

def lnprob12(theta, X, y, yivar):
	tau0, gamma, f0, alpha, A1, S1, A2, S2, A3, S3, A4, S4 = theta

	if 1e-4 < tau0 < 1e-2 and 2 < gamma < 5 and 0.2 < f0 < 2 and -5 < alpha < 2:
		wave, z_comp = X.T
		model = flux_model12(theta, wave, z_comp)
		return -0.5 * np.sum(yivar * (y - model) ** 2)

	return -np.inf

def mcmcComp(myfile, niter, type, frange, input_guess):
	# READ IN THE FILE
	A = fitsio.FITS(myfile)

	flux, ivar = A[0].read(), A[1].read()
	redshift = A[2]['REDSHIFT'][:]
	wavelength = A[3]['WAVELENGTH'][:]

	base = flux[4]

	# FOREST REGION
	forest = np.where((wavelength > frange[0]) & (wavelength < frange[1]))[0]
	flux, ivar = flux[:, forest], ivar[:, forest]

	# We want to convert restframe wavelengths and quasar redshifts to a uniform grid 
	lam, zq = np.meshgrid(wavelength[forest], redshift)

	X = np.vstack((lam.ravel(), zq.ravel())).T
	y, y_ivar = np.ravel(flux), np.ravel(ivar)

	mask = y_ivar > 0

	# FINAL DATA_POINTS
	X, y, yivar = X[mask], y[mask], y_ivar[mask]
	print('Total number of pixels to train the model:', len(X))

	# MODEL FITTING
	np.random.seed()
	if type == 4:
		guess = [0.002, 3.7, 1.06, -1.3]

		# MIN-LOGLIKELIHOOD VALUE
		import scipy.optimize as op
		nll = lambda *args: -lnprob4(*args)
		result = op.minimize(nll, guess, args=(X, y, yivar), method='Powell')
		print(result['fun'] / (len(y) - 4))

		# Initialize around a tiny gasussian ball
		guess = result['x']

		nwalkers, ndim = 100, len(guess)
		p0 = [guess + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob4, args=(X, y, yivar))

		print('MCMC sampling started')
		start = timer()

		sampler.run_mcmc(p0, niter);
		print('MCMC sampling completed. Time=', timer() - start)

	if type == 6:
		guess = [0.002, 3.7, 1.06, -1.3, 0.15, 10]

		# MIN-LOGLIKELIHOOD VALUE
		import scipy.optimize as op
		nll = lambda *args: -lnprob6(*args)
		result = op.minimize(nll, guess, args=(X, y, yivar), method='Powell')
		print(result['fun'] / (len(y) - 6))

		# Initialize around a tiny gasussian ball
		guess = result['x']

		nwalkers, ndim = 100, len(guess)
		p0 = [guess + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob6, args=(X, y, yivar))

		print('MCMC sampling started')
		start = timer()

		sampler.run_mcmc(p0, niter);
		print('MCMC sampling completed. Time=', timer() - start)
		
	if type == 8:
		guess = [0.002, 3.7, 1.06, -1.3, 0.15, 10, 0.25, 12]

		# MIN-LOGLIKELIHOOD VALUE
		import scipy.optimize as op
		nll = lambda *args: -lnprob8(*args)
		result = op.minimize(nll, guess, args=(X, y, yivar), method='Powell')
		print(2 * result['fun'] / (len(y) - 8))

		# Initialize around a tiny gasussian ball
		guess = result['x']

		nwalkers, ndim = 100, len(guess)
		p0 = [guess + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob8, args=(X, y, yivar))

		print('MCMC sampling started')
		start = timer()

		sampler.run_mcmc(p0, niter);
		print('MCMC sampling completed. Time=', timer() - start)

	if type == 10:
		#guess = [0.002, 3.7, 0.9, alpha_guess, 0.15, 9.5, 0.23, 13.6, 1, 28]

		guess = input_guess

		# MIN-LOGLIKELIHOOD VALUE
		import scipy.optimize as op
		nll = lambda *args: -lnprob10(*args)
		result = op.minimize(nll, guess, args=(X, y, yivar), method='Powell')
		print(2 * result['fun'] / (len(y) - 10))

		# Initialize around a tiny gasussian ball
		#guess = result['x']

		nwalkers, ndim = 100, len(guess)
		p0 = [guess + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob10, args=(X, y, yivar))

		print('MCMC sampling started')
		start = timer()

		sampler.run_mcmc(p0, niter);
		print('MCMC sampling completed. Time=', timer() - start)

	if type == 12:
		guess = [0.002, 3.8, 0.86, -1.75, 0.17, 10.35, 0.27, 13.87, 1.08, 30.51, 0.8, 6]

		# MIN-LOGLIKELIHOOD VALUE
		import scipy.optimize as op
		nll = lambda *args: -lnprob12(*args)
		result = op.minimize(nll, guess, args=(X, y, yivar), method='Powell')
		print(2 * result['fun'] / (len(y) - 12))

		# Initialize around a tiny gasussian ball
		#guess = result['x']

		nwalkers, ndim = 100, len(guess)
		p0 = [guess + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob12, args=(X, y, yivar))

		print('MCMC sampling started')
		start = timer()

		sampler.run_mcmc(p0, niter);
		print('MCMC sampling completed. Time=', timer() - start)

	# Best-fit estimates for the parameters
	lInd = int(niter * 0.4)
	samps = sampler.chain[:, lInd:, :].reshape((-1, ndim))
	CenVal = np.median(samps, axis=0)
	print(CenVal)

	fig, ax1 = plt.subplots(1)

	# Plot continuum
	wl = wavelength[forest]
	cont = CenVal[2] * (wl/1450.)**CenVal[3]
	ax1.plot(wavelength, CenVal[2] * (wavelength/1450.)**CenVal[3])
	ax1.plot(wavelength, base)
	ax1.plot(wl, cont + Gauss(wl, CenVal[4], 1123, CenVal[5]))
	ax1.plot(wl, cont + Gauss(wl, CenVal[6], 1069, CenVal[7]))
	ax1.plot(wl, cont + Gauss(wl, CenVal[8], 1215.67, CenVal[9]))
	# ax1.plot(wl, cont + Gauss(wl, CenVal[10], 1034, CenVal[11]))

	# Plotting fit and residues
	for i in range(len(redshift))[::3]:
		# DATA
		ax1.plot(wavelength[forest], flux[i], '-k')
		
		if type==4:
			ax1.plot(wavelength[forest], flux_model4(CenVal, wavelength[forest], redshift[i]), '--r')
		elif type==6:
			ax1.plot(wavelength[forest], flux_model6(CenVal, wavelength[forest], redshift[i]), '--r')
		elif type==8:
			ax1.plot(wavelength[forest], flux_model8(CenVal, wavelength[forest], redshift[i]), '--r')
		elif type==10:
			ax1.plot(wavelength[forest], flux_model10(CenVal, wavelength[forest], redshift[i]), '--r')
		else:
			ax1.plot(wavelength[forest], flux_model12(CenVal, wavelength[forest], redshift[i]), '--r')
	plt.show()

	return sampler.chain