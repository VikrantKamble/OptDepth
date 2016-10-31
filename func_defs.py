import numpy as np
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel
from scipy.optimize import minimize
from scipy.integrate import quad

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

Omega_m, Omega_lam = 0.3, 0.7
invQ = lambda x: 1.0/np.sqrt(Omega_m*(1+x)**3 + Omega_lam)

def lum_dist(z):
	c, H0 = 2.99792 * 10**5, 70
	return (1+z)*(c/H0)*quad(invQ, 0, z)[0]

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