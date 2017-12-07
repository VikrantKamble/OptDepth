"""
Parallel processing of input spectra is carried out to minimize computation time.

alpha_lam = alpha_nu - 2
alpha_nu = - alpha_paris

Thin disc approximation gives alpha_nu = 1/3 
"""

import numpy as np
import traceback
from scipy import optimize as op
import matplotlib.pyplot as plt

def fit_alpha(S, zq, plotit = False, wav_range = [[1280, 1290], [1315, 1325], [1350, 1360], [1440, 1470]]):
    """ Estimate the spectral index 

    Parameters:
    -----------------------------------
    S: a container with wave, flux, inverse variance as its elements
    zq: the redshift of the object
    plotit: a flag to indicate if to plot
    wav_range: the wavelength chunks over which to perform fitting

    Returns:
    -----------------------------------
    [normalization constant, spectral_index], reduced chi-square, error on spectral index

    Throws:
    -----------------------------------
    ValueError if precision loss
    RuntimeError if number of function evaluations are exceeded
    """
    def alpha_func(x, theta):
    return theta[0] * (x/1450.) ** theta[1]

    def mychisq(theta, X, y, yivar):
    model = alpha_func(X, theta)
    return 0.5 * np.sum(yivar * (y - model) ** 2)

    wl, flux, ivar = *S
    redshift = zq

    # Wavelength chucks used for fitting
    wInd = np.array([])
    for w in range(len(wav_range)):
        wTemp = np.where((wl > wav_range[w][0]) & (wl < wav_range[w][1]))[0]
        wInd = np.concatenate((wInd, wTemp))
    wave = wl[wInd.astype(int)]

    spectra, invvar = flux[wInd.astype(int)], ivar[wInd.astype(int)]
   
    l_wave, l_spec, l_ivar = wave, spectra, invvar  # Changes on each iteration
    try:
        chisq_r0 = np.inf

        # fit iteratively removing points that are less than 
        # 3 sigma of the fit in each iteration
        for k in range(3):
            result = op.minimize(mychisq, [1, -1], args=(l_wave, l_spec, l_ivar));
            chisq_r = result['fun']/(len(l_spec) - 2)

            if chisq_r < chisq_r0:
                chisq_r0 = chisq_r

                # Remove 3 sigma one-sided points
                outlier = np.sqrt(invvar) * (spectra - alpha_func(wave, result['x']))  > -3
                l_wave, l_spec, l_ivar = wave[outlier], spectra[outlier], invvar[outlier]
            else:
                break
        if result['success'] == True:
            # using the asymptotic hessian-inverse provided by the fitting routine
            # as a proxy for the error on the spectral index
            alpha_sig = np.sqrt(result['hess_inv'][1, 1])
        else:
            raise ValueError
        
        if plotit :
            plt.figure(figsize=(15, 6))
            plt.plot(wl, flux, linewidth=0.4, color='k', label=r'$z = %.2f$' %redshift)
            plt.plot(wl, alpha_func(wl, result['x']), lw=0.4, color='r', label=r'$\alpha = %.2f, \chi^2_r = %.2f$' %(result['x'][1], 2 * chisq_r))
            plt.xlim(1040, 1800)
            plt.ylim(np.percentile(flux, 1), np.percentile(flux, 99))
            plt.legend()

            plt.plot(l_wave, l_spec, '*')
            plt.show()
            
        return result['x'], 2 * chisq_r, alpha_sig

    except:
        traceback.print_exc()
        return [0, 0], -1, 0


# linear fit to the continuum
def lin_funct(wave, a, b): return a*wave + b 

# model the CIV feature as a sum of two Gaussians?
def gauss_funct(x, *theta):
    A1, A2, S1, S2, C1, C2 = theta
    model = A1*np.exp(-(x - C1)**2 / (2*S1**2)) + A2*np.exp(-(x - C2)**2 / (2*S2**2))
    return model

# equivalent width definition
def f_ratio(x, *theta):
    model = gauss_funct(x, *theta[2:8]) / lin_funct(x, theta[0], theta[1])
    return model

def mychisq(theta, X, y, yinv):
    model = gauss_funct(X, *theta)
    return np.sum(yinv * (y - model) ** 2, axis=0)


def fit_EW(S, zq, plotit = False, lin_range=np.array([[1450, 1465], [1685, 1700]]), gauss_range=np.array([1500, 1580])):
    """ Fits a double gaussian profile to CIV line 

    Parameters:
    -----------------------------------
    S: a container with wavelength, flux and inverse variance as its elements
    zq: the redshift of the object
    plotit: a flag to toggle plotting
    lin_range: wavelength ranges over which to fit the local continuum
    gauss_range: wavelength range for the line fitting

    Returns:
    -----------------------------------
    equivalent width, fwhm, linear_fit params, gauss_fit params, reduced chi-square
    """
    wl, flux, invvar, redshift = *S, zq

    plotInd = np.where((wl > 1400) & (wl < 1700))[0]
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
