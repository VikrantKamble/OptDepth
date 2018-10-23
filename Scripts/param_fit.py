# -*- coding: utf-8 -*-

"""
Parallel processing of input spectra is carried out to minimize
computation time.

alpha_lam = alpha_nu - 2
alpha_nu = - alpha_paris

Thin disc approximation gives alpha_nu = 1/3
"""

import numpy as np
import matplotlib.pyplot as plt
import traceback
from scipy import optimize as op
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel
from scipy.integrate import quad


wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)
alpha_fit_ranges = [[1280, 1290], [1317, 1325], [1350, 1360], [1440, 1470]]


def fit_alpha(flux, ivar, redshift=-1, wl=wl, wav_range=alpha_fit_ranges,
              plotit=False, ax=None):
    """ Estimate the spectral index

    Parameters --
    flux     : flux vector
    ivar     : precision vector
    redshift : redshift of the object (optional), used only while plotting
    wl       : wavelength vector
    wav_range: the wavelength chunks over which to perform fitting
    plotit   : a flag to indicate if to plot

    Returns --
    normalization constant, spectral_index, reduced chi-square
    and error on spectral index

    Throws --
    ValueError if precision loss
    RuntimeError if number of function evaluations are exceeded
    """
    def alpha_func(x, theta):
        return theta[0] * (x / 1450.) ** theta[1]

    def mychisq(theta, X, y, yivar):
        model = alpha_func(X, theta)
        return 0.5 * np.sum(yivar * (y - model) ** 2)

    # Wavelength chucks used for fitting
    wInd = np.array([])
    for w in range(len(wav_range)):
        wTemp = np.where((wl > wav_range[w][0]) & (wl < wav_range[w][1]))[0]
        mask = ivar[wTemp] > 0
        # Do analysis only if 20 pixels in each chunck available
        if mask.sum() < 10:
            return np.nan, np.nan, -1, np.nan
        wInd = np.concatenate((wInd, wTemp[mask]))
    wave = wl[wInd.astype(int)]

    spectra, invvar = flux[wInd.astype(int)], ivar[wInd.astype(int)]

    # Changes on each iteration
    l_wave, l_spec, l_ivar = wave, spectra, invvar
    try:
        chisq_r0 = np.inf

        # fit iteratively removing points that are less than 3 sigma
        for k in range(3):
            result = op.minimize(mychisq, [np.mean(l_spec), -1],
                                 args=(l_wave, l_spec, l_ivar),
                                 method='Nelder-Mead')
            chisq_r = result['fun'] / (len(l_spec) - 2)

            if chisq_r < chisq_r0:
                chisq_r0 = chisq_r

                outlier = np.sqrt(invvar) * (spectra - alpha_func(wave, result['x'])) > -3
                l_wave, l_spec, l_ivar = wave[outlier], spectra[outlier], invvar[outlier]
            else:
                break

        if result['success']:
            # using the analytic hessian
            temp = - np.sum(l_ivar * (np.log(l_wave / 1450.) ** 2 *
                            alpha_func(l_wave, result['x']) * (l_spec - 2 *
                            alpha_func(l_wave, result['x'])
                    )))
            alpha_sig = 1. / np.sqrt(temp)
        else:
            raise ValueError

        if plotit:
            if ax is None:
                ax = plt.gca()
            ax.plot(wl, flux, linewidth=0.4, color='gray', label=r'$z = %.2f$' % redshift)
            ax.plot(wl, alpha_func(wl, result['x']), lw=0.6, c='r',
                    label=r'$\alpha = %.2f, \chi^2_r = %.2f, \sigma_\alpha = %.2f$' %(result['x'][1], 2 * chisq_r, alpha_sig))
            ax.set_xlim(1040, 1800)
            ax.set_ylim(np.percentile(flux, 1), np.percentile(flux, 99))
            plt.legend()

            plt.plot(l_wave, l_spec, '+')
            plt.show()

        return result['x'][0], result['x'][1], 2 * chisq_r, alpha_sig

    except (RuntimeError, ValueError, TypeError):
        traceback.print_exc()
        return np.nan, np.nan, -1, np.nan


# linear fit to the continuum
def lin_funct(wave, a, b): return a*wave + b


# model the CIV feature as a sum of two Gaussians?
def gauss_funct(x, *theta):
    A1, A2, S1, S2, C1, C2 = theta
    model = A1 * np.exp(-0.5 * (x - C1) ** 2 / S1 ** 2) +\
        A2 * np.exp(-0.5 * (x - C2) ** 2 / S2 ** 2)
    return model


# equivalent width definition
def f_ratio(x, *theta):
    model = gauss_funct(x, *theta[2:8]) / lin_funct(x, theta[0], theta[1])
    return model


def mychisq(theta, X, y, yinv):
    model = gauss_funct(X, *theta)
    return np.sum(yinv * (y - model) ** 2, axis=0)


def fit_EW(flux, ivar, redshift=-1, wl=wl, plotit=False,
           lin_range=np.array([[1425, 1470], [1685, 1710]]),
           gauss_range=np.array([1500, 1580]), ax=None):
    """ Fits a double gaussian profile to CIV line

    Parameters --
    zq: the redshift of the object m
    lin_range: wavelength ranges over which to fit the local continuum
    gauss_range: wavelength range for the line fitting

    Returns --gau
    equivalent width, fwhm, linear_fit params, gauss_fit params,
    reduced chi-square
    """
    plotInd = np.where((wl > 1430) & (wl < 1700))[0]
    wave, spectra, error = np.array([]), np.array([]), np.array([])

    for j in range(2):
        foo = np.where((wl > lin_range[j][0]) & (wl < lin_range[j][1]))[0]

        # Remove one-sided 3 percentile points to mitigate
        # narrow absorption lines
        blah = np.where(flux[foo] > np.percentile(flux[foo], 5))[0]

        wave = np.hstack((wave, wl[foo][blah]))
        spectra = np.hstack((spectra, flux[foo][blah]))
        error = np.hstack((error, 1.0/np.sqrt(ivar[foo][blah])))

    # select pixels that provide information for linear function
        indices = np.isfinite(error)
        wave, spectra, error = wave[indices], spectra[indices], error[indices]
    try:
        popt, pcov = curve_fit(lin_funct, wave, spectra, sigma=error)

        ind = np.where((wl > gauss_range[0]) & (wl < gauss_range[1]))

        l_wave, l_ivar = wl[ind], ivar[ind]
        l_spec = flux[ind] - lin_funct(l_wave, popt[0], popt[1])

        # Select point that provide information for Gaussian fit
        s = np.where(l_ivar > 0)[0]

        # Remove absorption troughs more than 3 sigma away
        smooth = convolve(l_spec[s], Box1DKernel(30))
        t = np.where((smooth - l_spec[s]) * np.sqrt(l_ivar[s]) < 3)[0]

        # Final vectors used for fitting
        W, S, I = l_wave[s][t], l_spec[s][t], l_ivar[s][t]

        # Initial guess for minimization routine
        A1, A2 = max(smooth), max(smooth) / 3.0
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
            result = op.minimize(mychisq, p_init, args=(fit_W, fit_S, fit_I),
                                 method="L-BFGS-B", bounds=bounds)
            chisq_r = result['fun'] / (len(fit_W) - len(p_init))

            if chisq_r < chisq_r0:
                chisq_r0, p0 = chisq_r, result['x']

                # Remove points that are 3 sigma away from the fit
                outlier = (S - gauss_funct(W, *result['x'])) * np.sqrt(I) > -3
                fit_W, fit_S, fit_I = W[outlier], S[outlier], I[outlier]
            else:
                break

        # Calculate equivalent width using flux ratios
        ew = quad(f_ratio, 1460, 1640,
                  args=tuple([popt[0], popt[1]] + list(p0)))[0]

        # Calculate FWHM - sum of red HWHM and blue HWHM
        loc = op.minimize(lambda x: -gauss_funct(x, *p0), 1550).x[0]

        r_loc = op.minimize(lambda x: np.abs(gauss_funct(x, *p0) -
                            gauss_funct(loc, *p0)/2.), loc + 2.,
                            method='COBYLA').x
        b_loc = op.minimize(lambda x: np.abs(gauss_funct(x, *p0) -
                            gauss_funct(loc, *p0)/2.), loc - 2.,
                            method='COBYLA').x
        fwhm = r_loc - b_loc

        if plotit:
            if ax is None:
                fig, ax = plt.subplots(1)

            ax.plot(wl[plotInd], flux[plotInd], lw=0.5, alpha=0.6,
                    color='k', label=r'$z = %.2f$' % redshift)
            ax.plot(wl[plotInd], lin_funct(wl[plotInd], popt[0], popt[1]) +
                    p0[0] * np.exp(-(wl[plotInd] - p0[4])**2 / (2 * p0[2] ** 2)),
                    color='orange', label='component 1')
            ax.plot(wl[plotInd], lin_funct(wl[plotInd], popt[0], popt[1]) +
                    p0[1] * np.exp(-(wl[plotInd] - p0[5])**2 / (2 * p0[3] ** 2)),
                    color='orange', label='component 2')
            ax.plot(wl[plotInd], lin_funct(wl[plotInd], popt[0], popt[1]) +
                    gauss_funct(wl[plotInd], *p0), color='b',
                    label=r'$EW = %.2f, FWHM = %.2f, \chi^2_r = %.2f$' % (ew, fwhm, chisq_r0))

            # ax.axvline(r_loc, color='k')
            # ax.axvline(b_loc, color='k')
            ax.axvline(loc, color='k')
            ax.axvline(1549.48, linestyle='dashed', color='k')
            ax.set_xlabel(r'$\lambda_{\mathrm{rf}}$')
            ax.set_ylabel(r'$f_\lambda$')

            plt.show()

        return ew, fwhm, popt, p_init, chisq_r0

    except(RuntimeError, ValueError, TypeError):
        traceback.print_exc()
        return -1, -1,  np.zeros(2), np.zeros(6), -1

# Local implementation to find CIV line parameters for the full sample
# using mp library


def _g(theta):
    return fit_alpha(*theta)


def _h(theta):
    return fit_EW(*theta)


def process_all(flux, ivar, param='alpha'):
    from multiprocessing import Pool

    pool = Pool()
    if param == 'alpha':
        res = pool.map(_g, zip(flux, ivar), chunksize=5)
    elif param == 'EW':
        res = pool.map(_h, zip(flux, ivar), chunksize=5)
    else:
        pass

    pool.close()
    pool.join()
    return res
