"""
The aim of this script is to estimate the spectral index of quasars using robust outlier rejection.
Parallel processing of input spectra is carried out to minimize computation time.
"""

# alpha_lam = alpha_nu - 2
# alpha_nu = - alpha_paris

# Thin disc approximation gives alpha_nu = 1/3 
import numpy as np
from scipy import optimize as op
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def alpha_func(x, theta):
    return theta[0] * (x/1450.) ** theta[1]

def mychisq(theta, X, y, yerr):
    model = alpha_func(X, theta)
    return np.sum((y - model) ** 2 / yerr ** 2, axis=0)

def fit_alpha(S, plotit = False, wav_range = [[1280, 1290], [1315, 1325], [1350, 1360], [1440, 1470]]):
    flux, ivar, redshift = S[0], S[1], S[2]

    # Wavelength chucks used for fitting
    wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)
    wInd = np.array([])

    for w in range(len(wav_range)):
        wTemp = np.where((wl > wav_range[w][0]) & (wl < wav_range[w][1]))[0]
        wInd = np.concatenate((wInd, wTemp))
    wave = wl[wInd.astype(int)]

    spectra, error = flux[wInd.astype(int)], 1.0 / np.sqrt(ivar[wInd.astype(int)])

    # Changes on each iteration
    l_wave, l_spec, l_error = wave, spectra, error
    try:
        chisq_r0 = np.inf

        for k in range(3):
            result = op.minimize(mychisq, [1, -1], args=(l_wave, l_spec, l_error));
            chisq_r = result['fun']/(len(l_spec) - 2)

            if chisq_r < chisq_r0:
                chisq_r0 = chisq_r

                # Remove 3 sigma one-sided points
                outlier = (spectra - alpha_func(wave, result['x'])) / error > -3
                l_wave, l_spec, l_error = wave[outlier], spectra[outlier], error[outlier]
            else:
                break

        if plotit :
            plt.figure(figsize=(15, 6))
            plt.plot(wl, flux, linewidth=0.4, color='k', label=r'$z = %.2f$' %redshift)
            plt.plot(wl, alpha_func(wl, result['x']), lw=0.4, color='r', label=r'$\alpha = %.2f, \chi^2_r = %.2f$' %(result['x'][1], chisq_r))
            plt.xlim(1040, 1800)
            plt.ylim(np.percentile(flux, 1), np.percentile(flux, 99))
            plt.legend()

            plt.plot(l_wave, l_spec, '*')
            plt.show()
            
        sigma = (chisq_r - 1) / np.sqrt(2. / (len(l_wave) - 2))
        return result['x'], chisq_r, sigma

    except (RuntimeError, ValueError, TypeError):
        return [0, 0], -1, 0

# Local implementation to find spectral index for the full sample using mp library
"""
from multiprocessing import Pool

pool = Pool()
res = pool.map(fit_alpha, zip(spec[0:1000], ivar[0:1000], z[0:1000]))

pool.close()
pool.join()

alpha_new = [ele[0] for ele in res]
chisq_new = [ele[1] for ele in res]

"""


