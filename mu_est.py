"""
1. Helper scipt to carry out likelihood estimation of the location parameter at a given pixel
2. Uses brute-force evaluation - still sufficient
3. The output sigma should not be interpreted to be solely arising from intrinsic lss variance
"""

#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from astroML.stats import mean_sigma
from scipy.optimize import curve_fit

# # Function definations
def logG(xi, ei, mu, sigma):
	extent = np.broadcast(mu, sigma).shape

	xi = xi.reshape(xi.shape + tuple([1 for e in extent]))
	ei = ei.reshape(ei.shape + tuple([1 for e in extent]))

	return -0.5 * np.sum(np.log(sigma ** 2 + ei ** 2) + (xi - mu) ** 2 / (sigma ** 2 + ei ** 2), 0)

def gaussian(x, amp, cen, wid):
	return amp * np.exp(- 0.5 * (x - cen)**2 / wid**2)

def param_est(xi, ei, plot_marginal=False):
	try:
		# Suitable range over with to obtain posterior 
		loc, scale = mean_sigma(xi)
		loc_scale = scale / np.sqrt(len(xi))

		# Define parameter ranges
		mu = np.linspace(loc - 4 * loc_scale, loc + 4 * loc_scale, 200)
		sigma = np.linspace(0, 0.6 * loc, 200)

		# Obtain the posterior likelihood
		logL = logG(xi, ei, mu, sigma[:, np.newaxis])
		logL -= logL.max()

		# Marginalized results
		L = np.exp(logL)

		p_mu = L.sum(0)
		p_mu /= (mu[1] - mu[0]) * p_mu.sum()

		p_sigma = L.sum(1)
		p_sigma /= (sigma[1] - sigma[0]) * p_sigma.sum()

		# Fit with gaussian
		mu_popt = curve_fit(gaussian, mu, p_mu, p0=(1, loc, loc_scale))[0]
		sigma_popt = curve_fit(gaussian, sigma, p_sigma)[0]

		mu_range = [mu_popt[1], mu_popt[1] - mu_popt[2], mu_popt[1] + mu_popt[2]]
		sigma_range = [sigma_popt[1], sigma_popt[1] - sigma_popt[2], sigma_popt[1] + sigma_popt[2]]

		if plot_marginal:
			# Plot the results
			fig = plt.figure(figsize=(12,6))
			fig.subplots_adjust(left=0.1, right=0.95, wspace=0.24, bottom=0.15, top=0.9)

			ax = fig.add_subplot(121)
			ax.plot(mu, p_mu, '-k')

			for s in mu_range:
				ax.axvline(s)

			ax.set_xlabel(r'$\mu$')
			ax.set_ylabel(r'$p(\mu)$')

			ax = fig.add_subplot(122)
			ax.plot(sigma, p_sigma, '-k')

			for s in sigma_range:
				ax.axvline(s)

			ax.set_xlabel(r'$\sigma$')
			ax.set_ylabel(r'$p(\sigma)$')

			plt.legend()
			plt.show()

		return mu_popt[1], mu_popt[2] , sigma_popt[1], sigma_popt[2]

	except RuntimeError:
		return tuple(np.zeros(4))
