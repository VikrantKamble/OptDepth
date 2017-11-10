import numpy as np
import emcee
import matplotlib.pyplot as plt

import scipy.optimize as op
from getdist import plots, MCSamples
from astroML.plotting.mcmc import convert_to_stdev

## LIKELIHOOD DEFINATIONS
# Model with constant LSS variance across redshift
def lnlike1(theta, xi, yi, ei):
	f0, t0, gamma, lss_sigma = theta

	# Constant prior on lss_sigma in log-space
	if 0 <= f0 < 5 and -11 <= t0 < -2 and 1 <= gamma < 7 and 0 < lss_sigma < 0.6:
		model = f0 * np.exp(-np.exp(t0) * (1 + xi) ** gamma)
		return -0.5 * np.sum((yi - model) ** 2 / (ei ** 2 + lss_sigma ** 2) + np.log(ei ** 2 + lss_sigma ** 2), 0)

	return -np.inf

# Model with LSS variance parameterized by a power law as a function of redshift
def lnlike2(theta, xi, yi, yivar):
	f0, t0, gamma = theta

	# Model for transmission
	model = f0 * np.exp(-t0 * (1 + xi) ** gamma)

	return -0.5 * np.sum((yi - model) ** 2 * yivar)


# -----------------------------------------------------------------------------

# Helper function for minimization using op.minimize
nll1 = lambda *args: -lnlike1(*args)
nll2 = lambda *args: -lnlike2(*args)

# Initial guess
guess1 = [1.5, -8, 4.5, 0.1]
guess2 = [1.5, 0.0017, 3.8, 0.1, 0]

# Configure GetDist
names = ["f0", "t0", "gamma", "sigma"]
labels = ["f_0", r"\tau_0", "\gamma", "\sigma"]

def mcmcSkewer(bundleObj, logdef=1, niter=1500, do_mcmc=True, plotit=False, return_sampler=False, triangle=False, evalgrid=True, visualize=False):
	"""
	Script to fit simple flux model on each restframe wavelength skewer

	Parameters:
	bundleObj: A list of [z, f, ivar] with the skewer_index 
	logdef: Which model to use (lnlike1: constant variance, lnlike2: power-law variance)
	niter: The number of iterations to run the mcmc (500 for burn-in fixed)
	do_mcmc: Flag whether to perform mcmc
	plotit: Plot the data along with best fit from scipy and mcmc
	return_sampler: Whether to return the raw sampler results without flatchaining
	triangle: Display triangle plot of the parameters
	evalgrid: Whether to compute loglikelihood on a specified grid

	"""
	z, f, ivar = bundleObj[0].T

	ind = np.where(ivar > 0)[0]
	z, f, sigma = z[ind], f[ind], 1.0 / np.sqrt(ivar[ind])

	if plotit:
		fig, ax1 = plt.subplots(1)
		ax1.errorbar(z, f, sigma, fmt='o', color='gray')
		
	if logdef == 1:
		result = op.minimize(nll1, guess1, args=(z, f, sigma), method='Nelder-mead')
		zline = np.linspace(2.2, 4.2, 100)
		
		if plotit:
			ax1.plot(zline, result['x'][0] * np.exp(-np.exp(result['x'][1]) * (1 + zline) ** result['x'][2]), '-r')

		print(result['success'], result['x'])

		if do_mcmc:
			np.random.seed()
			nwalkers, ndim = 100, 4
			p0 = [guess1 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

			# Configure the sampler
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike1, args=(z, f, sigma))

			# Burn-in time - Is this enough?
			p0,_,_ = sampler.run_mcmc(p0, 500);
			sampler.reset()
			print('Burning step completed')

			sampler.run_mcmc(p0, 1500);

			if return_sampler:
				return sampler.chain
			else:
				lInd = int(niter * 0.4)
				samps = sampler.chain[:, lInd:, :].reshape((-1, ndim))
				CenVal = np.median(samps, axis=0)

				estimates = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samps, [16, 50, 84], axis=0))))

				for count, ele in enumerate(names):
					print(ele + ' = %.3f^{%.3f}_{%.3f}' %(estimates[count][0], estimates[count][1], estimates[count][2]))

				if plotit:
					ax1.plot(zline, CenVal[0] * np.exp(-np.exp(CenVal[1]) * (1 + zline) ** CenVal[2]), '-g')

				# Instantiate a getdist object 
				MC = MCSamples(samples=samps, names=names, labels=labels, ranges={'f0':(0, 5), 't0':(-11, -2), 'gamma':(1, 7), 'lss_sigma':(0, 0.6)})

				if triangle:
					g = plots.getSubplotPlotter()
					g.triangle_plot(MC)

				if evalgrid:
					pdist = MC.get2DDensity('t0', 'gamma')

					# Create a grid
					tau_grid, gamma_grid = np.mgrid[-11:-2:200j, 1:7:200j]
					positions = np.vstack([tau_grid.ravel(), gamma_grid.ravel()]).T

					# Evalaute density on a grid
					pgrid = np.array([pdist.Prob(*ele) for ele in positions])
					# Prune to remove negative densities
					pgrid[pgrid < 0] = 0

					# Convert to logLikelihood
					logP = np.log(pgrid)
					logP -= logP.max()
					logP = logP.reshape(tau_grid.shape)

					# Visualize
					if visualize:
						plt.figure(figsize=(5,5))
						plt.contour(tau_grid, gamma_grid, convert_to_stdev(logP), levels=[0.683, 0.95], alpha=0.5, colors='k')
						plt.show()

					# Save to disk
					fileName1 = 'gridlnlike_' + str(bundleObj[1]) + '.dat'
					np.savetxt(fileName1, logP)
					fileName2 = 'estimates_' + str(bundleObj[1]) + '.dat'
					np.savetxt(fileName2, estimates)

# EOF

