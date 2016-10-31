import numpy as np
import config_read as cfg
import matplotlib.pyplot as plt
"""
STEPS:
1. Collect all the rest-frame wavelength pixels
2. Convert them to observer frame using redshift
3. Collect the corresponding flux, invar and weights vector
4. Normalize everything to the same base by using median in a narrow observed wavelength range common to all
4. Do a simple binning statistic(weighted mean, weighted median)
"""

def calibrate(spec, ivar, z_q, w_hist, rest_range, norm_min, norm_max):

	rest_range_ind = np.array([])

	for j in range(len(rest_range)):
		foo = np.where((cfg.wl > rest_range[j][0]) & (cfg.wl < rest_range[j][1]))[0]
		rest_range_ind = np.concatenate((rest_range_ind, foo))

	rest_range_ind = rest_range_ind.astype(int)
	# ----------------------------------------------------------------------------

	obs_lam = np.array(np.mat(cfg.wl[rest_range_ind]).T * np.mat(1 + z_q)).T # observed wavelength

	flux, ivar = spec[:,rest_range_ind], ivar[:,rest_range_ind]

	# ----------------------------------------------------------------------------
	norm_value = np.zeros(len(rest_range_ind))

	for i in range(len(rest_range_ind)):
		foo = np.where((obs_lam[:,i] > norm_min) & (obs_lam[:,i] < norm_max))[0]
		norm_value[i] = np.median(flux[:,i][foo])

	# scale the flux by the normalization
	tot_flux = flux / norm_value

	# scale the weights correspondingly
	tot_weights = (ivar * w_hist[:,None]) * norm_value**2

	# ----------------------------------------------------------------------------

	rObs_lam, rFlux, rWeights = np.ravel(obs_lam), np.ravel(tot_flux), np.ravel(tot_weights)

	obs_lam_divs = np.arange(min(rObs_lam), max(rObs_lam), 4)

	BinFValue = np.zeros(len(obs_lam_divs)-1)

	for k in range(len(obs_lam_divs) - 1):
		BinInd = np.where((rObs_lam > obs_lam_divs[k]) & (rObs_lam < obs_lam_divs[k+1]))[0]
		zeros_ind = np.where((rWeights[BinInd] > 0) & (rFlux[BinInd] > 0))[0]

		if (len(zeros_ind) > 0):
			BinFValue[k] = np.average(rFlux[BinInd][zeros_ind], weights=rWeights[BinInd][zeros_ind])
			#BinFValue[k] = np.median(rFlux[BinInd][zeros_ind])

	BinZValue = (obs_lam_divs[1:] + obs_lam_divs[:-1]) / 2.

	plt.plot(BinZValue, BinFValue)
	
	# save to a file 