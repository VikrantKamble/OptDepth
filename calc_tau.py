from __future__ import division
from scipy.optimize import curve_fit
import numpy as np
import config_read as cfg
import fitsio

def tau_model(x,a,b,c):
	return a*(1+x)**b - c

def calc_tau(infile, ly_range, ly_line, zn=3.0, zset=False, zdiv=50, trim_obs_lam = 0):
	# Read in the input fits file
	A = fitsio.FITS(infile)
	flux, invar, z = A[0].read(), A[1].read(), A[2]['REDSHIFT'][:]
	A.close()

	lyInd = np.where((cfg.wl > ly_range[0]) & (cfg.wl < ly_range[1]))[0]

	# The redshift of the resonant scattering layer
	ZR = np.array(((np.mat(cfg.wl[lyInd]).T * np.mat(1+z))/ly_line - 1).T)

	F, S = flux[:,lyInd], np.sqrt(1.0/invar[:,lyInd])

	# Trimming composites in the blue region - calibration systematics
	if trim_obs_lam > 0:
		F[ZR < ((trim_obs_lam/ly_line) - 1)] = 0

	fn = np.zeros(len(lyInd))

	buck = []
	# Finding the flux at normalization redshift
	for r in range(len(lyInd)):
		f, s, zr = F[:,r], S[:,r], ZR[:,r]
		pos = np.where(f > 0)

		try:
			popt, pcov = curve_fit(tau_model, zr[pos], -np.log(f[pos]) , sigma=s[pos]/f[pos])
			fn[r] = np.exp(-tau_model(zn, popt[0], popt[1], popt[2]))
			buck.append(r)
		except RuntimeError:
			continue

	DT = -np.log(F[:,buck]) + np.log(fn[buck])

	ZR, DT = np.ravel(ZR[:,buck]), np.ravel(DT)

	# REMOVE ALL THE ENTRIES THAT HAVE NO FLUX
	trimInd = np.where(np.isfinite(DT))[0]
	ZR, DT = ZR[trimInd], DT[trimInd]

	if (zset==False):
		zbins = np.arange(min(ZR), max(ZR), 0.05)
	else:
		zbins = zdiv

	ind = [np.where((ZR > zbins[i]) & (ZR <= zbins[i+1]))[0] for i in range(len(zbins)-1)]

	Z_val = (zbins[1:] + zbins[:-1]) / 2
	T_val = [np.mean(DT[ind[i]]) for i in range(len(ind))]

	if (zset==False):
		return Z_val, T_val, zbins
	else:
		return T_val




