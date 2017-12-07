"""
1. Estimate the fractional corrections to pipeline varaince using low-z quasars
2. The corrections are huge 
3. Fit it by simple functions - Using quasars themselves. 

4. Someone needs to do independent experiements to calibrate the spectrographs if we are to do this correctly
5. REMEMBER ITS CORRECTION TO VARIANCE AND NOT SIGMA - SO CAN BE APPLIED DIRECTLY TO THE VARIANCE VECTOR
"""

#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip


#[  6.82539672e-07  -8.88155048e-03   3.01672220e+01]
#[ 0.00024917  0.06737015]

def VarCorrect(fluxData, ivarData, zq, wl, sn, plotit):
	zqInd = np.where((sn > 5) & (zq > 1.6) & (zq < 4))[0]
	print('Total number of such spectra:', len(zqInd))

	# Select Indices of the corresponding bins
	IndSet1 = np.where((wl > 1350) & (wl < 1360))[0]
	IndSet2 = np.where((wl > 1470) & (wl < 1480))[0]

	# Store final dataset here
	X, y = [], []

	# Iterate over all the relevant quasars to create dataset
	for ele in zqInd:
		# Use pixels that contain information
		foo = np.where(ivarData[ele][IndSet1] > 0)[0]
		blah = np.where(ivarData[ele][IndSet2] > 0)[0]

		if len(foo) > 10:
			loc_flux, loc_ivar, loc_wave = fluxData[ele][IndSet1[foo]], ivarData[ele][IndSet1[foo]], wl[IndSet1[foo]]

			# Remove possible metal absoprtion lines using 3-sigma clipping
			clipholes = ~sigma_clip(loc_flux).mask
			loc_flux, loc_ivar = loc_flux[clipholes], loc_ivar[clipholes]

			X.append(np.mean(loc_wave[clipholes]) * (1 + zq[ele]))

			mu= np.average(loc_flux, weights = loc_ivar)
			eta = len(loc_flux) / np.sum((loc_flux - mu) ** 2 * loc_ivar)

			y.append(eta)

		if len(blah) > 10:
			loc_flux, loc_ivar, loc_wave = fluxData[ele][IndSet2[blah]], ivarData[ele][IndSet2[blah]], wl[IndSet2[blah]]

			clipholes = ~sigma_clip(loc_flux).mask
			loc_flux, loc_ivar = loc_flux[clipholes], loc_ivar[clipholes]

			X.append(np.mean(loc_wave[clipholes]) * (1 + zq[ele]))

			mu= np.average(loc_flux, weights = loc_ivar)
			eta = len(loc_flux) / np.sum((loc_flux - mu) ** 2 * loc_ivar)

			y.append(eta)

	X, y = np.array(X), np.array(y)
	if plotit:
		Xbins = np.linspace(min(X), max(X), 100)
		ybins = []
		for i in range(len(Xbins)-1):
			temp = np.where((X > Xbins[i]) & (X < Xbins[i+1]))[0]
			if len(temp) > 10:
				ybins.append(np.mean(y[temp]))
			else:
				ybins.append(np.nan)

		Xbins = (Xbins[1:] + Xbins[:-1]) / 2.
	return Xbins, np.array(ybins)


def parameterize(X, y):
	ind = np.where(np.isfinite(y) & (X > 5850))[0]
	popt1 = np.polyfit(X[ind], y[ind], deg=2)
	myfunc1 = np.poly1d(popt1)

	ind = np.where(np.isfinite(y) & (X < 5850))[0]
	popt2 = np.polyfit(X[ind], y[ind], deg=1)
	myfunc2 = np.poly1d(popt2)

	plt.plot(X, y, '+', color='k', markersize=12)

	x = np.linspace(3500, 7500, 1000)


	plt.plot(x, myfunc1(x), '-k')
	plt.plot(x, myfunc2(x), '--k')

	y_new = np.piecewise(x, [x < 5850, x >=5850], [lambda x: myfunc2(x), lambda x: myfunc1(x)])
	plt.plot(x, y_new)

	plt.show()

	return x, y_new
	print(popt1)
	print(popt2)


def calibrate(wl, spec, ivar, zq, rest_range, norm_min, norm_max, plotit, savefile):
    # Collect relevant indices over restframe
    r_ind = []
    for j in range(len(rest_range)):
        foo = np.where((wl > rest_range[j][0]) & (wl < rest_range[j][1]))[0]
        r_ind = np.concatenate((r_ind , foo))
    rInd = r_ind.astype(int)

    # Obtain the corresponding data matrices
    lam_obs = np.array(np.mat(wl[rInd]).T * np.mat(1 + zq)).T
    cflux , civar = spec[:, rInd], ivar[:, rInd]

    # Scale to the same baseline 
    # this will introduce addtional errors that we are neglecting
    nValue = np.zeros(len(rInd))
    for i in range(len(rInd)):
        blah = np.where((lam_obs[:, i] > norm_min) & (lam_obs[:, i] < norm_max) & (civar[:,i] > 0))[0]
        nValue[i] = np.average(cflux[:, i][blah], weights=civar[:, i][blah])
        #nValue[i] = np.median(cflux[:, i][blah])

    # Scale fluxes and ivars accordingly
    NormFlux = cflux / nValue
    NormIvar = civar * nValue ** 2

    pixObs, pixFlux, pixIvar = np.ravel(lam_obs), np.ravel(NormFlux), np.ravel(NormIvar)

    # Controls the smoothing of the results
    ObsBin = np.arange(3500, 7000, 4)

    # Correction vector
    Cvec = np.zeros(len(ObsBin) - 1)

    for k in range(len(Cvec)):
        bInd = np.where((pixObs > ObsBin[k]) & (pixObs <= ObsBin[k + 1]) & (pixIvar > 0) & np.isfinite(pixFlux))[0]
        if len(bInd) > 5:
            Cvec[k] = np.average(pixFlux[bInd], weights=pixIvar[bInd])
            #Cvec[k] = np.median(pixFlux[bInd])
    Lvec = (ObsBin[1:] + ObsBin[:-1])/2.

    if (plotit == True):
        #plt.figure(figsize=(10 , 4))
        good = Cvec != 0
        plt.plot(Lvec[good], Cvec[good], '-g')
        plt.xlabel(r'$\lambda_{obs}$' , fontsize=20)
        plt.ylabel(r'$Correction$' , fontsize=20)
        # plt.xlim(1.8 , 3)
        plt.ylim(0.9 , 1.1)
        plt.axhline(1, c='r')
        plt.show()

        if savefile:
            np.savetxt('../Data/calibration.dat', [Lvec[good], Cvec[good]])

# EOF
