"""
1. Assess flux calibrations using quasars itself
2. Using weighted-average throughout
"""


import numpy as np
import matplotlib.pyplot as plt


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
    nValue = np.zeros(len(rInd))

    for i in range(len(rInd)):
        blah = np.where((lam_obs[:, i] > norm_min) & (lam_obs[:, i] < norm_max) & (civar[:,i] > 0))[0]
        # nValue[i] = np.average(cflux[:, i][blah], weights=civar[:, i][blah])
        nValue[i] = np.median(cflux[:, i][blah])

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
            # Cvec[k] = np.average(pixFlux[bInd], weights=pixIvar[bInd])
            Cvec[k] = np.median(pixFlux[bInd])
    Lvec = (ObsBin[1:] + ObsBin[:-1])/2.

    if (plotit == True):
        #plt.figure(figsize=(10 , 4))
        good = Cvec != 0
        plt.plot(Lvec[good], Cvec[good], '-k')
        plt.xlabel(r'$\lambda_{obs}$' , fontsize=20)
        plt.ylabel(r'$Correction$' , fontsize=20)
        # plt.xlim(1.8 , 3)
        plt.ylim(0.9 , 1.1)
        plt.axhline(1, c='r')
        plt.show()

        if savefile:
            np.savetxt('calibration.dat', [Lvec[good], Cvec[good]])

# EOF
