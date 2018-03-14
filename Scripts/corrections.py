import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


def var_correct(qso, rest_ranges, zq_range=[3, 3.5], fit_model=True):
    """ Corrections to the vriance assigned to the pixels by the pipeline

    Notes:
        1. Scaling spectra doesn't affect this estimate
        2. If the spectra changes by a lot(isn't flat), it will lead to
               underestimates of eta
    """
    zq_ind = np.where((qso.zq > zq_range[0]) & (qso.zq <= zq_range[1]))[0]

    # restframe ranges over which to analyze
    # currently this allows for only a single bin
    if np.asarray(rest_ranges).ndim == 1:
        raise TypeError("Please provide the ranges as two dimensional array")

    lambda_mean, eta = [], []
    for ranges in rest_ranges:
        ind_set = (qso.wl > ranges[0]) & (qso.wl <= ranges[1])

        # create local flux and ivar matrices
        loc_flux = qso.flux[zq_ind[:, None], ind_set]
        loc_ivar = qso.ivar[zq_ind[:, None], ind_set]

        # mask to select pixels that provide information
        ivar_mask = loc_ivar > 0

        # sum of good pixels along each spectra where num is greater than 10
        num = np.sum(ivar_mask, 1)
        num_ind = num > 10

        # chi-square along each spectra
        # eta = N / sum((f_i - mu)^2 / sigma_i^2)
        mu = np.average(loc_flux[num_ind], weights=loc_ivar[num_ind], axis=1)

        chisq = np.sum((
            loc_flux[num_ind] - mu[:, None]) ** 2 * loc_ivar[num_ind], axis=1)

        lambda_obs = np.array((np.mat(qso.wl[ind_set]).T *
                               np.mat(1 + qso.zq[zq_ind][num_ind]))).T

        # mean of observed wavelength spanned along each spectra
        lambda_mean += list(np.average(lambda_obs, weights=ivar_mask[num_ind], axis=1))

        # eta values along each spectra
        eta += list(num[num_ind] / chisq)

    # binned statistic with scipy
    y, bin_edges, binnumber = binned_statistic(lambda_mean, eta,
                                               statistic='mean', bins=100)

    bin_width = (bin_edges[1] - bin_edges[0])
    X = bin_edges[1:] - bin_width/2

    # plot the results if specified
    plt.plot(X, y, '+', color='k', markersize=8)

    # fit a simple piecewise function to the data
    if fit_model:
        popt1 = np.polyfit(X[X < 5850], y[X < 5850], deg=1)
        popt2 = np.polyfit(X[X > 5850], y[X > 5850], deg=2)

        xline = np.linspace(3500, 7500, 1000)
        plt.plot(xline, np.polyval(popt1, xline), '-k')
        plt.plot(xline, np.polyval(popt2, xline), '--k')
        plt.show()

        np.savetxt("var_correct.txt", list(popt1) + list(popt2))


def calibrate(wl, spec, ivar, zq, rest_range, norm_min, norm_max, plotit, savefile):
    """ Obtain flux calibration vector by doing optical depth analysis redwards
        of Lyman-Alpha

        Only the shape is estimated, the overall normalization is unconstrained
    """
    # Collect relevant indices over restframe
    r_ind = []
    for j in range(len(rest_range)):
        foo = np.where((wl > rest_range[j][0]) & (wl < rest_range[j][1]))[0]
        r_ind = np.concatenate((r_ind, foo))
    rInd = r_ind.astype(int)

    # Obtain the corresponding data matrices
    lam_obs = np.array(np.mat(wl[rInd]).T * np.mat(1 + zq)).T
    cflux, civar = spec[:, rInd], ivar[:, rInd]

    # Scale to the same baseline
    # this will introduce addtional errors that we are neglecting
    nValue = np.zeros(len(rInd))
    for i in range(len(rInd)):
        blah = np.where((lam_obs[:, i] > norm_min) & (lam_obs[:, i] < norm_max)
                        & (civar[:, i] > 0))[0]
        nValue[i] = np.average(cflux[:, i][blah], weights=civar[:, i][blah])

    # Scale fluxes and ivars accordingly
    NormFlux = cflux / nValue
    NormIvar = civar * nValue ** 2

    pixObs, = np.ravel(lam_obs)
    pixFlux, pixIvar = np.ravel(NormFlux), np.ravel(NormIvar)

    # Controls the smoothing of the results
    ObsBin = np.arange(3500, 7000, 3)

    # Correction vector
    Cvec = np.zeros(len(ObsBin) - 1)

    for k in range(len(Cvec)):
        bInd = np.where((pixObs > ObsBin[k]) & (pixObs <= ObsBin[k + 1])
                        & (pixIvar > 0) & np.isfinite(pixFlux))[0]

        if len(bInd) > 5:
            Cvec[k] = np.average(pixFlux[bInd], weights=pixIvar[bInd])
    Lvec = (ObsBin[1:] + ObsBin[:-1])/2.

    if plotit:
        # plt.figure(figsize=(10 , 4))
        good = Cvec != 0
        plt.plot(Lvec[good], Cvec[good], '-r')
        plt.xlabel(r'$\lambda_{obs}$', fontsize=20)
        plt.ylabel(r'$Correction$', fontsize=20)
        # plt.xlim(1.8 , 3)
        plt.ylim(0.9, 1.1)
        plt.axhline(1, c='r')
        plt.show()

        if savefile:
            np.savetxt('../Data/calibration.dat', [Lvec[good], Cvec[good]])

# EOF
