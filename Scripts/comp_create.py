"""
1. Script to create composites from raw files
2. Implements various methods to combine including MAD and likelihood
3. Histrogram binning is currently available

4. Saves to a fits file 
"""

import sys
import os
import imp
import numpy as np
import matplotlib.pyplot as plt

import fitsio
import mu_est

def compcompute(spec, ivar, z, wave, p, zbins, n_chop, histbin, statistic, outfile):
    n_zbins = len(zbins) - 1

    myspec, myivar, myz, myp = spec, ivar, z, p

    # Left closed, right open partitioning
    z0_bins = zbins
    z0_bins[-1] += 0.001
    z_ind = np.digitize(myz, z0_bins)

    # -----------------------------------------------------------------------------------
    if histbin:
        chop1 = np.linspace(min(p[0]), max(p[0]), n_chop[0])
        chop2 = np.linspace(min(p[1]), max(p[1]), n_chop[1])

        # CREATING A 3D DATACUBE OF WEIGHTS
        cube = np.zeros((n_zbins, n_chop[0] - 1, n_chop[1] - 1))

        for i in range(n_zbins):
            ind = np.where((myz >= zbins[i]) & (myz < zbins[i + 1]))[0]
            cube[i] = np.histogram2d(myp[0][ind], myp[1][ind], bins=(chop1, chop2))[0]

        # Trim bins with no objects
        # Outer - parameter; Inner - redshift
        for i in range(n_chop[0] - 1):
            for j in range(n_chop[1] - 1):
                # Sets all bins to 0 if any one bin has no objects in it
                if 0 in cube[:, i, j]:
                    cube[:, i, j] = 0

        count = np.sum(cube, axis=0)

        # A. NORMALIZED WEIGHTS ACROSS ALL REDSHIFTS
        p0_bins, p1_bins = chop1, chop2

        p0_bins[-1] += 0.001  # <-- Required since histogram2d and digitize have different
        p1_bins[-1] += 0.001  # binning scheme

        foo = np.digitize(myp[0], p0_bins)
        blah = np.digitize(myp[1], p1_bins)

        weight_mat = count/cube
        weight_mat[np.isnan(weight_mat)] = 0

        # To obtain consistent weights across all redshifts
        weight_mat = weight_mat/np.linalg.norm(weight_mat, axis=(1, 2))[:, None, None]

        # Final histogram weights to be applied
        hist_weights = weight_mat[z_ind - 1, foo - 1, blah - 1]
    else:
        hist_weights = np.ones(len(z_ind))

    # Arrays that will store the composites and respective errors
    comp_flux, comp_ivar = np.zeros((len(zbins) - 1, len(wave))), np.zeros((len(zbins) - 1, len(wave)))
    comp_red = np.zeros(len(zbins) - 1)

    forest = np.where((wave > 1050) & (wave < 1175))[0]

    for i in range(len(zbins) - 1):
        zind = np.where((myz > zbins[i]) & (myz < zbins[i+1]))[0]
        loc_spec, loc_ivar = myspec[zind], myivar[zind]

        print('%.2f - %.2f : %d' % (zbins[i], zbins[i+1], len(zind)))

        """
        To verify that the histogram rebinning has been done correctly
        plt.figure()
        plt.hist2d(l_p[0][ind], l_p[1][ind], bins=(chop1, chop2), weights=lever)[0]
        plt.colorbar()
        plt.show()
        """
        fig, ax = plt.su

        # At least 30 data points should be available per pixel to estimate the location
        if statistic == 'median':
            for j in range(len(wave)):
                temp = np.where(loc_ivar[:, j] > 0)[0]

                if len(temp) > 30:
                    data = loc_spec[:, j][temp]

                    comp_flux[i, j] = np.median(data)
                    sigma = 1.253 * np.std(data) / np.sqrt(len(temp))
                    #sigma = 0.93 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.sqrt(len(temp))

                    comp_ivar[i,j] = 1.0 / sigma ** 2

            comp_red[i] = np.average(myz[zind])

        if statistic == 'mean':
            for j in range(len(wave)):
                temp = np.where(loc_ivar[:, j] > 0)[0]

                if len(temp) > 30:
                    data = loc_spec[:, j][temp]

                    comp_flux[i, j] = np.average(data)
                    sigma = np.std(data) / np.sqrt(len(temp))

                    comp_ivar[i,j] = 1.0 / sigma ** 2

            comp_red[i] = np.average(myz[zind])

        if statistic == 'MAD':
            for j in range(cfg.NPIX):
                temp = np.where(loc_ivar[:, j] > 0)[0]

                if len(temp) > 30:
                    data = loc_spec[:, j][temp]

                    # Outlier removal using MAD
                    mad = 1.4826 * np.median(np.abs(data - np.median(data)))

                    # Decision criterion to remove outliers (2.5 = Decent)
                    out = np.where(np.abs((data - np.median(data))/mad) <= 3)[0]

                    if len(out) > 10:
                        comp_flux[i, j] = np.mean(data[out])
                        sigma = np.std(data[out]) / np.sqrt(len(out))

                        comp_ivar[i,j] = 1.0 /sigma ** 2

            comp_red[i] = np.average(l_z[ind])

        if statistic == 'hmean':
            for j in range(len(wave)):
                temp = np.where(loc_ivar[:, j]*lever > 0)[0]

                if len(temp) > 30:
                    data, hweights = loc_spec[:, j][temp], lever[temp]

                    comp_flux[i - target, j] = np.average(data, weights=hweights)
                    comp_sigma[i - target, j] = np.std(data) * np.sqrt(np.sum(hweights ** 2)) / np.sum(hweights) 

            comp_red[i - target] = np.average(l_z[ind], weights=lever)

        if statistic == 'likelihood':
            for j in forest:
                vi = loc_ivar[:, j]
                xi = loc_spec[:, j]

                temp = np.where((vi > 0) & (np.abs(xi) < 10))[0]

                if len(temp) > 30:
                    xi = xi[temp]
                    ei = 1.0 / np.sqrt(vi[temp])

                    res = mu_est.param_est(xi, ei)

                    comp_flux[i, j], comp_ivar[i, j] = res[0], 1.0 / res[1] ** 2

            comp_red[i] = np.average(myz[zind])

    # ----------------------------------------------------------------------------
    outfile = '../Composites/' + outfile + '_' + statistic + '.fits'
    if os.path.exists(outfile): os.remove(outfile)

    fits = fitsio.FITS(outfile, 'rw')

    # Writing operation
    fits.write(comp_flux)
    fits.write(comp_ivar)
    fits.write([comp_red], names=['REDSHIFT'])
    fits.write([wave], names=['WAVELENGTH'])

    fits.close()
    print("Writing of the composite to fits file complete. Filename: %s" % outfile)
