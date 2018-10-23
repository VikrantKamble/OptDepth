"""
1. Script to create composites from raw files
2. Implements various methods to combine including MAD and likelihood
3. Histrogram binning is currently available

4. Saves to a fits file
"""

import os
import numpy as np
import fitsio

from Scripts.plotting import plotcomp


def compcompute(spec, ivar, z, wave, zbins,
                statistic='mean', outfile='temp', plot_comp=True):
    n_zbins = len(zbins) - 1

    # Arrays that will store the composites and respective errors
    comp_flux = np.zeros((n_zbins, len(wave)))
    comp_ivar = np.zeros((n_zbins, len(wave)))
    comp_red = np.zeros(n_zbins)

    for i in range(n_zbins):
        zind = np.where((z > zbins[i]) & (z < zbins[i+1]))[0]
        loc_spec, loc_ivar = spec[zind], ivar[zind]

        print('%.2f - %.2f : %d' % (zbins[i], zbins[i+1], len(zind)))

        # At least 30 data points should be available per pixel to
        # estimate the location
        if statistic == 'median':
            for j in range(len(wave)):
                temp = np.where(loc_ivar[:, j] > 0)[0]

                if len(temp) > 50:
                    data = loc_spec[:, j][temp]

                    comp_flux[i, j] = np.median(data)
                    sigma = 1.253 * np.std(data) / np.sqrt(len(temp))

                    comp_ivar[i, j] = 1.0 / sigma ** 2

            comp_red[i] = np.average(z[zind])

        if statistic == 'mean':
            for j in range(len(wave)):
                temp = np.where(loc_ivar[:, j] > 0)[0]

                if len(temp) > 50:
                    data = loc_spec[:, j][temp]

                    comp_flux[i, j] = np.mean(data)

                    # Bootstrap to get CI
                    shuffled = np.random.choice(data, replace=True, size=(500, len(data)))
                    sigma = np.std(np.mean(shuffled, axis=1))

                    comp_ivar[i, j] = 1.0 / sigma ** 2

            comp_red[i] = np.average(z[zind])

    # ----------------------------------------------------------------------------
    outfile = '../Composites/' + outfile + '_' + statistic + '.fits'
    if os.path.exists(outfile):
        os.remove(outfile)

    fits = fitsio.FITS(outfile, 'rw')

    # Writing operation
    fits.write(comp_flux)
    fits.write(comp_ivar)
    fits.write([comp_red], names=['REDSHIFT'])
    fits.write([wave], names=['WAVELENGTH'])

    fits.close()
    print("Writing of the composite to fits file complete. Filename: %s" % outfile)

    if plot_comp:
        plotcomp(outfile, nskip=3)
