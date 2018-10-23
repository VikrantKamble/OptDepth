import os
import imp
import shutil
import numpy as np

from timeit import default_timer as timer
from multiprocessing import Pool
from functools import partial

# local imports
from Scripts import corrections
from Scripts import comp_simple
from Scripts import get_comp
from Scripts import mcmc_skewer

imp.reload(comp_simple)
imp.reload(mcmc_skewer)
imp.reload(corrections)
imp.reload(get_comp)


def find_zbins(z, zstart=2.1, deltaz=0.2):
    """ Create bins in quasar redshifts with a given minimum
    number of objects in each bin.
    * Use this in conjuction with the composite creation code
    """
    curr_zbins = [zstart]
    curr_z = zstart

    while True:
        pos = np.where((z > curr_z) & (z <= curr_z + 0.2))[0]

        if len(pos) > 50:
            curr_z += 0.2
            curr_zbins.append(curr_z)
        else:
            pos = np.where((z > curr_z) & (z <= curr_z + 0.18))[0]
            if len(pos) > 50:
                curr_z += 0.18
                curr_zbins.append(curr_z)
            else:
                pos = np.where((z > curr_z) & (z <= curr_z + 0.24))[0]
                if len(pos) > 50:
                    curr_z += 0.24
                    curr_zbins.append(curr_z)
                else:
                    break
    return np.array(curr_zbins)


def hist_weights(p1, p2, z, zbins, n_chop=4, truncated=True):
    """ Function to get the appropriate weights, such that
    quasars i n different bins all have the same probability
    distribution in different redshift bins

    * This assumes that the redshifts z have been properly
    truncated to be in the zbins range.
    """
    if not truncated:
        ixs = (z >= zbins[0]) & (z < zbins[-1])
        z = z[ixs]
        p1, p2 = p1[ixs], p2[ixs]

    n_zbins = len(zbins) - 1

    # Left closed, right open partitioning
    z0_bins = zbins
    z0_bins[-1] += 0.001
    z_ind = np.digitize(z, z0_bins)

    chop1 = np.linspace(min(p1), max(p1), n_chop)
    chop2 = np.linspace(min(p2), max(p2), n_chop)

    # CREATING A 3D DATACUBE OF WEIGHTS
    cube = np.zeros((n_zbins, n_chop - 1, n_chop - 1))

    for i in range(n_zbins):
        ind = (z >= zbins[i]) & (z < zbins[i + 1])
        cube[i] = np.histogram2d(p1[ind], p2[ind], bins=(chop1, chop2))[0]

    # Trim bins with no objects
    # Outer - parameter; Inner - redshift
    for i in range(n_chop - 1):
        for j in range(n_chop - 1):
            # Sets all bins to 0 if any one bin has no objects in it
            if 0 in cube[:, i, j]:
                cube[:, i, j] = 0

    cube_sum = np.sum(cube, axis=0)

    # A. NORMALIZED WEIGHTS ACROSS ALL REDSHIFTS
    p0_bins, p1_bins = chop1, chop2

    # <-- Required since histogram2d and digitize have different
    # binning schemes
    p0_bins[-1] += 0.001
    p1_bins[-1] += 0.001

    foo = np.digitize(p1, p0_bins)
    blah = np.digitize(p2, p1_bins)

    weight_mat = cube_sum / cube
    weight_mat[np.isnan(weight_mat)] = 0

    # To obtain consistent weights across all redshifts
    weight_mat /= np.linalg.norm(weight_mat, axis=(1, 2))[:, None, None]

    # Final histogram weights to be applied
    h_weights = weight_mat[z_ind - 1, foo - 1, blah - 1]

    """
    # To verify that the histogram rebinning has been done correctly
    for i in range(n_zbins):
        ind = (z >= zbins[i]) & (z < zbins[i + 1])
        plt.figure()
        plt.hist2d(p1[ind], p2[ind], bins=(chop1, chop2), weights=h_weights[ind], normed=True)[0]
        plt.colorbar()
    plt.show()
    """
    return h_weights


def analyze(binObj, task='skewer', rpix=False, distort=True, CenAlpha=None,
            histbin=False, statistic='mean', frange=[1070, 1160],
            cutoff=[3700, 8000], suffix='temp', overwrite=False,
            skewer_index=[-1], zq_cut=[0, 5], parallel=False,
            calib_kwargs=None, skewer_kwargs={}):

    outfile = task + '_' + suffix

    if task == 'skewer':
        lyInd = np.where((binObj.wl > frange[0]) & (binObj.wl < frange[1]))[0]
        if skewer_index == 'all':
            skewer_index = np.arange(len(lyInd))

        print('Total skewers available: {}, skewers analyzed in this '
              'run: {}'.format(len(lyInd), len(skewer_index)))

        myspec = binObj._flux[:, lyInd[skewer_index]]
        myivar = binObj._ivar[:, lyInd[skewer_index]]
        zMat = binObj._zAbs[:, lyInd[skewer_index]]
        mywave = binObj.wl[lyInd[skewer_index]]
    else:
        myspec, myivar, zMat = binObj._flux, binObj._ivar, binObj._zAbs
        mywave = binObj.wl

    myz, myalpha = binObj._zq, binObj._alpha

    # selecting according to quasar redshifts
    zq_mask = (myz > zq_cut[0]) & (myz < zq_cut[1])

    myspec = myspec[zq_mask]
    myivar = myivar[zq_mask]
    zMat = zMat[zq_mask]
    myz, myalpha = myz[zq_mask], myalpha[zq_mask]

    # B. DATA PREPROCESSING ---------------------------------------------------
    if histbin:
        """ Histogram binning in parameter space """
        myp1, myp2 = binObj._par1, binObj._par2

        myzbins = find_zbins(myz)
        hInd = np.where((myz >= myzbins[0]) & (myz < myzbins[-1]))

        # Modify the selection to choose only objects that fall in the
        # zbins range
        myz, myalpha = myz[hInd], myalpha[hInd]
        myp1, myp2 = myp1[hInd], myp2[hInd]

        myspec, myivar = myspec[hInd], myivar[hInd]
        zMat = zMat[hInd]

        if binObj._hWeights is None:
            h_weights = hist_weights(myp1, myp2, myz, myzbins)
            binObj._hWeights = h_weights

            myivar = myivar * h_weights[:, None]
        else:
            myivar = myivar * binObj._hWeights[:, None]

    if rpix:
        """ Restrict wavelength coverage in observer frame
            Defined in redshift units not wavelegth units
        """
        outfile += '_rpix'
        cutoff = np.array(cutoff) / 1215.67 - 1
        myivar[(zMat < cutoff[0]) | (zMat > cutoff[1])] = 0
        print('Pixels masked in the observer range', cutoff)

    if distort:
        """ Distort spectra in alpha space """
        outfile += '_distort'

        if CenAlpha is None:
            CenAlpha = np.median(myalpha)
        distortMat = np.array([(mywave / 1450.) ** ele for ele in (CenAlpha - myalpha)])

        myspec *= distortMat
        myivar /= distortMat ** 2

        print('All spectra distorted to alpha:', CenAlpha)

    # C. CALIBRATION VS ESTIMATION --------------------------------------------
    if task == 'calibrate':
        Lind = (myz > 1.6) & (myz < 4)
        print('Number of spectra used for calibration are: %d' % Lind.sum())

        rest_range = [[1280, 1290], [1320, 1330], [1345, 1360], [1440, 1480]]

        # normalization range used
        obs_min, obs_max = 4600, 4640

        corrections.calibrate(binObj.wl, myspec[Lind], myivar[Lind], myz[Lind],
                              rest_range, obs_min, obs_max, binObj.name, True)

    # D. COMPOSITE CREATION IF SPECIFIED --------------------------------------
    if task == 'composite':
        """
            Create composites using the spectra
            Update the composite code in light of new changes here
        """
        # zbins = find_zbins(myz)
        zbins = np.arange(2.1, 4.5, 0.2)
        comp_simple.compcompute(myspec, myivar, myz, mywave,
                                zbins, statistic, outfile)

    if task == 'binned':
        mu, sig = get_comp.get_comp(myspec, myivar, zMat, binObj.wl, frange=frange)
        return mu, sig

    # E. LIKELIHOOD SKEWER ----------------------------------------------------
    if task == 'skewer':
        currDir = os.getcwd()
        destDir = '../LogLikes' + '/Bin_' + outfile +\
                  str(frange[0]) + '_' + str(frange[1])

        if not os.path.exists(destDir):
            os.makedirs(destDir)
        else:
            if overwrite:
                shutil.rmtree(destDir)
                os.makedirs(destDir)
        os.chdir(destDir)

        start = timer()

        # Do not plot graphs while in parallel
        res = None
        if parallel:
            print('Running in parallel now')

            myfunc_partial = partial(mcmc_skewer.mcmcSkewer, **skewer_kwargs)

            pool = Pool()
            res = pool.map(myfunc_partial,
                           zip(np.array([zMat, myspec, myivar]).T, skewer_index))
            pool.close()
            pool.join()
        elif len(skewer_index) > 1:
            for count, ele in enumerate(skewer_index):
                res = mcmc_skewer.mcmcSkewer(
                    [np.array([zMat[:, count], myspec[:, count], myivar[:, count]]).T, ele],
                    **skewer_kwargs)
        else:
            res = mcmc_skewer.mcmcSkewer(
                [np.array([zMat, myspec, myivar]).T, skewer_index[0]], **skewer_kwargs)
            # return zMat, myspec, myivar
        stop = timer()
        print('Time elapsed:', stop - start)

        os.chdir(currDir)
        return mywave, res


def reconstruct(bin, tau_mean, tau_cov, niter=100, frange=[1070, 1160],
                b_kwargs={}):
    """
    Create functional pdf of reconstructed continuum using a
    pdf on optical depth parameters

    Parameters:
    ----------
    bin      : the binObj on which to apply the function
    tau_mean : location of optical depth params
    tau_cov  : their covariance matrix
    niter    : number of continuum evaluations following the pdf

    Returns:
    ---------
    A 2D array of size niter * n_forest_pixels
    """
    # Wavelength to return for plotting
    ixs = (bin.wl > frange[0]) & (bin.wl < frange[1])

    # Sample points from the Gaussian pdf
    if tau_cov is None:
        ln_tau0, gamma = tau_mean[:, None]
        niter = 1
    else:
        ln_tau0, gamma = np.random.multivariate_normal(tau_mean,
                                                       tau_cov, size=niter).T
    tau0 = np.exp(ln_tau0)

    # Create 2d array to store the result
    wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)
    ixs = (wl > frange[0]) & (wl < frange[1])

    data = np.zeros((niter, ixs.sum()))

    # Loop through and aggregate the results
    for i in range(niter):
        s_kwargs = {'logdef': 4, 'truths':  [tau0[i], gamma[i]]}

        __, data[i] = analyze(bin, parallel=True, frange=frange,
                              skewer_index='all', skewer_kwargs=s_kwargs,
                              **b_kwargs)
    return bin.wl[ixs], data
