import os
import importlib
import shutil
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from scipy.special import erf
from multiprocessing import Pool

import calibrate
import comp_create
import mcmc_skewer

importlib.reload(comp_create)
importlib.reload(calibrate)
importlib.reload(mcmc_skewer)

ly_line = 1215.67


def find_zbins(z, zstart=2.3):
    curr_zbins = [zstart]
    curr_z = zstart

    while True:
        pos = np.where((z > curr_z) & (z <= curr_z + 0.12))[0]

        if len(pos) > 50:
            curr_z += 0.12
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


def makeComp(qso, pSel, snt=[2, 50], task='composite', rpix=True, calib=False, distort=True, skew=False, histbin=False, statistic='mean', frange=[1060, 1170], cutoff=4000, suffix='temp', skewer_index=-1, parallel=False):

    """
    Creates composites using a given parameter settings as below

    :param qso: An object of the QSO class 
    :param snt: Signal-to-noise range of the objects selected
    :param pSel: A 2D matrix to specify the range of the parameters 
    :param task: Indicates whether to create 'composite' or to 'calibrate'
    :param rpix: Remove pixels below a certain 'cutoff' in the observer-frame
    :param distort: Whether to distort each spectra to a common spectral index
    :param skew: An artificial distortion according to any function defination to be applied in the observer frame
    :param histbin: Whether to do histogram rebinning
    :param statistic: The statistic used in creating the composites - 'mean', 'median', 'MAD', 'likelihood'
    :param frange: The range in rest-frame over which the composites are created. Set to 0 to span the full rest-frame wavelength
    :param suffix: The name of the output file - stored in the pwd folder
    :param skewer_index: The index over which to test mcmc on a skewer
    :param parallel: Whether to do mcmc_skewer in parallel
    """

    # A. DATA ACQUISITION AND TRIMMING --------------------------------------------------
    cut = np.where((qso.sn > snt[0]) & (qso.sn < snt[1])  & (qso.p1 > pSel[0][0]) & (qso.p1 <= pSel[0][1]) & (qso.p2 > pSel[1][0]) & (qso.p2 <= pSel[1][1]))[0]
    
    outfile = 'comp_' + suffix

    myspec, myivar = qso.flux[cut], qso.ivar[cut]
    myz, myp, myalpha = qso.zq[cut], np.array([qso.p1[cut], qso.p2[cut]]), qso.alpha[cut]
    print('Total number of spectra after selection cuts: %d' %len(myspec))

    # B. DATA PREPROCESSING -------------------------------------------------------------
    # 1. Calibrate the flux vector
    # if calib:   # DO SOMEHTING HERE

    # The observer wavelengths and redshifts
    lObs = np.array((np.mat(qso.wl).T * np.mat(1 + myz))).T
    zMat = lObs / ly_line - 1
    
    # 2. Clip pixels off blue-end
    if rpix:
        outfile += '_rpix'
        myivar[lObs < cutoff] = 0

    # 3. Skew spectra in the observer frame - TESTING !!!
    if skew:
        outfile += '_skew'
        skewMat = erf((lObs - 2384) * 0.000743)
        myspec *= skewMat
        myivar /= skewMat ** 2

    # 4. Distort spectra to remove power-law variations
    if distort:
        outfile += '_distort'
        CenAlpha = np.median(myalpha)
        distortMat = np.array([(qso.wl / 1450.) ** ele for ele in (CenAlpha - myalpha)])
        myspec *= distortMat
        myivar /= distortMat ** 2
        print('All spectra distorted to alpha:', CenAlpha)

    # C. CALIBRATION VS ESTIMATION ------------------------------------------------------
    if task == 'calibrate':
        Lind = (myz > 1.6) & (myz < 4)
        print('Number of spectra used for calibration are: %d' %Lind.sum())
        rest_range = [[1350, 1360], [1450, 1500]]
        # normalization range used
        obs_min, obs_max = 4680, 4720 

        calibrate.calibrate(qso.wl, myspec[Lind], myivar[Lind], myz[Lind], rest_range, obs_min, obs_max, plotit=True, savefile=True)

    if frange == 0:
        lyInd = np.arange(len(qso.wl))
    else:
        lyInd = np.where((qso.wl > frange[0]) & (qso.wl < frange[1]))[0]

    myspec, myivar,  = myspec[:, lyInd], myivar[:, lyInd]
    zMat, mywave = zMat[:, lyInd], qso.wl[lyInd]

    # D. COMPOSITE CREATION IF SPECIFIED ------------------------------------------------
    if task  == 'composite':

        zbins = find_zbins(myz)
    
        comp_create.compcompute(myspec, myivar, myz, mywave, myp, zbins, qso.n_chop, histbin, statistic, outfile)

    # E. LIKELIHOOD SKEWER --------------------------------------------------------------
    if task == 'skewer':
        myspec, myivar, zMat = myspec[:, skewer_index], myivar[:,skewer_index], zMat[:,skewer_index]

        currDir = os.getcwd()
        destDir =  '../LogLikes' + '/Bin_' + suffix + str(frange[0]) + '_' + str(frange[1])
        if os.path.exists(destDir): shutil.rmtree(destDir)

        os.makedirs(destDir)
        os.chdir(destDir)

        start = timer()

        # Do not plot graphs while in parallel
        if parallel:
            pool = Pool()
            pool.map(mcmc_skewer.mcmcSkewer, zip(np.array([zMat, myspec, myivar]).T, skewer_index))
            pool.close()
            pool.join()
        else:
            for count, ele in enumerate(skewer_index):
                res = mcmc_skewer.mcmcSkewer([np.array([zMat[:,count], myspec[:,count], myivar[:,count]]).T, ele])

        stop = timer()
        print('Time elapsed:', stop - start)

        os.chdir(currDir)


# EOF