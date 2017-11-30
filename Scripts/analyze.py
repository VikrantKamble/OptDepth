import os
import imp
import shutil
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from timeit import default_timer as timer
#from scipy.special import erf
from multiprocessing import Pool

import calibrate
import comp_create
import mcmc_skewer

imp.reload(comp_create)
imp.reload(calibrate)
imp.reload(mcmc_skewer)

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


def analyze(qso, pSel, snt=[2, 50], task='composite', rpix=True, calib=False, distort=True, histbin=False, statistic='mean', frange=[1060, 1170], cutoff=4000, suffix='temp', overwrite=True, skewer_index=-1, parallel=False, triangle=False, visualize=False, verbose=False):

    """
    Creates composites using a given parameter settings as below
    ===========================================================================
    Parameters:

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
    if calib:
        x, y = np.loadtxt('../Data/final_calib.dat')
        f = interp1d(x, y)

        mask = (lObs < 3620) | (lObs > 6950)
        lObs[mask] = np.nan

        corrections = f(lObs)
        myspec /= corrections
        myivar *= corrections

        myivar[mask] = 0

    if rpix:
        outfile += '_rpix'
        myivar[lObs < cutoff] = 0

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
        #rest_range = [[1350, 1360], [1450, 1500]]
        rest_range = [[1280,1290],[1320,1330],[1345,1360],[1440,1480]]
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
        print('Total number of skewers to be analyzed are:', len(skewer_index))
        myspec, myivar, zMat = myspec[:, skewer_index], myivar[:,skewer_index], zMat[:,skewer_index]

        currDir = os.getcwd()
        destDir =  '../LogLikes' + '/Bin_' + suffix + str(frange[0]) + '_' + str(frange[1])

        if not os.path.exists(destDir):
            os.makedirs(destDir)
        else:
            if overwrite:
                shutil.rmtree(destDir)
                os.makedirs(destDir)                

        os.chdir(destDir)

        start = timer()

        # Do not plot graphs while in parallel
        if parallel:
            print('Running in parallel now')
            pool = Pool()
            pool.map(mcmc_skewer.mcmcSkewer, zip(np.array([zMat, myspec, myivar]).T, skewer_index))
            pool.close()
            pool.join()
        elif len(skewer_index) > 1:
            for count, ele in enumerate(skewer_index):
                res = mcmc_skewer.mcmcSkewer([np.array([zMat[:,count], myspec[:,count], myivar[:,count]]).T, ele])
        else:
            res = mcmc_skewer.mcmcSkewer([np.array([zMat, myspec, myivar]).T, skewer_index], return_sampler=True, triangle=triangle, visualize=visualize, verbose=True)

        stop = timer()
        print('Time elapsed:', stop - start)

        os.chdir(currDir)
        return res

def transform(xprime):
    D = np.array([[-0.85627484,  0.51652047],[ 0.51652047,  0.85627484]])
    modPos = np.dot(np.linalg.inv(D), xprime).T + np.array([-5.0625, 3.145])
    print(modPos)

# EOF
