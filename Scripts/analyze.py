import os
import imp
import shutil
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from timeit import default_timer as timer
from multiprocessing import Pool

from spec_corrections import calibrate
import comp_create
import mcmc_skewer

imp.reload(comp_create)
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


def hist_weights(p1, p2, z, zbins, n_chop=4):
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

    p0_bins[-1] += 0.001  # <-- Required since histogram2d and digitize have different
    p1_bins[-1] += 0.001  # binning scheme

    foo = np.digitize(p1, p0_bins)
    blah = np.digitize(p2, p1_bins)

    weight_mat = cube_sum / cube
    weight_mat[np.isnan(weight_mat)] = 0

    # To obtain consistent weights across all redshifts
    weight_mat = weight_mat / np.linalg.norm(weight_mat, axis=(1, 2))[:, None, None]

    # Final histogram weights to be applied
    h_weights = weight_mat[z_ind - 1, foo - 1, blah - 1]

    """
    #To verify that the histogram rebinning has been done correctly
    for i in range(n_zbins):
        ind = (z >= zbins[i]) & (z < zbins[i + 1])
        plt.figure()
        plt.hist2d(p1[ind], p2[ind], bins=(chop1, chop2), weights=h_weights[ind], normed=True)[0]
        plt.colorbar()
    plt.show()
    """

    return h_weights

# dictionary to hold the bin ranges
myDict = {
    'one': [[-2.8, -2.13], [20, 40]],
    'two': [[-2.8, -2.13], [40, 60]],
    'three': [[-2.13, -1.46], [20, 33.3]],
    'four': [[-2.13, -1.46], [33.3, 46.6]],
    'five': [[-2.13, -1.46], [46.6, 60]],
    'six': [[-1.46, -0.8], [20, 40]],
    'seven': [[-1.46, -0.8], [40, 60]] 
}

def analyze(qso, nbin, snt=[2, 50], task='composite', rpix=True, calib=False, distort=True, histbin=False, statistic='mean', frange=[1060, 1170], cutoff=[4000, 8000], suffix='temp', overwrite=True, skewer_index=[-1], parallel=False, **kwargs):

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
    pSel = myDict[nbin]
    cut = np.where((qso.sn > snt[0]) & (qso.sn < snt[1])  & (qso.p1 > pSel[0][0]) & (qso.p1 <= pSel[0][1]) & (qso.p2 > pSel[1][0]) & (qso.p2 <= pSel[1][1]))[0]
    
    outfile = 'comp_' + suffix + '_'

    myspec, myivar = qso.flux[cut], qso.ivar[cut]
    myz, myp1, myp2, myalpha = qso.zq[cut], qso.p1[cut], qso.p2[cut], qso.alpha[cut]

    print('Total number of spectra after selection cuts: %d' %len(myspec))

    # B. DATA PREPROCESSING -------------------------------------------------------------
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
        myivar *= corrections # <--- mistake

        myivar[mask] = 0
        print('All spectra corrected for flux calibration errors')

    if histbin:
        myzbins = find_zbins(myz)
        hInd = np.where((myz >= myzbins[0]) & (myz < myzbins[-1]))

        myp1, myp2, myz, myalpha = myp1[hInd], myp2[hInd], myz[hInd], myalpha[hInd]
        myspec, myivar = myspec[hInd], myivar[hInd]

        lObs = lObs[hInd]
        zMat = zMat[hInd]

        h_weights = hist_weights(myp1, myp2, myz, myzbins)
        myivar = myivar * h_weights[:, None]

    if rpix:
        """ Restrict wavelength coverage in observer frame """
        outfile += '_rpix'
        myivar[(lObs < cutoff[0]) | (lObs > cutoff[1])] = 0

    # 4. Distort spectra to remove power-law variations
    if distort:
        outfile += '_distort'
        
        CenAlpha = np.median(myalpha)
        #CenAlpha = -1.8
        distortMat = np.array([(qso.wl / 1450.) ** ele for ele in (CenAlpha - myalpha)])
        
        myspec *= distortMat
        myivar /= distortMat ** 2
        
        print('All spectra distorted to alpha:', CenAlpha)

    # C. CALIBRATION VS ESTIMATION ------------------------------------------------------
    if task == 'calibrate':
        Lind = (myz > 1.6) & (myz < 4)
        print('Number of spectra used for calibration are: %d' %Lind.sum())
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
        res = None
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
            
            res = mcmc_skewer.mcmcSkewer([np.array([zMat, myspec, myivar]).T, skewer_index], **kwargs)
            #return zMat, myspec, myivar
        stop = timer()
        print('Time elapsed:', stop - start)

        os.chdir(currDir)
        return res

# EOF
