from scipy.optimize import minimize
import numpy as np
import importlib
import emcee
import matplotlib.pyplot as plt

from getdist import plots, MCSamples
from astroML.plotting.mcmc import convert_to_stdev as cts
import plotting

importlib.reload(plotting)


# LIKELIHOOD DEFINATIONS
def lnlike1(theta, xi, yi, ei):
    """ Model with constant LSS variance across redshift"""
    f0, t0, gamma, sigma = theta
    # Constant prior on tau_0 in log-space
    if 0 <= f0 < 3 and -14 <= t0 < 4 and -2 <= gamma < 9 and 0 < sigma < 0.6:
        model = f0 * np.exp(-np.exp(t0) * (1 + xi) ** gamma)
        return -0.5 * np.sum((yi - model) ** 2 / (ei ** 2 + sigma ** 2) +
                             np.log(ei ** 2 + sigma ** 2), 0)
    return -np.inf


def lnlike2(theta, xi, yi, ei):
    """ Model with LSS variance modeled by a power law as a
    function of redshift"""
    f0, t0, gamma, var_a, var_b = theta

    if 0 <= f0 < 3 and -10 <= t0 < -2 and 1 <= gamma < 7 and \
       0.001 <= var_a < 1 and 1 <= var_b < 5:

        model = f0 * np.exp(-np.exp(t0) * (1 + xi) ** gamma)

        # Model for LSS variance - Lee et. al.
        var = var_a * ((1 + xi)/3.25) ** var_b * model ** 2
        return -0.5 * np.sum((yi - model) ** 2 / (ei ** 2 + var) +
                             np.log(ei ** 2 + var))
    return -np.inf


def lnlike3(theta, xi, yi, ei):
    # Model with LSS variance parameterized by a power law as a
    # function of redshift but parameters fixed
    f0, t0, gamma = theta
    if 0 <= f0 < 3 and -14 <= t0 < 4 and -2 <= gamma < 9:
        model = f0 * np.exp(-np.exp(t0) * (1 + xi) ** gamma)

        # Model for LSS variance - Lee et. al.
        var = 0.065 * ((1 + xi)/3.25) ** 3.8 * model ** 2
        return -0.5 * np.sum((yi - model) ** 2 / (ei ** 2 + var) +
                             np.log(ei ** 2 + var))
    return -np.inf


def outer(tau0, gamma):
    def lnback(f0, xi, yi, ei):
        """ Unabsorbed continuum estimates for each bin using the
        best fit optical depth parameters
        """
        if 0 < f0 < 5:
            model = f0 * np.exp(-tau0 * (1 + xi) ** gamma)

            # Model for LSS variance
            var = 0.065 * ((1 + xi)/3.25) ** 3.8 * model ** 2

            return -0.5 * np.sum((yi - model) ** 2 / (ei ** 2 + var) +
                                 np.log(ei ** 2 + var))
        return -np.inf
    return lnback


zline = np.linspace(1.8, 5, 100)

# Initial guess
guess1 = [1.5, -6, 3.8, 0.1]
guess2 = [1.5, -8, 6, 0.065, 3.8]
guess3 = [1.5, -6, 3.8]

# function defination for scipy optimize
chisq1 = lambda *args: -lnlike1(*args)
chisq2 = lambda *args: -lnlike2(*args)
chisq3 = lambda *args: -lnlike3(*args)


# Configure GetDist
names1 = ["f0", "t0", "gamma", "sigma"]
labels1 = ["f_0", r"\tau_0", "\gamma", "\sigma"]
kranges1 = {'f0': (0, 3), 't0': (-14, 4), 'gamma': (-2, 9), 'sigma': (0, 0.6)}

names2 = ["f0", "t0", "gamma", "sigma_a", "sigma_b"]
labels2 = ["f_0", r"\tau_0", "\gamma", "\sigma_a", "\sigma_b"]
kranges2 = {'f0': (0, 3), 't0': (-10, -2), 'gamma': (1, 7),
            'sigma_a': (0.001, 1), 'sigma_b': (1, 5)}

names3 = ["f0", "t0", "gamma"]
labels3 = ["f_0", r"\tau_0", "\gamma"]
kranges3 = {'f0': (0, 3), 't0': (-14, 4), 'gamma': (-2, 9)}

# Create a grid - delegate this to the inputs of this function
shift = np.array([-5.0625, 3.145])
tilt = np.array([[-0.85627484,  0.51652047],
                [0.51652047,  0.85627484]])

x0, x1 = np.mgrid[-7:10:200j, -0.25:0.25:200j]

x0_line = np.linspace(-7, 10, 200)
x1_line = np.linspace(-0.25, 0.25, 200)

origPos = np.vstack([x0.ravel(), x1.ravel()])
modPos = np.dot(np.linalg.inv(tilt), origPos).T + shift


def mcmcSkewer(bundleObj, logdef=3, niter=2500, do_mcmc=True, plotit=False,
               return_sampler=False, triangle=False, evalgrid=True,
               visualize=False, VERBOSITY=False, seed=None, true=[0.002, 3.8]):
    """
    Script to fit simple flux model on each restframe wavelength skewer

    Parameters:
        bundleObj: A list of [z, f, ivar] with the skewer_index
        logdef: Which model to use
        niter: The number of iterations to run the mcmc (500 for burn-in fixed)
        do_mcmc: Flag whether to perform mcmc
        plotit: Plot the data along with best fit from scipy and mcmc
        return_sampler: Whether to return the raw sampler  without flatchaining
        triangle: Display triangle plot of the parameters
        evalgrid: Whether to compute loglikelihood on a specified grid

    Returns:
        mcmc_chains if return_sampler, else None
    """
    print('Carrying analysis for skewer', bundleObj[1])
    z, f, ivar = bundleObj[0].T

    ind = (ivar > 0) & (np.isfinite(f))
    z, f, sigma = z[ind], f[ind], 1.0 / np.sqrt(ivar[ind])

    # -------------------------------------------------------------------------
    if logdef == 4:
        chisq4 = lambda *args: -outer(*true)(*args)

        opR = minimize(chisq4, 1.5, args=(z, f, sigma),
                       method='Nelder-mead')
        return opR['x']

    elif logdef == 1:
        nll, names, labels, guess = chisq1, names1, labels1, guess1
        ndim, kranges, lnlike = 4, kranges1, lnlike1

    elif logdef == 2:
        nll, names, labels, guess = chisq2, names2, labels2, guess2
        ndim, kranges, lnlike = 5, kranges2, lnlike2

    elif logdef == 3:
        nll, names, labels, guess = chisq3, names3, labels3, guess3
        ndim, kranges, lnlike = 3, kranges3, lnlike3
    # --------------------------------------------------------------------------
    # Try to fit with scipy optimize routine
    opR = minimize(nll, guess, args=(z, f, sigma), method='Nelder-mead')

    if VERBOSITY:
        print('Scipy optimize results:')
        print('Success =',  opR['success'], 'params =', opR['x'], '\n')

    if plotit:
        fig, ax1 = plt.subplots(1)
        ax1.errorbar(z, f, sigma, fmt='o', alpha=0.2)
        ax1.plot(zline, opR['x'][0] * np.exp(-np.exp(opR['x'][1]) *
                 (1 + zline) ** opR['x'][2]), '-r')

    if do_mcmc:
        np.random.seed()

        nwalkers = 100
        p0 = [guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

        # configure the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnlike, args=(z, f, sigma))

        # burn-in time - Is this enough?
        p0, __, __ = sampler.run_mcmc(p0, 500)
        sampler.reset()

        sampler.run_mcmc(p0, niter)
        print("Burn-in and Sampling completed \n")

        if return_sampler:
            return sampler.chain
        else:
            # pruning 40 percent of the samples as extra burn-in
            lInd = int(niter * 0.4)
            samps = sampler.chain[:, lInd:, :].reshape((-1, ndim))

            # using percentiles as confidence intervals
            CenVal = np.median(samps, axis=0)
            # print BIC at the best estimate point, BIC = - 2 * ln(L_0) + k ln(n)
            print('BIC:', -2 * lnlike(CenVal, z, f, sigma) + ndim * np.log(len(z)))

            estimates = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samps, [16, 50, 84], axis=0))))

            if VERBOSITY:
                for count, ele in enumerate(names):
                    print(ele + ' = %.3f^{%.3f}_{%.3f}' %(estimates[count][0], estimates[count][1], estimates[count][2]))

            if plotit:
                ax1.plot(zline, CenVal[0] * np.exp(-np.exp(CenVal[1]) * (1 + zline) ** CenVal[2]), '-g')

            # instantiate a getdist object 
            MC = MCSamples(samples=samps, names=names, labels=labels, ranges=kranges)

            # MODIFY THIS TO BE PRETTIER
            if triangle:
                g = plots.getSubplotPlotter()
                g.triangle_plot(MC)

            # Evaluate the pdf on a rotated grid for better estimation
            if evalgrid:
                print('Evaluating on the grid specified \n')
                pdist = MC.get2DDensity('t0', 'gamma')

                # Evalaute density on a grid
                pgrid = np.array([pdist.Prob(*ele) for ele in modPos])
                # Prune to remove negative densities
                pgrid[pgrid < 0] = 1e-50

                # Convert to logLikelihood
                logP = np.log(pgrid)
                logP -= logP.max()
                logP = logP.reshape(x0.shape)

                # Visualize
                if visualize:
                    fig, ax2 = plt.subplots(1, figsize=(5,5))
                    ax2.contour(x0, x1, cts(logP), levels=[0.683, 0.955, ], colors='k')
                    ax2.set_xlabel(r'$x_0$')
                    ax2.set_ylabel(r'$x_1$')
                    plt.show()

                # fileName1: the log-probability evaluated in the tilted grid
                fileName1 = 'gridlnlike_' + str(bundleObj[1]) + '.dat'
                np.savetxt(fileName1, logP)
                # fileName2: the estimates of f0, ln_t0, gamma, and sigma from MCMC 
                fileName2 = 'estimates_' + str(bundleObj[1]) + '.dat'
                np.savetxt(fileName2, estimates)
                # fileName3: the parameters of 2D gaussian fit to ln_t0-gamma plane
                fileName3 = 'fitparams_' + str(bundleObj[1]) + '.dat'
                fitparams = list(np.mean(samps[:, 1:3], 0)) + \
                                list(np.cov(samps[:, 1:3].T).flat)
                np.savetxt(fileName3, fitparams)



def gauss_like(theta, X, C):
    from numpy.linalg import inv, det
    # define the log-likelihood of the data
    loc, a, corr, c = theta[0:2], np.exp(theta[2]), theta[3], np.exp(theta[4])
    if -1 < corr < 1:
        b = corr * np.sqrt(a * c)
        modC = C + np.array([[a, b],[b, c]])
        temp = [np.dot(loc - X[i], np.dot(inv(modC[i]), loc - X[i])) + np.log(det(modC[i])) for i in range(len(X))]
        foo = 0.5 * np.sum(temp, 0)
        return foo
    return np.inf


def addLogs(fname, npix=200, suffix_list=None, mod_ax=None, orig_ax=None, orig_space=True, mycolor='k', individual=True, getsys=False):
    """Plot the log-likelihood surface for each skewer
    Input:
        fname: the path to the folder containing the files

    Returns:
        None
        """
    import corner
    import glob
    import os
    import time
    from plotting import plot_cov_ellipse
    from scipy.interpolate import RectBivariateSpline

    if not os.path.exists(fname):
        print('Oops! There is no such folder')
        return None

    currdir = os.getcwd()
    os.chdir(fname)

    try:
        # Read data from the files
        file_list = glob.glob('gridlnlike*')
        data_cube = np.empty((len(file_list), npix, npix))

        suffix = []
        for count, ele in enumerate(file_list):
            data_cube[count] = np.loadtxt(ele)

            temp = str.split(ele, '_')
            suffix.append(int(temp[1][:-4]))

        if suffix_list is not None:
            ind = [i for i, ele in enumerate(suffix) if ele in suffix_list]
            data_cube = data_cube[ind]
            suffix = np.array(suffix)[ind]

        # sort for visualization in terms of restframe wavelength indices
        data_cube = np.array([ele for _,ele in sorted(zip(suffix, data_cube))])
        suffix.sort()

        # joint pdf ###########################################################
        joint_pdf = np.sum(data_cube, axis=0)

        # Plot indivdual skewer contours along with the joint estimate
        if mod_ax is None:
            fig, mod_ax = plt.subplots(1)

        """ Routine to get the triangle plot in likelihood space
        It also compute the contour in the original space using a RectBivariateSpline
        interpolation over the data.
        """
        if individual:
            colormap = plt.cm.rainbow 
            colors = [colormap(i) for i in np.linspace(0, 1,len(suffix))]
            for i in range(len(suffix)):
                CS = mod_ax.contour(x0, x1, cts(data_cube[i]), levels=[0.683, ], colors=(colors[i],))
                CS.collections[0].set_label(suffix[i])

        # Plotting individual + joint contour in likelihood space
        mod_ax.contour(x0, x1, cts(joint_pdf), levels=[0.683, 0.955], colors=(mycolor,), linestyles='--')
        mod_ax.legend(loc = 'upper center', ncol=6)
        mod_ax.set_xlabel('$x_0$')
        mod_ax.set_ylabel('$x_1$')

        # simple point statistics in modified space
        x0_pdf = np.sum(np.exp(joint_pdf), axis=1)
        x0_pdf /= x0_pdf.sum() * (x0_line[1] - x0_line[0])
        x1_pdf = np.sum(np.exp(joint_pdf), axis=0)
        x0_pdf /= x0_pdf.sum() * (x0_line[1] - x0_line[0])

        mu_x0 = (x0_line * x0_pdf).sum() / x0_pdf.sum()
        mu_x1 = (x1_line * x1_pdf).sum() / x1_pdf.sum()
        sig_x0 = np.sqrt((x0_line ** 2 * x0_pdf).sum() / x0_pdf.sum() - mu_x0 ** 2)
        sig_x1 = np.sqrt(np.sum(x1_line ** 2 * x1_pdf) / np.sum(x1_pdf) - mu_x1 ** 2)

        # 1. Find the appropriate ranges in tau0-gamma space
        corners = np.array([[mu_x0 - 5 * sig_x0, mu_x1 - 5 * sig_x1],
                    [mu_x0 - 5 * sig_x0, mu_x1 + 5 * sig_x1], 
                    [mu_x0 + 5 * sig_x0, mu_x1 - 5 * sig_x1], 
                    [mu_x0 + 5 * sig_x0, mu_x1 + 5 * sig_x1]
                    ])
        extents = get_transform(corners, dir='down')

        extent_t0 = [extents[:,0].min(), extents[:,0].max()]
        extent_gamma = [extents[:,1].min(), extents[:,1].max()]

        # suitable ranges for spline interpolation in modified space
        range_stats = np.array([mu_x0 - 5 * sig_x0, mu_x0 + 5 * sig_x0, mu_x1 - 5 * sig_x1, mu_x1 + 5 * sig_x1])
        mask_x0 = np.where((x0_line > range_stats[0]) & (x0_line < range_stats[1]))[0]
        mask_x1 = np.where((x1_line > range_stats[2]) & (x1_line < range_stats[3]))[0]

        # create a rectbivariate spline in the modified space
        _b = RectBivariateSpline(x0_line[mask_x0], x1_line[mask_x1], cts(joint_pdf[mask_x0[:,None], mask_x1]))

        # Rectangular grid in original space
        _tau0, _gamma = np.mgrid[extent_t0[0]:extent_t0[1]:100j, extent_gamma[0]:extent_gamma[1]:99j]
        _point_orig = np.vstack([_tau0.ravel(), _gamma.ravel()]).T
        _grid_in_mod = get_transform(_point_orig, dir='up')

        values_orig = _b.ev(_grid_in_mod[:,0], _grid_in_mod[:,1])
        values_orig = values_orig.reshape(_tau0.shape)
        
        if orig_ax is None:
            fig, orig_ax = plt.subplots(1)
        orig_ax.contour(_tau0, _gamma,  values_orig, levels=[0.668, 0.955], colors=(mycolor,))

        if get_estimates:
            labels = [r"$f_0$", r"$\ln \tau_0$", r"$\gamma$", r"$\sigma$"]

            mod_best = np.array([estimates_cube[:,1,0], estimates_cube[:,2,0]]).T
            mod_ax.scatter(mod_best[:,0], mod_best[:,1])

            orig_best = get_transform(mod_best, dir='up')
            orig_ax.scatter(orig_best[:,0], orig_best[:,1])
            # ax2[0].hist(estimates_cube[:,1,0])
            # ax2[1].hist(estimates_cube[:,2,0])

        plt.show()

        if getsys:
            # means and covariance of all the skewers
            # the assumption is that the data comes from a 2D gaussian 
            # is neglecting the effects of truncation valid or justifiable?
            X = Ps[:, 0:2]
            covMat = Ps[:, 2:].reshape(-1, 2, 2)
            # this isn't much helpful as the contours are strongly correlated
            [morph_gauss(X[i], covMat[i], ax=ax1, lw=0.6, a=0.6) for i in range(len(X))]

            # Delegate this to another function
            temp = plotting.gaussfit_2d(X, covMat)
            fit_means = np.median(temp[:, 0:2], 0)
            fit_cov = np.cov(temp[:, 0:2].T)

            morph_gauss(fit_means, fit_cov, ax=ax1, fc='r', ls='dashed', lw=2) 

            corner.corner(temp)
    except:
        os.chdir(currdir)
        raise

    os.chdir(currdir)


def morph_gauss(pos, cov, ax=None, shift=shift, tilt=tilt, fc='k', ec=[0,0,0], a=1, lw=2, ls='solid'):
    from scipy.special import erf
    from scipy.stats import chi2
    import numpy as np
    import matplotlib.pyplot as plt

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.arctan2(*vecs[:,0][::-1])

    scale = np.sqrt(chi2.ppf(erf(1 / np.sqrt(2)), df=2))
    phi = np.linspace(0, 2 * np.pi, 100)

    p1 = scale * np.sqrt(vals[0]) * np.cos(phi) 
    p2 = scale * np.sqrt(vals[1]) * np.sin(phi)

    x = pos[0] + p1 * np.cos(theta) - p2 * np.sin(theta)
    y = pos[1] + p1 * np.sin(theta) + p2 * np.cos(theta)
    points = np.vstack((x, y))

    modPos = np.dot(tilt, (points - shift[:, None]))

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(modPos[0], modPos[1], lw=lw, alpha=a, color=fc, ls=ls)
    plt.show()


def get_transform(pos, dir='up'):
    if np.array(pos).ndim == 1:
        pos = np.array(pos)[None, :]
    if dir == 'up':
        return np.dot((pos - shift), tilt.T)
    elif dir == 'down':
        return np.dot(np.linalg.inv(tilt), pos.T).T + shift



if __name__=="__main__":
    # fNames = ['3_rpix_distort_1060_1170', '4_rpix_distort_1060_1170', '5_rpix_distort_1060_1170', '1_rpix_distort_1060_1170', '2_rpix_distort_1060_1170', 'testcode9_1060_1170', 'testcode8_1060_1170']
    fNames = ['testcode9_1060_1170',]

    colormap = plt.cm.rainbow
    colors = [colormap(i) for i in np.linspace(0, 1,len(fNames))]

    fig, ax = plt.subplots(1)
    for count, ele in enumerate(fNames):
        curr_ele = '/uufs/astro.utah.edu/common/home/u0882817/Work/OptDepth/LogLikes/Bin_' + ele
        addLogs(curr_ele, ax=ax, individual=False, mycolor=colors[count])

    nlines = 50
    t0_bins = np.linspace(-14, 4, nlines)
    gamma_bins = np.linspace(-2, 9, nlines)

    for x in t0_bins:
        locX = np.vstack((x * np.ones(nlines), gamma_bins))
        locXprime = np.dot(tilt, locX - shift[:, None])
        ax.plot(locXprime[0], locXprime[1], 'r', lw=0.6)
    for y in gamma_bins:
        locX = np.vstack((t0_bins, (y * np.ones(nlines))))
        locXprime = np.dot(tilt, locX - shift[:, None])
        ax.plot(locXprime[0], locXprime[1], 'k', lw=0.6)