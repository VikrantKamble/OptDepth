import numpy as np
import emcee
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import binned_statistic
from getdist import plots, MCSamples
from astroML.plotting.mcmc import convert_to_stdev as cts

from Scripts import helper

# global parameters
binx = np.arange(2.0, 4.61, 0.05)
centers = (binx[1:] + binx[:-1]) / 2.


# function to return std deviation using bootstrapping
def sig_func(a, threshold=20, nboot=200):
    if len(a) < threshold:
        return -1
    else:
        a_boot = np.random.choice(a, replace=True, size=(nboot, len(a)))
        return np.std(np.mean(a_boot, axis=1))


# LIKELIHOOD DEFINATIONS
def simpleln(theta, xi, yi, ei):
    """ Simple chisquare model with only measurement
    uncertainties """
    f0, t0, gamma = theta
    if 0 <= f0 < 3 and -14 <= t0 < 4 and -2 <= gamma < 9:
        model = f0 * np.exp(-np.exp(t0) * (1 + xi) ** gamma)
        return -0.5 * np.sum((yi - model) ** 2 / ei ** 2)
    return -np.inf


def lnlike1(theta, xi, yi, ei):
    """ Model with constant LSS variance across redshift """
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
            var = 0.065 * ((1 + xi) / 3.25) ** 3.8 * model ** 2
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
lsq = lambda *args: -simpleln(*args)
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
shift = np.array([-5.27, 3.21])
tilt = np.array([[-0.8563,  0.5165], [0.5165,  0.8563]])

# x0, x1 = np.mgrid[-1.16:1.84:200j, -0.13:0.07:200j]
x0, x1 = np.mgrid[-7:10:200j, -0.25:0.25:200j]

# Changed this
x0_line = np.linspace(-7, 10, 200)
x1_line = np.linspace(-0.25, 0.25, 200)

# x0_line = np.linspace(-1, 2, 200)
# x1_line = np.linspace(-0.18, 0.02, 200)

origPos = np.vstack([x0.ravel(), x1.ravel()])
modPos = np.dot(np.linalg.inv(tilt), origPos).T + shift


def mcmcSkewer(bundleObj, logdef=3, binned=False, niter=2500, do_mcmc=True,
               return_sampler=False,
               evalgrid=True, in_axes=None, viz=False, VERBOSITY=False,
               seed=None, truths=[0.002, 3.8]):
    """
    Script to fit simple flux model on each restframe wavelength skewer

    Parameters:
    -----------
        bundleObj : A list of [z, f, ivar] with the skewer_index
        logdef : Which model to use
        niter : The number of iterations to run the mcmc (40% for burn-in)
        do_mcmc : Flag whether to perform mcmc
        plt_pts : Plot the data along with best fit from scipy and mcmc
        return_sampler : Whether to return the raw sampler  without flatchaining
        triangle : Display triangle plot of the parameters
        evalgrid : Whether to compute loglikelihood on a specified grid
        in_axes : axes over which to draw the plots
        xx_viz : draw marginalized contour in modifed space
        VERBOSITY : print extra information
        seed : how to seed the random state
        truths : used with logdef=4, best-fit values of tau0 and gamma

    Returns:
        mcmc_chains if return_sampler, else None
    """

    z, f, ivar = bundleObj[0].T

    ind = (ivar > 0) & (np.isfinite(f))
    z, f, sigma = z[ind], f[ind], 1.0 / np.sqrt(ivar[ind])
    # -------------------------------------------------------------------------
    # continuum flux estimate given a value of (tau0, gamma)
    if logdef == 4:
        if VERBOSITY:
            print('Continuum estimates using optical depth parameters:', truths)
        chisq4 = lambda *args: - outer(*truths)(*args)

        opt_res = minimize(chisq4, 1.5, args=(z, f, sigma), method='Nelder-Mead')
        return opt_res['x']

    if VERBOSITY:
        print('Carrying analysis for skewer', bundleObj[1])

    if logdef == 1:
        nll, names, labels, guess = chisq1, names1, labels1, guess1
        ndim, kranges, lnlike = 4, kranges1, lnlike1

    elif logdef == 2:
        nll, names, labels, guess = chisq2, names2, labels2, guess2
        ndim, kranges, lnlike = 5, kranges2, lnlike2

    elif logdef == 3:
        nll, names, labels, guess = chisq3, names3, labels3, guess3
        ndim, kranges, lnlike = 3, kranges3, lnlike3

    # Try to fit with scipy optimize routine
    opt_res = minimize(nll, guess, args=(z, f, sigma), method='Nelder-Mead')
    print('Scipy optimize results:')
    print('Success =',  opt_res['success'], 'params =', opt_res['x'], '\n')

    if viz:
        if in_axes is None:
            fig, in_axes = plt.subplots(1)
        in_axes.errorbar(z, f, sigma, fmt='o', color='gray', alpha=0.2)
        in_axes.plot(zline, opt_res['x'][0] * np.exp(-np.exp(opt_res['x'][1]) *
                     (1 + zline) ** opt_res['x'][2]))

    if binned:
        mu = binned_statistic(z, f, bins=binx).statistic
        sig = binned_statistic(z, f, bins=binx, statistic=sig_func).statistic

        ixs = sig > 0
        z, f, sigma = centers[ixs], mu[ixs], sig[ixs]

        if viz:
            in_axes.errorbar(z, f, sigma, fmt='o', color='r')

        nll, names, labels, guess = lsq, names3, labels3, guess3
        ndim, kranges, lnlike = 3, kranges3, simpleln

    # --------------------------------------------------------------------------
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

        # Production step
        sampler.run_mcmc(p0, niter)
        print("Burn-in and production completed \n")

        if return_sampler:
            return sampler.chain
        else:
            # pruning 40 percent of the samples as extra burn-in
            lInd = int(niter * 0.4)
            samps = sampler.chain[:, lInd:, :].reshape((-1, ndim))

            # using percentiles as confidence intervals
            CenVal = np.median(samps, axis=0)

            # print BIC at the best estimate point, BIC = - 2 * ln(L_0) + k ln(n)
            print('CHISQ_R', -2 * lnlike(CenVal, z, f, sigma) / (len(z) - 3))
            print('BIC:', -2 * lnlike(CenVal, z, f, sigma) + ndim * np.log(len(z)))

            # Rotate the points to the other basis and 1D estimates
            # and write them to the file

            # Format : center, top error, bottom error
            tg_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(samps, [16, 50, 84], axis=0))))

            xx = helper.xfm(samps[:, 1:], shift, tilt, dir='up')
            xx_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(xx, [16, 50, 84], axis=0))))

            f_name2 = 'tg_est_' + str(bundleObj[1]) + '.dat'
            np.savetxt(f_name2, tg_est)
            f_name3 = 'xx_est_' + str(bundleObj[1]) + '.dat'
            np.savetxt(f_name3, xx_est)

            if viz:
                in_axes.plot(zline, CenVal[0] * np.exp(-np.exp(CenVal[1]) *
                             (1 + zline) ** CenVal[2]), '-g')

            # instantiate a getdist object
            MC = MCSamples(samples=samps, names=names, labels=labels, ranges=kranges)

            # MODIFY THIS TO BE PRETTIER
            if viz:
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

                # Visualize the contour in modified space per skewer
                if viz:
                    fig, ax2 = plt.subplots(1)
                    ax2.contour(x0, x1, cts(logP), levels=[0.683, 0.955, ], colors='k')
                    ax2.axvline(xx_est[0][0] + xx_est[0][1])
                    ax2.axvline(xx_est[0][0] - xx_est[0][2])
                    ax2.axhline(xx_est[1][0] + xx_est[1][1])
                    ax2.axhline(xx_est[1][0] - xx_est[1][2])
                    ax2.set_xlabel(r'$x_0$')
                    ax2.set_ylabel(r'$x_1$')
                    plt.show()

                # fileName1: the log-probability evaluated in the tilted grid
                f_name1 = 'gridlnlike_' + str(bundleObj[1]) + '.dat'
                np.savetxt(f_name1, logP)


def addLogs(fname, npix=200, sfx_lst=None, mod_ax=None, get_est=False,
            basis='orig', orig_ax=None, orig_space=True, mycolor='k',
            mylabel='temp', ls='solid', individual=False, save=True, model=False,
            force=False, plot_marg=True):
    """
    Plots the log-likelihood surface for each skewer in a given folder

    Parameters:
    -----------
        fname : the path to the folder containing the files
        npix : # of grid points in modified space
        suffix_list : indices of the skewers to plot, None for all
        mod_ax : axes over which to draw the contours in modified space
        orig_ax : axes over which to draw the contours in original space
        orig_space : do conversions to original space?
        mycolor : edgecolor of the joint pdf contour (JPC)
        mylabel : label of the JPC
        ls : linestyle of JPC
        individual : whether to draw contours for individual skewers

    Returns:
    --------
        None
    """

    import glob
    import os
    from scipy.interpolate import RectBivariateSpline

    if not os.path.exists(fname):
        print('Oops! There is no such folder')
        return None

    currdir = os.getcwd()
    os.chdir(fname)

    try:
        if get_est:
            if basis == "orig":
                n_dim = 3
                e_lst = glob.glob('tg_est*')
                labels = [r"$f_0$", r"$\ln \tau_0$", r"$\gamma$"]
            else:
                n_dim = 2
                e_lst = glob.glob('xx_est*')
                labels = [r"$x_0$", r"$x_1$"]

            sfx = []

            # 2. pull the names from the files and read the data
            e_cube = np.empty((len(e_lst), n_dim, 3))
            for ct, ele in enumerate(e_lst):
                temp = str.split(ele, '_')
                sfx.append(int(temp[2][:-4]))

                e_cube[ct] = np.loadtxt(ele)

            # Sort the data according to the skewer index
            e_cube = np.array([ele for _, ele in sorted(zip(sfx, e_cube))])
            sfx = np.array(sfx)
            sfx.sort()

            # Plotting
            fig, axs = plt.subplots(nrows=n_dim, sharex=True, figsize=(9, 5))

            for i in range(n_dim):
                axs[i].errorbar(sfx, e_cube[:, i, 0], yerr=[e_cube[:, i, 2], e_cube[:, i, 1]],
                                fmt='.-', color='k', lw=0.6)
                axs[i].set_ylabel(labels[i])

            # Best-fit after modeling correlation matrix
            if model:
                loc0, sig0 = helper.get_corrfunc(e_cube[:, -2, 0], e_cube[:, -2, 1],
                                                 model=True, est=True)
                print(loc0, sig0)
                loc1, sig1 = helper.get_corrfunc(e_cube[:, -1, 0], e_cube[:, -1, 1],
                                                 model=True, est=True)
                print(loc1, sig1)

            plt.tight_layout()
            plt.show()
            os.chdir(currdir)
            return sfx, e_cube

        # Do all other things after get_est has been taken care of
        # Add pdf from all the skewers - bypass if already carried out
        # Read data from the files
        if not os.path.isfile('joint_pdf.dat') or force:
            f_lst = glob.glob('gridlnlike_*')

            d_cube = np.empty((len(f_lst), npix, npix))

            # Read the skewer number from file itself for now
            sfx = []
            for ct, ele in enumerate(f_lst):
                d_cube[ct] = np.loadtxt(ele)

                temp = str.split(ele, '_')
                sfx.append(int(temp[1][:-4]))

            # sort the data for visualization
            d_cube = np.array([ele for _, ele in sorted(zip(sfx, d_cube))])

            sfx = np.array(sfx)
            sfx.sort()

            # choose a specific subset of the skewers
            if sfx_lst is not None:
                ind = [(ele in sfx_lst) for ele in sfx]
                d_cube = d_cube[ind]
                sfx = sfx[ind]

            # joint pdf #######################################################
            joint_pdf = d_cube.sum(0)
            joint_pdf -= joint_pdf.max()
            if save:
                np.savetxt('joint_pdf.dat', joint_pdf)
        else:
            print("****** File already exists. Reading from it *******")
            joint_pdf = np.loadtxt('joint_pdf.dat')

        # simple point statistics in modified space
        if mod_ax is None:
            fig, mod_ax = plt.subplots(1)

        print("Modified space estimates:")
        res = helper.marg_estimates(x0_line, x1_line, joint_pdf,
                                    mod_ax, plot_marg, labels=["x_0", "x_1"])
        mu_x0, sig_x0, mu_x1, sig_x1 = res

        # Plotting individual + joint contour in likelihood space
        if individual:
            colormap = plt.cm.rainbow
            colors = [colormap(i) for i in np.linspace(0, 1, len(sfx))]
            for i in range(len(sfx)):
                CS = mod_ax.contour(x0, x1, cts(d_cube[i]), levels=[0.68, ], colors=(colors[i],))
                CS.collections[0].set_label(sfx[i])

        mod_ax.legend(loc='upper center', ncol=6)
        mod_ax.set_xlabel('$x_0$')
        mod_ax.set_ylabel('$x_1$')

        # 1. Find the appropriate ranges in tau0-gamma space
        corners = np.array([[mu_x0 - 5 * sig_x0, mu_x1 - 5 * sig_x1],
                           [mu_x0 - 5 * sig_x0, mu_x1 + 5 * sig_x1],
                           [mu_x0 + 5 * sig_x0, mu_x1 - 5 * sig_x1],
                           [mu_x0 + 5 * sig_x0, mu_x1 + 5 * sig_x1]
                            ])
        extents = helper.xfm(corners, shift, tilt, dir='down',)

        extent_t0 = [extents[:, 0].min(), extents[:, 0].max()]
        extent_gamma = [extents[:, 1].min(), extents[:, 1].max()]

        # suitable ranges for spline interpolation in modified space
        range_stats = np.array([mu_x0 - 5 * sig_x0, mu_x0 + 5 * sig_x0,
                                mu_x1 - 5 * sig_x1, mu_x1 + 5 * sig_x1])

        mask_x0 = np.where((x0_line > range_stats[0]) &
                           (x0_line < range_stats[1]))[0]
        mask_x1 = np.where((x1_line > range_stats[2]) &
                           (x1_line < range_stats[3]))[0]

        # create a rectbivariate spline in the modified space logP
        _b = RectBivariateSpline(x0_line[mask_x0], x1_line[mask_x1],
                                 joint_pdf[mask_x0[:, None], mask_x1])

        # Rectangular grid in original space
        _tau0, _gamma = np.mgrid[extent_t0[0]:extent_t0[1]:500j,
                                 extent_gamma[0]:extent_gamma[1]:501j]

        _point_orig = np.vstack([_tau0.ravel(), _gamma.ravel()]).T
        _grid_in_mod = helper.xfm(_point_orig, shift, tilt, dir='up')

        values_orig = _b.ev(_grid_in_mod[:, 0], _grid_in_mod[:, 1])
        values_orig = values_orig.reshape(_tau0.shape)

        # Best fit + statistical errors
        print("Original space estimates:")
        if orig_ax is None:
            fig, orig_ax = plt.subplots(1)
        helper.marg_estimates(_tau0[:, 0], _gamma[0], values_orig,
                              orig_ax, plot_marg, labels=[r"\ln \tau_0", "\gamma"])

        plt.show()
    except Exception as ex:
        os.chdir(currdir)
        raise

    os.chdir(currdir)

if __name__ == "__main__":
    pass
