import numpy as np
import matplotlib.pyplot as plt
import fitsio
import numpy.ma as ma
import matplotlib.gridspec as gridspec
from scipy.interpolate import RectBivariateSpline
from astropy.convolution import convolve, Box1DKernel
from astroML.plotting.mcmc import convert_to_stdev as cts

# local imports
from Scripts.mcmc_skewer import xfm, marg_estimates

shift = np.array([-5.27, 3.21])
tilt = np.array([[-0.8563,  0.5165], [0.5165,  0.8563]])


def contour_transform(x0, x1, joint_pdf):
    """
    Convert a 2D likelihood contour from modified to original space
    along with marginalized statistics

    Paramters:
    x0 : grid points in first dimension
    x1 : grid points in second dimension
    joint_pdf : posterior probability over the grid
    """
    mu_x0, sig_x0, mu_x1, sig_x1 = marg_estimates(x0, x1, joint_pdf)

    # Convert the convert to original space
    corners = np.array([[mu_x0 - 5 * sig_x0, mu_x1 - 5 * sig_x1],
                       [mu_x0 - 5 * sig_x0, mu_x1 + 5 * sig_x1],
                       [mu_x0 + 5 * sig_x0, mu_x1 - 5 * sig_x1],
                       [mu_x0 + 5 * sig_x0, mu_x1 + 5 * sig_x1]
                        ])

    extents = xfm(corners, shift, tilt, dir='down')

    extent_t0 = [extents[:, 0].min(), extents[:, 0].max()]
    extent_gamma = [extents[:, 1].min(), extents[:, 1].max()]

    # suitable ranges for spline interpolation in modified space
    range_stats = np.array([mu_x0 - 5 * sig_x0, mu_x0 + 5 * sig_x0,
                            mu_x1 - 5 * sig_x1, mu_x1 + 5 * sig_x1])

    x0_line, x1_line = x0[:, 0], x1[0]
    mask_x0 = np.where((x0_line > range_stats[0]) & (x0_line < range_stats[1]))[0]
    mask_x1 = np.where((x1_line > range_stats[2]) & (x1_line < range_stats[3]))[0]

    # create a rectbivariate spline in the modified space
    _b = RectBivariateSpline(x0_line[mask_x0], x1_line[mask_x1],
                             cts(joint_pdf[mask_x0[:, None], mask_x1]))

    # Rectangular grid in original space
    tau0, gamma = np.mgrid[extent_t0[0]:extent_t0[1]:250j,
                           extent_gamma[0]:extent_gamma[1]:250j]

    _point_orig = np.vstack([tau0.ravel(), gamma.ravel()]).T
    _grid_in_mod = xfm(_point_orig, shift, tilt, dir='up')

    values_orig = _b.ev(_grid_in_mod[:, 0], _grid_in_mod[:, 1])
    values_orig = values_orig.reshape(tau0.shape)

    return tau0, gamma, values_orig


def plot_ellipse(pos, cov, nsig=[1], ax=None, fc='none',
                 ec=[0, 0, 0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are
    passed on to the ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
    """

    from scipy.stats import chi2
    from matplotlib.patches import Ellipse
    from scipy.special import erf

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    kwrg = {'facecolor': fc, 'edgecolor': ec, 'alpha': a, 'linewidth': lw}

    # Width and height are "full" widths, not radius
    for ele in nsig:
        scale = np.sqrt(chi2.ppf(erf(ele / np.sqrt(2)), df=2))
        width, height = 2 * scale * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

        ax.add_artist(ellip)
        ellip.set_clip_box(ax.bbox)

    # Limit the axes correctly to show the plots
    ax.set_xlim(pos[0] - 2 * width, pos[0] + 2 * width)
    ax.set_ylim(pos[1] - 2 * height, pos[1] + 2 * height)


def gaussfit_2d(X, C):
    from numpy.linalg import inv, det
    import emcee

    # Joint LogLikelihood to maximize
    def lnlike(theta, data, covar):
        loc, lna, corr, lnc = theta[0:2], *theta[2:]
        if -1 < corr < 1 and -10 < lna < 5 and -10 < lnc < 5:
            b = corr * np.sqrt(np.exp(lna) * np.exp(lnc))
            modC = covar + np.array([[np.exp(lna), b], [b, np.exp(lnc)]])
            temp = [np.dot(loc - data[i], np.dot(inv(modC[i]), loc - data[i])) + np.log(det(modC[i])) for i in range(len(data))]
            foo = - 0.5 * np.sum(temp, 0)
            return foo
        return -np.inf

    nwalkers, ndim = 100, 5

    init_cov = np.mean(C, 0)
    init_params = list(np.mean(X, 0)) + [np.log(init_cov[0, 0]), 0.5, np.log(init_cov[1, 1])]
    print('Using this initial guess:', init_params)

    p0 = [init_params + 1e-4 * np.random.randn(5) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(X, C))
    sampler.run_mcmc(p0, 1000)

    samples = sampler.chain[:, 500:, :].reshape(-1, 5)
    return samples


def plotcomp(myfile, suffix=None, nskip=1, conf_int=False,
             ratio_range=[1600, 1800], alpha_fit=None):
    """
    Helper routine to plot composites

    Parameters:

        myfile : input file containing the composites
        suffix : name of the file to save the plot to, if not None
        nskip : plot every nskip_th composite
        conf_int : draw 1-sig band for each composite
        ratio_range : the range over which delta_alpha is defined
        alpha_fit : draw a power-law spectral with this spectral index

    Returns:
        None
    """
    try:
        f = fitsio.FITS(myfile)
        comp, ivar = f[0].read(), f[1].read()
        comp_mask = ma.masked_array(comp, ivar <= 0)
        z = f[2]['REDSHIFT'][:]
        wl = f[3]['WAVELENGTH'][:]

        mean_comp = np.mean(comp_mask, 0)
        mean_comp = comp_mask[0]

        ixr = (wl > ratio_range[0]) & (wl < ratio_range[1])

        plt.figure(figsize=(6, 4.4))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        gs.update(hspace=0.0)

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        # ax1.set_color_cycle(sns.hls_palette(8, l=.3, s=.8))
        # http://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial

        for i in range(len(comp))[::nskip]:
            if conf_int:
                ixs = (ivar[i] > 0)
                sig = 1.0 / np.sqrt(ivar[i][ixs])
                ax1.fill_between(wl[ixs], comp[i][ixs] + sig, comp[i][ixs] - sig,
                                 color='gray', alpha=0.8)
            ax1.plot(wl, comp_mask[i], lw=0.5, label=r'$z = %.2f$' % z[i])
            ax2.plot(wl[wl > 1230], convolve(comp_mask[i][wl > 1230] / mean_comp[wl > 1230],
                     Box1DKernel(5)), lw=0.5)

            ratio = np.mean(comp_mask[i][ixr] / mean_comp[ixr])
            print(np.log10(ratio) / np.log10(1700 / 1450.))

        if alpha_fit is not None:
            ax1.plot(wl, (wl / 1450.) ** alpha_fit, '-g')

        ax1.get_xaxis().set_visible(False)
        ax1.set_ylabel(r'$F_\lambda (\mathrm{Arbitrary\ Units})$')
        ax1.set_xlim(1000, 2200)
        ax1.legend(ncol=3, frameon=False)
        ax1.set_ylim(0.2, 4)

        ax3.plot(wl[wl > 1240], np.std(comp_mask[:, wl > 1240], 0), lw=0.3, c='k')

        ax2.get_xaxis().set_visible(False)
        ax3.set_ylabel(r'$\sigma$')
        ax3.set_ylim(0, 0.04)
        ax3.set_yticks(np.arange(0, 0.06, 0.02))

        # ax.set_color_cycle(sns.hls_palette(8, l=.3, s=.8))
        ax2.set_ylabel(r'$\mathrm{ratio}$')
        ax2.set_ylim(0.93, 1.07)

        ax3.set_xlabel(r'$\lambda_{\mathrm{rf}}$')

        plt.tight_layout()
        plt.legend(ncol=2)
        if suffix is not None:
            plt.savefig(suffix+'.pdf', rasterized=True)
        plt.show()
        return ax1, ax2, ax3
    except Exception as e:
        raise


def other_plot(loc, cov):
    """
    Plots the optical-depth results from other studies

    Parameters:
        loc : the best-fit from current work
        cov : the covariance matrix of the estimates

    Returns:
        None
    """
    x = np.linspace(2, 5, 100)

    def log_trans(x, a, b, t0_in_log=True):
        if t0_in_log:
            a = np.exp(a)
        return a * (1 + x) ** b

    # best_fit and 1-sigma band from this Work
    pts = np.random.multivariate_normal(loc, cov, size=500)
    d_pts = np.array([log_trans(x, *pt) for pt in pts])
    my_avg, my_err = d_pts.mean(0), d_pts.std(0)

    plt.figure()
    plt.plot(x, my_avg, c='k', label='This Work')
    plt.fill_between(x, my_avg + my_err, my_avg - my_err, color='gray', alpha=0.5)

    # For Kim
    gamma_kim, tau_0_kim = 3.65, 0.0023
    plt.plot(x, log_trans(x, tau_0_kim, gamma_kim, t0_in_log=False), ls='dashed', c='g',
             label='Kim et. al. (2007)')

    # For Kirkmann
    gamma_s, tau_0_s = 4.164, 10 ** (-2.910)
    plt.plot(x, log_trans(x, tau_0_s, gamma_s, t0_in_log=False), ls='dotted',
             label='Kirkman et. al. (2005)')

    # For Scheye
    gamma_s, tau_0_s = 4.057, 10 ** (-2.853)
    plt.plot(x, log_trans(x, tau_0_s, gamma_s, t0_in_log=False), ls='dashdot',
             c='black', label='Scheye et. al. (2003)')

    # For Faucher
    z_data = np.arange(2., 4.3, 0.2)
    f_data = [0.146, 0.187, 0.229, 0.280, 0.360, 0.423, 0.452, 0.615, 0.766,
              0.885, 0.986, 1.113]
    err_data = [0.023, 0.02, 0.019, 0.022, 0.026, 0.029, 0.031, 0.039, 0.051,
                0.051, 0.064, 0.0121]
    gamma_f, tau_0_f = 3.92, 0.0018
    plt.errorbar(z_data, f_data, err_data, fmt='.', color='r', capsize=2)

    plt.plot(x, log_trans(x, tau_0_f, gamma_f, t0_in_log=False), ls='dashdot',
             c='r', label='Faucher-Giguere et. al. (2008c)')

    # For Becker 2012
    z_data = np.arange(2.15, 4.95, 0.1)
    f_data = [0.8806, 0.8590, 0.8304, 0.7968, 0.7810, 0.7545, 0.7371, 0.7167,
              0.6966, 0.6670, 0.6385, 0.6031, 0.5762, 0.5548, 0.5325, 0.4992,
              0.4723, 0.4470, 0.4255, 0.4030, 0.3744, 0.3593, 0.3441, 0.3216,
              0.3009, 0.2881, 0.2419, 0.2225]
    err_data = [0.0103, 0.0098, 0.0093, 0.0089, 0.0090, 0.0088, 0.0088, 0.0086,
                0.0084, 0.0082, 0.0080, 0.0079, 0.0074, 0.0071, 0.0071, 0.0069,
                0.0068, 0.0072, 0.0071, 0.0071, 0.0074, 0.0075, 0.0102, 0.0094,
                0.0104, 0.0117, 0.0201, 0.0151]

    z0, beta, C, tau0 = 3.5, 2.90, -0.132, 0.751
    plt.plot(x, tau0 * (((1+x)/(1+z0)) ** beta) + C, ls='dashed', c='orange',
             label='Becker et. al. (2012)')

    plt.ylabel(r'$\tau_{\mathrm{eff}}(z)$')
    plt.xlabel(r'$z$')
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()


# num_bins = 7
# bin_colors = sns.hls_palette(7, l=.3, s=.8)


# import mcmc_skewer
# import importlib

# importlib.reload(mcmc_skewer)


# def bin_plot(ax1, ax2):
#     """ Plots optical depth measurements for all the bins
#     """

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_1_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[0], ls='dashed', mylabel='Bin 1')

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_2_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[1], ls='dashed', mylabel='Bin 2')

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_3_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[2], ls='dashed', mylabel='Bin 3')

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_4_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[3], ls='dashed', mylabel='Bin 4')

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_5_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[4], ls='dashed', mylabel='Bin 5')

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_6_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[5], ls='dashed', mylabel='Bin 6')

#     # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_7_rpix_distort1070_1160/',
#     #                     individual=False, orig_ax=ax2, mod_ax=ax1,
#     #                     mycolor=bin_colors[6], ls='dashed', mylabel='Bin 7')

#     fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))
#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin1_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[0], ls='dashed', mylabel='Bin 1')

#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin2_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[1], ls='dashed', mylabel='Bin 2')

#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin3_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[2], ls='dashed', mylabel='Bin 3')

#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin4_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[3], ls='dashed', mylabel='Bin 4')

#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin5_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[4], ls='dashed', mylabel='Bin 5')

#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin6_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[5], ls='dashed', mylabel='Bin 6')

#     mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin7_highsn_distort_distort1070_1160/',
#                         individual=False, orig_ax=ax2, mod_ax=ax1,
#                         mycolor=bin_colors[6], ls='dashed', mylabel='Bin 7')

#     return ax1, ax2


def fancy_scatter(x, y, values=None, bins=60, names=['x', 'y'], marg=False):
    """ Scatter plot of paramters with number desnity contours
    overlaid. Marginalized 1D distributions also plotted.

    Make sure that the data is appropriately cleaned before using
    this routine
    """
    # Simple cuts to remove bad-points

    from scipy.stats import binned_statistic_2d as bs2d

    ixs = np.isfinite(x) & np.isfinite(y)

    range1 = np.percentile(x[ixs], [5, 95])
    range2 = np.percentile(y[ixs], [5, 95])

    # Use interquartile ranges to suitably set the ranges for the bins
    width1, width2 = np.diff(range1), np.diff(range2)

    if values is None:
        res = bs2d(x[ixs], y[ixs], None, 'count', bins,
                   range=[[range1[0] - width1, range1[1] + width1],
                          [range2[0] - width2, range2[1] + width2]])

    else:
        res = bs2d(x[ixs], y[ixs], values[ixs], 'mean', bins,
                   range=[[range1[0] - width1, range1[1] + width1],
                          [range2[0] - width2, range2[1] + width2]])

    if not marg:
        fig, ax2d = plt.subplots(1, figsize=(6, 5))
    else:
        fig = plt.figure(figsize=(6, 6))
        ax2d = fig.add_subplot(223)
        ax1 = fig.add_subplot(221, sharex=ax2d)
        ax3 = fig.add_subplot(224, sharey=ax2d)

    if values is None:
        # plot the 2d histogram
        ax2d.imshow(np.log10(res.statistic.T), origin='lower', extent=[res.x_edge[0],
                    res.x_edge[-1], res.y_edge[0], res.y_edge[-1]], aspect='auto',
                    cmap=plt.cm.binary, interpolation='nearest')

        # overlay contours
        levels = np.linspace(0, np.log(res.statistic.max()), 10)[2:]
        ax2d.contour(np.log(res.statistic.T), levels, colors='k',
                     extent=[res.x_edge[0], res.x_edge[-1],
                     res.y_edge[0], res.y_edge[-1]])
    else:
        # plot the sinned statistic image
        cmap_multicolor = plt.cm.jet
        cmap_multicolor.set_bad('w', 1.)

        cs = ax2d.imshow(res.statistic.T, origin='lower', extent=[res.x_edge[0],
                         res.x_edge[-1], res.y_edge[0], res.y_edge[-1]], aspect='auto',
                         cmap=cmap_multicolor, interpolation='nearest')
        cb = plt.colorbar(cs)
        cb.set_clim(np.nanpercentile(res.statistic, 50),
                    np.nanpercentile(res.statistic, 60))

    ax2d.set_xlabel(r'{}'.format(names[0]))
    ax2d.set_ylabel(r'{}'.format(names[1]))

    if marg:
        ax1.hist(x, bins=res.x_edge, histtype='step', range=range1, color='k')
        ax3.hist(y, bins=res.y_edge, histtype='step', range=range2, color='k',
                 orientation='horizontal')

        ax1.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)

        ax1.set_ylabel(r'$\mathrm{N}$', fontsize=16)
        ax3.set_xlabel(r'$\mathrm{N}$', fontsize=16)

        # Put percentile cuts on the marginalized plots
        p1_cuts = np.percentile(x[ixs], [10, 90])
        p2_cuts = np.percentile(y[ixs], [10, 90])

        for ele in np.linspace(*p1_cuts, 4):
            ax2d.axvline(ele, color='r', linewidth=0.6)
        for ele in np.linspace(*p2_cuts, 4):
            ax2d.axhline(ele, color='r', linewidth=0.6)

        # print the bins along each dimension along with the count in each bin
        print(names[0], np.linspace(*p1_cuts, 4))
        print(names[1], np.linspace(*p2_cuts, 4))

        bin_counts = bs2d(x[ixs], y[ixs], None, 'count',
                          bins=[np.linspace(*p1_cuts, 4),
                          np.linspace(*p2_cuts, 4)]).statistic
        print(bin_counts)

    plt.tight_layout()
    plt.show()
