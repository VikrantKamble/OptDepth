import numpy as np
import matplotlib.pyplot as plt
import os
import fitsio
import numpy.ma as ma
import matplotlib.gridspec as gridspec
from scipy.interpolate import RectBivariateSpline
from astropy.convolution import convolve, Box1DKernel
from astroML.plotting.mcmc import convert_to_stdev as cts


import seaborn as sns


shift = np.array([-5.0625, 3.145])
tilt = np.array([[-0.85627484,  0.51652047],
                [0.51652047,  0.85627484]])


def marg_estimates(x0, x1, joint_pdf):
    """
    Marginalized statistics that follows from a jont likelihood
    """
    x0_line, x1_line = x0[:, 0], x1[0]

    x0_pdf = np.sum(np.exp(joint_pdf), axis=1)
    x0_pdf /= x0_pdf.sum() * (x0_line[1] - x0_line[0])
    x1_pdf = np.sum(np.exp(joint_pdf), axis=0)
    x0_pdf /= x0_pdf.sum() * (x0_line[1] - x0_line[0])

    mu_x0 = (x0_line * x0_pdf).sum() / x0_pdf.sum()
    mu_x1 = (x1_line * x1_pdf).sum() / x1_pdf.sum()
    sig_x0 = np.sqrt((x0_line ** 2 * x0_pdf).sum() / x0_pdf.sum() - mu_x0 ** 2)
    sig_x1 = np.sqrt(np.sum(x1_line ** 2 * x1_pdf) / np.sum(x1_pdf) - mu_x1 ** 2)

    print("%.4f pm %.4f" %(mu_x0, sig_x0))
    print("%.4f pm %.4f" %(mu_x1, sig_x1))

    return mu_x0, sig_x0, mu_x1, sig_x1


def contour_transform(x0, x1, joint_pdf):
    """
    Convert a 2D likelihood contour from modified to original space
    along with marginalized statistics
    """
    mu_x0, sig_x0, mu_x1, sig_x1 = marg_estimates(x0, x1, joint_pdf)

    # Convert the convert to original space
    corners = np.array([[mu_x0 - 5 * sig_x0, mu_x1 - 5 * sig_x1],
               [mu_x0 - 5 * sig_x0, mu_x1 + 5 * sig_x1],
               [mu_x0 + 5 * sig_x0, mu_x1 - 5 * sig_x1],
               [mu_x0 + 5 * sig_x0, mu_x1 + 5 * sig_x1]
                ])

    extents = get_transform(corners, shift, tilt, dir='down')

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
    _grid_in_mod = get_transform(_point_orig, shift, tilt, dir='up')

    values_orig = _b.ev(_grid_in_mod[:, 0], _grid_in_mod[:, 1])
    values_orig = values_orig.reshape(tau0.shape)

    return tau0, gamma, values_orig



def plot_ellipse(pos, cov, nsig=[1], ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
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
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

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


# Plotting composites
def plotcomp(myfile, suffix='temp', nskip=1, save=False, conf_int=False,
             ratio_range=[1600, 1800], alpha_fit=None):
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

        #ax1.set_color_cycle(sns.hls_palette(8, l=.3, s=.8))
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
            #print('%.3f' % ((ratio - 1) * (1700 / 1450.)))
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
        if save:
            plt.savefig(suffix+'.pdf', rasterized=True)
        plt.show()
        return ax1, ax2, ax3
    except Exception as e:
        raise

def wrapnorm(z_norm):
    def refmodel(x, a,b):
        return a*((1+x)**b - (1 + z_norm)**b)
    return refmodel

def wrapchi(z,tau,cov, z_norm): 
    def chisq_cor(tau0, gamma):
        model = tau0*((1+z)**gamma - (1 + z_norm)**gamma)
        value = np.dot((model - tau), np.dot(inv(cov), (model - tau).T))
        chi_abs = ((0.5549 - np.exp(-tau0*(1+3.4613)**gamma))/0.0216)**2 + ((0.5440 - np.exp(-tau0*(1+3.5601)**gamma))/0.0229)**2  # STOLEN FROM NAO'S PAPER
        return value + chi_abs
    return chisq_cor

# PRINTS THE ERRORBAR PLOT FOR A GIVEN INPUT DATA FILE
def graphanize(infile, z_norm, label='test', approxfit=True, truefit=False, linecolor='gray', markercolor='k'):
    f = np.loadtxt(infile).T

    # Read in the redshift, delta tau values for the masterrun and all the bootrun
    z, m_Tau, C = f[:,0] , f[:,1], f[:,2:]

    # Refine this later
    clipInd = np.where((np.isfinite(m_Tau)))[0]
    z, m_Tau, C = z[clipInd], m_Tau[clipInd], C[clipInd]

    # Calulate the covariance matrix and the correlation matrix
    Cov, Corr = np.cov(C),  np.corrcoef(C)

    # bootstrap bias correction
    #m_Tau = 2 * m_Tau - np.average(C, axis=1)

    # plt.figure()
    # plt.imshow(Corr, interpolation='None')

    print('Number of data-points are %d' %len(z))
    # Plot the graph with errorbars
    # plt.figure()
    # plt.errorbar(z, m_Tau , np.sqrt(np.diag(Cov)), fmt='o', color=markercolor, label=label)
    # plt.xlabel(r'$z$', fontsize=25)
    # plt.ylabel(r'$\Delta \tau_{eff}$', fontsize=25)

    # Plot the approximate(wrong) solution
    if approxfit==True:
        popt, pcov = curve_fit(wrapnorm(z_norm), z, m_Tau, sigma=np.sqrt(np.diag(Cov)))
        # plt.plot(z, wrapnorm(z_norm)(z, popt[0], popt[1]), color=linecolor, linewidth=0.6)

    # Plot the correct solution contour
    if truefit==True:
        #ex1, ex2 = np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1])

        #X = np.linspace(popt[0] - 2*ex1, popt[0] + 10*ex1, 300)
        #Y = np.linspace(popt[1] - 5*ex2, popt[1] + 2*ex2, 300)

        X = np.linspace(0, 0.005, 300)
        Y = np.linspace(2.5, 5.5, 300)
        
        CHI_COR = np.zeros((len(X), len(Y)))
        myfunc = wrapchi(z, m_Tau, Cov, z_norm)

        for i in range(len(X)):
            for j in range(len(Y)):
                CHI_COR[i,j] = myfunc(X[i], Y[j])

        l0 = min(np.ravel(CHI_COR)) # +2.30, +6.18 
        min_chi = l0/(len(z) -2)
        print('chi-squared per dof = %.2f' %min_chi)
    

        #np.savetxt('chisq.dat', CHI_COR.T)

        # FINAL CHI_SURFACE PLOT
        # plt.figure()
        #plt.contourf(X, Y, CHI_COR.T, [l0, l0 + 2.30], colors=linecolor)
        plt.contour(X, Y, CHI_COR.T, [l0, l0 + 2.30, l0 + 6.18], colors=linecolor)

        plt.xlabel(r'$\tau_0$', fontsize=25)
        plt.ylabel(r'$\gamma$', fontsize=25)
    plt.show()
    

def plotchi(xbin, ybin, colors, label):
    mydir = os.getcwd()
    
    alpha = np.array([-2.83, -2.12, -1.41, -0.7 ])
    CIV = np.array([20,40,60,80 ])

    X = np.linspace(0, 0.005, 300)
    Y = np.linspace(2.5, 5.5, 300)

    # colors = np.array([['r', 'g', 'b'], ['indigo', 'purple', 'brown'], ['gray', 'maroon', 'k']])

    #plt.figure(figsize=(10,10))
    finalchi = np.zeros((len(X), len(Y)))

    for i in xbin:
        for j in ybin:
            f = os.environ['OPT_COMPS'] + 'comp_V2_44_%d%d_equalz_tavg' %(i,j) 

            print('Going to directory %s' %f)

            try:
                os.chdir(f)
                data = np.loadtxt('chisq.dat')
                
                # 68 % and 95 % CI
                l0 = min(np.ravel(data))

                plt.contour(X, Y, data, [l0, l0+2.30], colors=colors[i-1,j-1])
                #CS = plt.contour(X, Y, data, [l0, l0+2.30, l0+6.18], colors=colors[i-1,j-1], label=r'$%.2f < \alpha < %.2f, %.2f < CIV < %.2f$' %(alpha[i-1], alpha[i], CIV[j-1], CIV[j]))

                #plt.clabel(CS, inline=1, fontsize=10)
                #CS.collections[0].set_label(r'$%.2f < \alpha < %.2f, %.2f < CIV < %.2f$' %(alpha[i-1], alpha[i], CIV[j-1], CIV[j]))

                # Do the contour plotting
                finalchi += data
            except Exception as e:
                print(e.__doc__)
                print(e.message)
                os.chdir(mydir)

    #plt.figure()
    l0 = min(np.ravel(finalchi))

    #plt.contour(X, Y, finalchi, [l0, l0+2.30], colors=('red'))
    #plt.contour(X, Y, finalchi, [l0,  l0+6.18], colors=('r'))

    os.chdir(mydir)

    plt.xlabel(r'$\tau_0$', fontsize=25)
    plt.ylabel(r'$\gamma$', fontsize=25)
    plt.legend()
    plt.show()

    #plt.savefig(label+'.pdf')

def chisq_paris():
    x = np.linspace(0.001, 0.0056, 1000)
    y = np.linspace(2.8, 4.3, 1000)

    X, Y = np.meshgrid(x, y)

    # STOLEN FROM NAO'S PAPER
    Z = ((0.5549 - np.exp(-X*(1+3.4613)**Y))/0.0216)**2 + ((0.5440 - np.exp(-X*(1+3.5601)**Y))/0.0229)**2  

    l0 = min(np.ravel(Z))

    #plt.figure()
    plt.contourf(X, Y, Z.T, [l0, l0 + 2.30], colors='r')
    plt.contour(X, Y, Z.T, [l0, l0 + 6.18], colors='r')
    plt.show()


if __name__=="__main__":
    np.random.seed()
    mu_true = np.array([2., 4.])

    N = 20

    sig_x = 3 * np.random.uniform(size=N)
    sig_y = 2 * np.random.uniform(size=N)
    rho_xy = np.random.uniform(-1, 1, size=N)
    sig_xy = rho_xy * sig_x * sig_y

    X = np.zeros((N, 2))
    covMat = np.zeros((N, 2, 2))

    covSys = np.array([[2, 1], [1, 2]])

    for j in range(N):
        cov_intrinsic = np.array([[sig_x[j]**2, sig_xy[j]], [sig_xy[j], sig_y[j]**2]])
        covMat[j] = cov_intrinsic
        X[j] = np.random.multivariate_normal(mu_true, cov_intrinsic + covSys)


    # fig, ax = plt.subplots(1)
    # [plot_cov_ellipse(X[i], covMat[i], ax=ax) for i in range(N)]
    # ax.set_xlim(mu_true[0] - 3 * sig_x.max(), mu_true[0] + 3 * sig_x.max())
    # ax.set_ylim(mu_true[0] - 3 * sig_y.max(), mu_true[0] + 3 * sig_y.max())
    # plt.show()

    gaussfit_2d(X, covMat)


def other_plot():
    def abs_tau(x, tau0, gamma):
        return tau0 * (1 + x) ** gamma

    plt.figure()

    x = np.linspace(2, 5, 100)

    def log_trans(x, a, b):
        tau = np.exp(a) * (1 + x) ** b
        return tau

    def trans(x, a, b):
        tau = a * (1 + x) ** b
        return tau
    
    # This work
    tau_best = [-5.261, 3.205]
    cov_orig =  [[ 0.03976282, -0.02380281],
                 [-0.02380281,  0.0146614 ]]

    pts = np.random.multivariate_normal(tau_best, cov_orig, size=500)
    d_pts = np.array([log_trans(x, *pt) for pt in pts])
    my_avg, my_err = d_pts.mean(0), d_pts.std(0)

    plt.plot(x, my_avg, c='k', label='This Work')
    plt.fill_between(x, my_avg + my_err, my_avg - my_err, color='gray', alpha=0.5)


    # For Kim
    gamma_kim, tau_0_kim = 3.65, 0.0023
    plt.plot(x, trans(x, tau_0_kim, gamma_kim), linestyle='dashed', c='g', label='Kim et. al. (2007)')

    # For Kirkmann
    gamma_s, tau_0_s = 4.164, 10**(-2.910)
    plt.plot(x, trans(x, tau_0_s, gamma_s), linestyle='dotted', label='Kirkman et. al. (2005)')

    # For Scheye
    gamma_s, tau_0_s = 4.057, 10**(-2.853)
    plt.plot(x, trans(x, tau_0_s, gamma_s), linestyle='dashdot', color='black', label='Scheye et. al. (2003)')

    # For Faucher
    z_data = np.arange(2., 4.3, 0.2)
    # f_data = [0.127, 0.164, 0.203, 0.251, 0.325, 0.386, 0.415, 0.570, 0.716, 0.832, 0.934, 1.061]
    f_data = [0.146, 0.187, 0.229, 0.280, 0.360, 0.423, 0.452, 0.615, 0.766, 0.885, 0.986, 1.113]
    err_data = [0.023, 0.02, 0.019, 0.022, 0.026, 0.029, 0.031, 0.039, 0.051, 0.051, 0.064, 0.0121]
    gamma_f, tau_0_f = 3.92, 0.0018
    plt.errorbar(z_data, f_data, err_data, fmt='.', color='r', capsize=2)

    plt.plot(x, trans(x, tau_0_f, gamma_f), linestyle='dashdot', color='r', label='Faucher-Giguere et. al. (2008c)')

    # For Becker 2012
    z_data = np.arange(2.15, 4.95, 0.1)
    f_data = [0.8806, 0.8590, 0.8304, 0.7968, 0.7810, 0.7545, 0.7371, 0.7167, 0.6966, 0.6670, 0.6385, 0.6031, 0.5762, 0.5548, 0.5325, 0.4992, 0.4723, 0.4470, 0.4255, 0.4030, 0.3744, 0.3593, 0.3441, 0.3216, 0.3009, 0.2881, 0.2419, 0.2225]
    err_data = [0.0103, 0.0098, 0.0093, 0.0089, 0.0090, 0.0088, 0.0088, 0.0086, 0.0084, 0.0082, 0.0080, 0.0079, 0.0074, 0.0071, 0.0071, 0.0069, 0.0068, 0.0072, 0.0071, 0.0071, 0.0074, 0.0075, 0.0102, 0.0094, 0.0104, 0.0117, 0.0201, 0.0151] 
    z0, beta, C, tau0 = 3.5, 2.90, -0.132, 0.751
    # plt.errorbar(z_data, f_data, err_data, fmt='.', color='orange', capsize=2)
    plt.plot(x, (tau0*(((1+x)/(1+z0))**beta) + C), linestyle='dashed', color='orange', label='Becker et. al. (2012)')

    plt.ylabel(r'$\tau_{\mathrm{eff}}(z)$')
    plt.xlabel(r'$z$')
    plt.legend(frameon=False)

    # plt.xlim(1.5, 4.2)
    # plt.ylim(0.0, 1.1)
    plt.show()


num_bins = 7
bin_colors = sns.hls_palette(7, l=.3, s=.8)


import mcmc_skewer
import importlib

importlib.reload(mcmc_skewer)


def bin_plot(ax1, ax2):
    """ Plots optical depth measurements for all the bins
    """

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_1_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[0], ls='dashed', mylabel='Bin 1')

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_2_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[1], ls='dashed', mylabel='Bin 2')

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_3_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[2], ls='dashed', mylabel='Bin 3')

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_4_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[3], ls='dashed', mylabel='Bin 4')

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_5_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[4], ls='dashed', mylabel='Bin 5')

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_6_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[5], ls='dashed', mylabel='Bin 6')

    # mcmc_skewer.addLogs('../LogLikes/Bin_skewer_final_7_rpix_distort1070_1160/',
    #                     individual=False, orig_ax=ax2, mod_ax=ax1,
    #                     mycolor=bin_colors[6], ls='dashed', mylabel='Bin 7')

    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))
    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin1_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[0], ls='dashed', mylabel='Bin 1')

    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin2_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[1], ls='dashed', mylabel='Bin 2')

    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin3_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[2], ls='dashed', mylabel='Bin 3')

    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin4_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[3], ls='dashed', mylabel='Bin 4')

    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin5_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[4], ls='dashed', mylabel='Bin 5')

    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin6_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[5], ls='dashed', mylabel='Bin 6')

    mcmc_skewer.addLogs('../LogLikes/Bin_skewer_bin7_highsn_distort_distort1070_1160/',
                        individual=False, orig_ax=ax2, mod_ax=ax1,
                        mycolor=bin_colors[6], ls='dashed', mylabel='Bin 7')

    return ax1, ax2


def fancy_scatter(param1, param2, values=None, bins=60, names=['x', 'y'], marginalized=False):
    """ Scatter plot of paramters with number desnity contours
    overlaid. Marginalized 1D distributions also plotted.

    Make sure that the data is appropriately cleaned before using
    this routine
    """
    # Simple cuts to remove bad-points
    import warnings
    warnings.filterwarnings('ignore')

    from scipy.stats import binned_statistic_2d as bs2d

    ixs = np.isfinite(param1) & np.isfinite(param2)

    range1 = np.percentile(param1[ixs], [5, 95])
    range2 = np.percentile(param2[ixs], [5, 95])

    # Use interquartile ranges to suitably set the ranges for the bins
    width1, width2 = np.diff(range1), np.diff(range2)

    if values is None:
        N, xedges, yedges, __ = bs2d(param1[ixs], param2[ixs], None,
                                 'count', bins, range=[[range1[0] - width1, range1[1] + width1], [range2[0] - width2, range2[1] + width2]])

    else:
        N, xedges, yedges, __ = bs2d(param1[ixs], param2[ixs], values[ixs], 'mean',
                                 bins, range=[[range1[0] - width1, range1[1] + width1], [range2[0] - width2, range2[1] + width2]])

    if not marginalized:
        fig, ax2d = plt.subplots(1, figsize=(6, 5))
    else:
        fig = plt.figure(figsize=(6, 6))
        ax2d = fig.add_subplot(223)
        ax1 = fig.add_subplot(221, sharex=ax2d)
        ax3 = fig.add_subplot(224, sharey=ax2d)

    if values is None:
        ax2d.imshow(np.log10(N.T), origin='lower', extent=[xedges[0],
                    xedges[-1], yedges[0], yedges[-1]], aspect='auto',
                    cmap=plt.cm.binary, interpolation='nearest')
    else:
        cmap_multicolor = plt.cm.jet
        cmap_multicolor.set_bad('w', 1.)

        cs = ax2d.imshow(N.T, origin='lower', extent=[xedges[0],
                         xedges[-1], yedges[0], yedges[-1]], aspect='auto',
                         cmap=cmap_multicolor, interpolation='nearest')
        cb = plt.colorbar(cs)
        cb.set_clim(np.nanpercentile(N, 50), np.nanpercentile(N, 60))
    # Overlay density contours in log space to normalize the contrast

    if values is None:
        levels = np.linspace(0, np.log(N.max()), 10)[2:]
        ax2d.contour(np.log(N.T), levels, colors='k',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    ax2d.set_xlim(xedges[0], xedges[-1])
    ax2d.set_ylim(yedges[0], yedges[-1])
    ax2d.set_xlabel(r'{}'.format(names[0]), fontsize=16)
    ax2d.set_ylabel(r'{}'.format(names[1]), fontsize=16)

    if marginalized:
        ax1.hist(param1, bins=xedges, histtype='step', range=range1, color='k')
        ax3.hist(param2, bins=yedges, histtype='step', range=range2, color='k',
                 orientation='horizontal')

        ax1.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)

        ax1.set_ylabel(r'$\mathrm{N}$', fontsize=16)
        ax3.set_xlabel(r'$\mathrm{N}$', fontsize=16)

        # Put percentile cuts on the marginalized plots
        p1_cuts = np.percentile(param1[ixs], [10, 90])
        p2_cuts = np.percentile(param2[ixs], [10, 90])

        for ele in p1_cuts:
            ax1.axvline(ele)
        for ele in p2_cuts:
            ax3.axhline(ele)

        for ele in np.linspace(*p1_cuts, 4):
            ax2d.axvline(ele, color='r', linewidth=0.6)
        for ele in np.linspace(*p2_cuts, 4):
            ax2d.axhline(ele, color='r', linewidth=0.6)

        print(names[0], np.linspace(*p1_cuts, 4))
        print(names[1], np.linspace(*p2_cuts, 4))

        bin_counts = bs2d(param1[ixs], param2[ixs], None, 'count',
                          bins=[np.linspace(*p1_cuts, 4), np.linspace(*p2_cuts, 4)]).statistic
        print(bin_counts)


    plt.show()
