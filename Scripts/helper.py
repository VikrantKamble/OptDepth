"""
  Store all the helper functions in this file
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from astroML.plotting.mcmc import convert_to_stdev as cts


def xfm(pos, shift, tilt, dir='down'):
    """
    Perform conversion from one system to another

    dir : direction to do the transform
        up : orig to mod
        down(default) : mod to orig
    """
    if np.array(pos).ndim == 1:
        pos = np.array(pos)[None, :]
    if dir == 'up':
        return np.dot((pos - shift), tilt.T)
    elif dir == 'down':
        return np.dot(np.linalg.inv(tilt), pos.T).T + shift


def marg_estimates(x0_line, x1_line, joint_pdf, ax=None, plot_marg=True,
                   labels=None):
    """
    Marginalized statistics that follows from a jont likelihood.
    Simple mean and standard deviation estimates.

    Parameters:
        x0 : vector in x-direction of the grid
        x1 : vector in y-direction of the grid
        joint_pdf : posterior log probability on the 2D grid

    Returns:
        [loc_x0, sig_x0, loc_x1, sig_x1]
    """
    x0_pdf = np.sum(np.exp(joint_pdf), axis=1)
    x0_pdf /= x0_pdf.sum() * (x0_line[1] - x0_line[0])
    x1_pdf = np.sum(np.exp(joint_pdf), axis=0)
    x0_pdf /= x0_pdf.sum() * (x0_line[1] - x0_line[0])

    mu_x0 = (x0_line * x0_pdf).sum() / x0_pdf.sum()
    mu_x1 = (x1_line * x1_pdf).sum() / x1_pdf.sum()
    sig_x0 = np.sqrt((x0_line ** 2 * x0_pdf).sum() / x0_pdf.sum() - mu_x0 ** 2)
    sig_x1 = np.sqrt((x1_line ** 2 * x1_pdf).sum() / x1_pdf.sum() - mu_x1 ** 2)

    print("param1 = %.4f pm %.4f" % (mu_x0, sig_x0))
    print("param2 = %.4f pm %.4f\n" % (mu_x1, sig_x1))

    if labels is None:
        labels = ["p0", "p1"]

    if ax is None:
        fig, ax = plt.subplots(1)
    ax.contour(x0_line, x1_line,  cts(joint_pdf.T), colors=('k',), levels=[0.668, 0.955])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim(mu_x0 - 4 * sig_x0, mu_x0 + 4 * sig_x0)
    ax.set_ylim(mu_x1 - 4 * sig_x1, mu_x1 + 4 * sig_x1)

    if plot_marg:
        xx_extent = 8 * sig_x0
        yy_extent = 8 * sig_x1

        pdf_xx_ext = x0_pdf.max() - x0_pdf.min()
        pdf_yy_ext = x1_pdf.max() - x1_pdf.min()

        ax.plot(x0_line, 0.2 * (x0_pdf - x0_pdf.min()) * yy_extent / pdf_xx_ext
                + ax.get_ylim()[0])
        ax.axvline(mu_x0 - sig_x0)
        ax.axvline(mu_x0 + sig_x0)
        ax.plot(0.2 * (x1_pdf - x1_pdf.min()) * xx_extent / pdf_yy_ext +
                ax.get_xlim()[0], x1_line)
        ax.axhline(mu_x1 - sig_x1)
        ax.axhline(mu_x1 + sig_x1)

        plt.title(r"$%s = %.3f \pm %.3f, %s = %.3f \pm %.3f$" % (labels[0], mu_x0, sig_x0,
                                                                 labels[1], mu_x1, sig_x1))
    plt.tight_layout()
    plt.show()

    return mu_x0, sig_x0, mu_x1, sig_x1


def get_corrfunc(x, x_err=None, n_frac=2, viz=True,
                 model=False, est=False):
    """
    Auto correlation of a signal and mean estimation

    Parameters
        x : samples of the first variable
        x_err : error vector for the first variable
        n_frac : number of pixels over which to estiamte correlation wrt
                 the size of the samples
        viz : plot the correlation function
        model : model the correlation function
        est : Get estimates on the best-fit values using the covariance
              matrices estmated

    Returns:
        loc, sig : the mean and uncertainty on the location parameter
    """
    npp = len(x)
    coef = [np.corrcoef(x[:npp - j], x[j:])[0, 1] for j in
            range(npp // n_frac)]

    if viz:
        fig, ax = plt.subplots(1)
        ax.plot(coef, '-k')

    if model:
        def model_func(x, cl):
            # A simple exponential model
            return np.exp(- x / cl)

        popt, __ = curve_fit(model_func, np.arange(npp // n_frac)[:20], coef[:20])

        if viz:
            x_plot = np.linspace(0, 50, 50)
            ax.plot(x_plot, model_func(x_plot, *popt), '-r')

            ax.text(10, 0.8, "$r_{x_0}=%.1f$" % popt[0])
            ax.set_xlim(0, 50)

        if est:
            if x_err is None:
                raise TypeError("Requires errorbars on the samples")

            # Obtain band-diagonal correlation matrix
            from scipy.linalg import toeplitz
            Xi = toeplitz(model_func(np.arange(npp), *popt))

            # Covariance matrix from errorbars and correlation matrix
            # C = D * Xi * D.T
            cov = np.diag(x_err).dot(Xi.dot(np.diag(x_err)))

            # Minimization using iminuit
            from iminuit import Minuit

            ico = np.linalg.inv(cov)

            def chi2(mu):
                dyi = x - mu
                return dyi.T.dot(ico.dot(dyi))

            mm = Minuit(chi2,
                        mu=x.mean(), error_mu=x.std(), fix_mu=False,
                        errordef=1., print_level=-1)

            mm.migrad()

            loc, sig = mm.values["mu"], mm.errors["mu"]

            if viz:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.plot(x)
                ax.fill_between(np.arange(npp), loc + sig,
                                loc - sig, color='r', alpha=0.2)

            return loc, sig
