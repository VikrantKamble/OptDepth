import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


# figure 1 - demo quasar
def demo_plot(qso):
    ind = np.where(qso.sn > 10)[0]
    fig = plt.figure(figsize=(6.2, 5))
    plt.plot(qso.wl, qso.flux[ind[2]], c='k', alpha=0.6, lw=0.5)
    plt.xlim(900, 2900)
    plt.minorticks_on()
    plt.xlabel(r'$\lambda_\mathrm{rf}\ [\mathrm{\AA}]$', fontsize=16)
    plt.ylabel(r'$f_\lambda\ [10^{-17} \mathrm{ergs}\ \mathrm{cm}^{-2}\ \mathrm{s}^{-1}\ \mathrm{\AA}^{-1}]$', fontsize=16)
    plt.plot(qso.wl,  (qso.wl/1450.) ** qso.tb['ALPHA_V0'][ind[2]], c='r', lw=0.7)

    # param_fit.fit_alpha(qso.flux[ind[2]], qso.ivar[ind[2]], qso.zq[ind[2]], plotit=True)
    ax = fig.add_axes([0.45, 0.45, 0.35, 0.4])

    param_fit.fit_EW(qso.flux[ind[2]], qso.ivar[ind[2]], qso.zq[ind[2]], plotit=True, ax=ax)

    ax.set_xlim(1420, 1680)
    ax.minorticks_on()
    # ax.set_ylim(6.5, 15)
    fig.text(0.15, 0.8, r'$z_q = 2.83$', fontsize=12)
    fig.text(0.15, 0.75, r'$\alpha=-1.56$', fontsize=12)
    fig.text(0.15, 0.7, r'$C IV EW=35.8$', fontsize=12)

# figure 2 - calibrations
# Done in their respective files

# figure 3 - parameter distribution
def param_dist(qso):

    ind = qso.sn > 5
    alphas = qso.tb['ALPHA_V0'][ind]
    ews = qso.tb['EW_V0'][ind]

    cleaned = np.isfinite(alphas) & np.isfinite(ews) &\
        (alphas > -3.4) & (alphas < -0.2) & (ews > 0) & (ews < 80)
    alphas, ews = alphas[cleaned], ews[cleaned]

    N, xedges, yedges = np.histogram2d(alphas, ews, bins=60)

    fig = plt.figure(figsize=(4, 4))
    ax2 = fig.add_subplot(223)
    ax1 = fig.add_subplot(221, sharex=ax2)
    ax3 = fig.add_subplot(224, sharey=ax2)

    ax2.imshow(np.log10(N.T), origin='lower', extent=[xedges[0],
               xedges[-1], yedges[0], yedges[-1]], aspect='auto',
               cmap=plt.cm.binary, interpolation='nearest')

    levels = np.linspace(0, np.log(N.max()), 7)[2:]
    ax2.contour(np.log(N.T), levels, colors='k',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], zorder=1)
    ax1.hist(alphas, bins=30, histtype='step', color='k', linewidth=0.6)
    ax3.hist(ews, bins=30, orientation='horizontal', histtype='step', color='k', linewidth=0.6)
    ax1.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax2.set_xlim(xedges[0], xedges[-1])
    ax2.set_ylim(yedges[0], yedges[-1])
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\mathrm{C\ IV\ EW}$')
    ax1.set_ylabel(r'$\#$')
    ax3.set_xlabel(r'$\#$')

    # create rectangles
    deltax, deltay = 0.66, 13.33
    p1 = Rectangle((-2.8, 20), deltax, 20, fill=False, edgecolor='r', linewidth=0.6)
    p2 = Rectangle((-2.8, 40), deltax, 20, fill=False, edgecolor='r', linewidth=0.6)
    p3 = Rectangle((-2.13, 20), deltax, deltay, fill=False, edgecolor='r', linewidth=0.6)
    p4 = Rectangle((-2.13, 33.33), deltax, deltay, fill=False, edgecolor='r', linewidth=0.6)
    p5 = Rectangle((-2.13, 46.66), deltax, deltay, fill=False, edgecolor='r', linewidth=0.6)
    p6 = Rectangle((-1.46, 20), deltax, 20, fill=False, edgecolor='r', linewidth=0.6)
    p7 = Rectangle((-1.46, 40), deltax, 20, fill=False, edgecolor='r', linewidth=0.6)
    p_list = [p1, p2, p3, p4, p5, p6, p7]
    p = PatchCollection(p_list, match_original=True)
    ax2.add_collection(p)

    ax1.axvline(-2.8, color='r', linewidth=0.6)
    ax1.axvline(-0.8, color='r', linewidth=0.6)
    ax3.axhline(20, color='r', linewidth=0.6)
    ax3.axhline(60, color='r', linewidth=0.6)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    plt.savefig('param_dist.pdf')


# figure 3
# histogram of redshifts for each bin
def hist_dist():
    fig, ax = plt.subplots(1, figsize=(6.2, 5))
    mybins = np.linspace(1.6, 4, 40)
    plt.hist(bin1._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 1$');
    plt.hist(bin2._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 2$');
    plt.hist(bin3._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 3$');
    plt.hist(bin4._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 4$');
    plt.hist(bin5._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 5$');
    plt.hist(bin6._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 6$');
    plt.hist(bin7._zq, histtype='step', bins=mybins, label=r'$\mathrm{Bin}\ 7$');
    plt.xlabel(r'$z_q$')
    plt.ylabel(r'$\#$')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('redshift_dist.pdf')
