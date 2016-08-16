import matplotlib.pyplot as plt
import seaborn.apionly as sns

#MASTER FILE THAT CONTAINS PLOT PRODCEDURES FOR MOST OF THE VISUAL INSPECTION REQUIRED


from matplotlib import rc
rc('text', usetex=True)
#rc('xtick', labelsize=20) 
#rc('ytick', labelsize=20)

# Plotting composites 
def plotcomp(infile):
	f = fitsio.FITS(infile)
	comp, ivar, z = f[0].read(), f[1].read(), f[2]['REDSHIFT'][:]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle(sns.hls_palette(8,l=.3,s=.8))

	for count, ele in enumerate(comp):
		ax.plot(wl, ele, linewidth=0.6, label=r'$z = %.2f$'%z[count])

	ax.set_xlabel(r'$\lambda_{rf}$', fontsize=25)
	ax.set_ylabel(r'$F_\lambda$',fontsize=25)
	ax.legend()

	plt.show()



# import matplotlib.gridspec as gridspec


# fig = plt.figure(221, figsize=(6.3, 6.3))


# axScatter = fig.add_subplot(223)
# axHistX = fig.add_subplot(221)
# axHistY = fig.add_subplot(224)
# axHistX.xaxis.set_major_formatter(nullfmt)
# axHistY.yaxis.set_major_formatter(nullfmt)

# axScatter.scatter(myalpha, myew, marker='+', color = 'gray', edgecolor='none', s=5, alpha=1)
# axScatter.set_xlim(-3., -0.5)
# axScatter.set_ylim(10., 120)

# axHistX.hist(myalpha, 20, histtype='step', color='gray')
# axHistY.hist(myew, 20, histtype='step', color='gray', orientation='horizontal')
# xbins = np.linspace(-2.83, -0.70,4)
# ybins = np.linspace(24.56, 108.34, 4)

# for i in ybins:
#     axScatter.axhline(i, color='g')
#     axHistY.axhline(i, color='g')

# for i in xbins:	
#     axScatter.axvline(i, color='g')
#     axHistX.axvline(i, color='g')

# axScatter.set_xlabel(r'$\alpha_{\lambda}$', fontsize=22)
# axScatter.set_ylabel(r'$CIV_{EW}$', fontsize=22)

# st = fig.suptitle("Parameters bins for composite creation", fontsize="x-large")

# for i in range(0,len(comp),4):
#     ax1.plot(wl, comp[i], linewidth=0.4, label=r'$z = %.2f$' %z[i])

# xlim(1030, 1190)


# gs = gridspec.GridSpec(2,2)
# ax1 = plt.subplot(gs[0,0])
# ax2 = plt.subplot(gs[0,1])

# ax3 = plt.subplot(gs[1,0])
# ax4 = plt.subplot(gs[1,1])
# temp = ax2.imshow(Corr, interpolation='None');
# colorbar(temp, ax=ax2)
# ax1.errorbar(z, m_Tau, np.sqrt(np.diag(Cov)), fmt='o', color='k')

# ax3.contour(X*10**3,Y,CHI_COR.T,[l1,l2],linewidths=2, cmap='copper')

# #ax4.plot(z, absmodel(z, popt[0], popt[1]), color='g', linewidth=0.6, label='Fit without Cov')
# ax4.errorbar(z, m_Tau+0.6682287, np.sqrt(np.diag(Cov)), fmt='o', color='k', markersize=2.0, label='Data: This work')

# ax4.plot(z, absmodel(z, 0.00322548,3.54606), color='r', linewidth=0.6, label='Fit: This work')
# ax1.set_xlabel(r'$z$', fontsize=20)

# ax1.set_ylabel(r'$\Delta \tau$', fontsize=20)
# ax3.set_xlabel(r'$\tau_0 \times 10^{-3}$', fontsize=20)
# ax3.set_ylabel(r'$\gamma$', fontsize=20)

# ax2.set_title('Correlation amongst points from bootstrapping')
# ax1.set_title(r'Relative $\tau$ measurement from composites')

# ax3.set_title(r'Chi-surface with 1 and 2-$\sigma$ CI')
# ax4.set_title(r'Absolute $\tau$ measurement')
# ax3.set_xlabel(r'$\tau_0 \times 10^{-3}$', fontsize=20)
# suptitle(r'Analysis for bin$:-2.12<  \alpha_{\lambda} < -1.41,\quad 24.56< CIV_{EW} < 52.48$', fontsize=15)

# ax4.set_ylabel(r'$\tau_{eff}$', fontsize=20)
# ax4.set_xlabel(r'$z$', fontsize=20)


# ax4.errorbar(beck_z, -np.log(beck_f), beck_sig/beck_f, markersize=4, fmt='s', label='Data: Becker et al.(2013)')
# ax4.plot(beck_z, becker(beck_z, 0.751, 2.90, -0.132), color='g', linewidth=0.6, label='Fit: Becker et al.(2013)')
# ax4.legend(loc='upper left')
# ax4.set_yscale('log')

