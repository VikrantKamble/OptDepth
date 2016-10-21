import matplotlib.pyplot as plt
import os
import fitsio
import config_read as cfg
import seaborn.apionly as sns


from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=25) 
rc('ytick', labelsize=25)

# Plotting composites 
def plotcomp(infile):
	temp = os.environ['OPT_COMPS'] + infile
	prev_dir = os.getcwd()

	try:
		os.chdir(temp)
		myfile = temp  + '/' + infile + '.fits'

		f = fitsio.FITS(myfile)
		comp, ivar, z = f[0].read(), f[1].read(), f[2]['REDSHIFT'][:]

		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111)

		ax.set_rasterization_zorder(1)
		ax.set_color_cycle(sns.hls_palette(8,l=.3,s=.8))

		for count, ele in enumerate(comp):
			ax.plot(cfg.wl, ele, linewidth=0.3, label=r'$z = %.2f$'%z[count], zorder=0)

		ax.set_xlabel(r'$\lambda_{rf}$', fontsize=25)
		ax.set_ylabel(r'$F_\lambda$ (Arbitrary Units)',fontsize=25)
		ax.legend()
		ax.set_xlim(1020, 2000)
		ax.set_ylim(0,4)

		plt.tight_layout()
		plt.legend(prop={'size':20})
		plt.show()
		
		fig.savefig(infile+'.eps', rasterized=True)

	except Exception as e:
		print e.__doc__
		print e.message
		os.chdir(prev_dir)

	os.chdir(prev_dir)


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

	print 'Number of data-points are %d' %len(z)
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
		print 'chi-squared per dof = %.2f' %min_chi
	

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

			print 'Going to directory %s' %f

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
				print e.__doc__
				print e.message
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