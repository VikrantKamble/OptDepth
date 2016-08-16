from __future__ import division
from numpy.linalg import inv
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os

X = ConfigMap('CalcTau')

ly_line = float(X['line'])
ly_range = json.loads(X['ly_range'])

z_norm = float(X['z_norm'])
zdiv = int(X['zdiv'])

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
def graphanize(infile, approxfit=True, truefit=False, linecolor='red', markercolor='k'):
	f = np.loadtxt(infile).T

	# Read in the redshift, delta tau values for the masterrun and all the bootrun
	z, m_Tau, C = f[:,0] , f[:,1], f[:,2:]

	# Refine this later
	clipInd = np.where((isfinite(m_Tau)) & (z > 2.1))[0]
	z, m_Tau, C = z[clipInd], m_Tau[clipInd], C[clipInd]

	# Calulate the covariance matrix and the correlation matrix
	Cov, Corr = np.cov(C),  np.corrcoef(C)

	# Plot the graph with errorbars
	plt.figure()
	plt.errorbar(z, m_Tau , np.sqrt(np.diag(Cov)), fmt='o', color=markercolor)

	# Plot the approximate(wrong) solution
	if approxfit==True:
		popt, pcov = curve_fit(wrapnorm(z_norm), z, m_Tau, sigma=np.sqrt(np.diag(Cov)))
		plt.plot(z, wrapnorm(z_norm)(z, popt[0], popt[1]), color=linecolor, linewidth=0.6)

	# Plot the correct solution contour
	if truefit==True:
		ex1, ex2 = np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1])

		X = np.linspace(popt[0] - 1*ex1, popt[0] + 10*ex1, 300)
		Y = np.linspace(popt[1] - 10*ex2, popt[1] + 1*ex2, 300)

		CHI_COR = np.zeros((len(X), len(Y)))
		myfunc = wrapchi(z, m_Tau, Cov, z_norm)

		for i in range(len(X)):
			for j in range(len(Y)):
				CHI_COR[i,j] = myfunc(X[i], Y[j])

		l0 = min(np.ravel(CHI_COR)) # +2.30, +6.18 
	
		# FINAL CHI_SURFACE PLOT
		figure()
		plt.contourf(X, Y, CHI_COR.T, [l0, l0 + 2.30, l0 + 6.18])
		plt.show()


# RUNS CALC_TAU FOR ALL THE FILES IN THE INPUT FOLDER
def runall(fname, plotit=False, savefile='data.dat', trim_obs_lam = 0):
	prev_dir = os.getcwd()
	try:
		os.chdir(fname)
		mastername = fname + '.fits'
		process = subprocess.Popen('ls *boot* > temp', shell=True, stdout=subprocess.PIPE)
		
		M_z, M_tau, M_zbins = calc_tau(mastername, ly_range, ly_line, z_norm, False, zdiv, trim_obs_lam = trim_obs_lam)

		DataArr = [] 
		DataArr.append(M_z)
		DataArr.append(M_tau)

		# temp is the file containing boot names
		with open('temp') as f:
			for line in f:
				foo = calc_tau(line,ly_range, ly_line, z_norm, True, M_zbins, trim_obs_lam = trim_obs_lam)[1]
				DataArr.append(foo)

		np.savetxt(savefile, np.array(DataArr))
	except Exception as e:
		print e.__doc__
		print e.message
		os.chdir(prev_dir)

		if plotit==True:graphanize('data.dat')
	os.chdir(prev_dir)
