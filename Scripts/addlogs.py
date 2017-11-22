import os
import traceback
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from astroML.plotting.mcmc import convert_to_stdev

def addLogs(fname, npixels=200):
	if not os.path.exists(fname):
		print('Oops! There is no such folder')
		return None
	
	currdir = os.getcwd()
	os.chdir(fname)

	try:
		process = subprocess.Popen('ls gridlnlike* > names.dat', shell=True, stdout=subprocess.PIPE)
		fNames = np.genfromtxt('names.dat', dtype=str)

		K = np.zeros((len(fNames), npixels, npixels))
		indices = []


		for i, ele in enumerate(fNames):
			K[i] = np.loadtxt(ele)
			temp = str.split(ele, '_')
			indices.append(int(temp[1][:-4]))

		F = np.sum(K, axis=0)

		x0, x1 = np.mgrid[-3:6:200j, -0.25:0.25:200j]
		
		fig, ax = plt.subplots(1, figsize=(5, 5))
		colormap = plt.cm.rainbow #nipy_spectral, Set1,Paired   
		colors = [colormap(i) for i in np.linspace(0, 1,len(indices))]

		for i in range(len(fNames)):
			CS = ax.contour(x0, x1, convert_to_stdev(K[i]), levels=[0.683, ], linewidths=0.6, colors=(colors[i],))
			CS.collections[0].set_label(indices[i])

		ax.contour(x0, x1, convert_to_stdev(F), levels=[0.683, 0.95], alpha=0.5, colors='k', linestyles='--')
		
		plt.legend()
		plt.show()
	except:
		traceback.print_exc()
		os.chdir(currdir)

	os.chdir(currdir)