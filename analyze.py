#!/usr/bin/python

"""
Calculates the Optical Depth for all the realizations using multiprocessing library
Input configuration parameters read from the file 'config.ini'

"""

import numpy as np
import multiprocessing as mp
import json
import sys
import os
from functools import partial
import traceback
import timeit
import subprocess

import config_read as cfg
import calc_tau
reload(cfg)

def main(fname , savefile, trim_obs_lam):

	X = cfg.ConfigMap('CalcTau')

	ly_range = json.loads(X['ly_range'])
	ly_line = float(X['line'])
	z_norm = float(X['z_norm'])
	zdiv = int(X['zdiv'])

	print 'The normalization redshift is %.2f' %z_norm

	prev_dir = os.getcwd()
	
	start = timeit.default_timer() # start the timer

	try:
		os.chdir(os.environ['OPT_COMPS'] + fname)

		mastername = fname + '.fits'
		#process = subprocess.Popen('ls *boot* > temp', shell=True, stdout=subprocess.PIPE)

		# The redshifts and relative optical depth from the main file - WILL BE APPENDED LATER		
		M_z, M_tau, M_zbins = calc_tau.calc_tau(mastername, ly_range, ly_line, z_norm, False, zdiv, trim_obs_lam = trim_obs_lam)

		# This is the iterable for pool
		NameFile = np.loadtxt('temp', 'str')
		print 'The number of bootstrap realizations are : %d' %len(NameFile)

		if len(NameFile) == 0:
			return 'Error. The file is empty!!'

		myfunc = partial(calc_tau.calc_tau, ly_range = ly_range, ly_line = ly_line, zn = z_norm, zset = True, zdiv = M_zbins, trim_obs_lam = trim_obs_lam) 

		# Invoke multiprocessing

		pool = mp.Pool()
		result = pool.map(myfunc, list(NameFile))

		pool.close()
		pool.join()

		result.insert(0, M_tau)
		result.insert(0, M_z)

		stop = timeit.default_timer() # stop the timer
		print 'The time taken to do the calculations is', stop - start

		np.savetxt(savefile, result) # save to file for future use
		
		os.chdir(prev_dir)

		return np.array(result)

	except:
		os.chdir(prev_dir)
		traceback.print_exc()

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3])


