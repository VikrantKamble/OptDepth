#!/bin/python

"""
1. Script that defines a QSO base class
2. Has all the properties as is contained in config_read.py 

3. Feel free to add attributes and methods to improve functionality
"""

from configparser import ConfigParser
from scipy.interpolate import interp1d
import fitsio
import numpy as np
import json
import imp


parser = ConfigParser()
parser.read('config.ini')

print(parser.sections())


# define a helper function to recover all the options in a given section
def ConfigMap(section):
    dict1 = {}
    options = parser.options(section)
    for option in options:
        dict1[option] = parser.get(section, option)
        if not dict1[option]:
            print('Got an empty string! Please check your .ini file')
            return 'FAIL'
    return dict1


class QSO:
	def __init__(self):
		G = ConfigMap('GLOBAL')
		if G == 'FAIL':
			sys.exit("Errors! Couldn't locate section in .ini file")

		self.catalog_file = G['catalog']

		# Which redshift estimate to use?
		self.drq_z = G['drq_z']
		self.zmin = G['zmin']

		# tb is the table that has all features of any given spectra
		attributes_file = G['attributes']

		print('File used to obtain the spectra attributes is:', attributes_file)
		self.tb = fitsio.read(attributes_file, ext=-1)

		self.zq, self.sn = self.tb[self.drq_z], self.tb['SN']
		self.scale, self.alpha = self.tb['SCALE'], self.tb['ALPHA_V0']

		X = ConfigMap('make_comp')
		if X == 'FAIL':
			sys.exit("Errors! Couldn't locate section in .ini file")

		# Load in the basis parameters with their rough span
		self.features = X['features'].split(',')
		self.f_span = json.loads(X['f_span'])

		# Overall cuts and specific bin selection - 3 * 3
		self.f_cuts = json.loads(X['f_cuts'])
		self.f_sel = json.loads(X['f_sel'])

		# Grid for histogram rebinning
		self.n_chop = json.loads(X['n_chop'])

		# SN calculated over the range same as that used for spectral index
		self.chisq_min, self.chisq_max = float(X['chisqr_min']), float(X['chisqr_max'])

		# The reduced chisquares of the respective feature fitting
		# chisq_p1 = tb[features[0] + '_CHISQ_R']
		# chisq_p2 = tb[features[1] + '_CHISQ_R']

		self.p0_bins = np.linspace(self.f_span[0][0], self.f_span[0][1], self.f_cuts[0])
		self.p1_bins = np.linspace(self.f_span[1][0], self.f_span[1][1], self.f_cuts[1])

		# Load in the basis parameters
		self.p1, self.p2 = self.tb[self.features[0]], self.tb[self.features[1]]
		print('%s and %s have been loaded as parameters p1 and p2' % (self.features[0], self.features[1]))

		self.wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)

		# Indicator flags
		self.vcorrected = False
		self.scaled = False

	def load_data(self, cfile):
		print('Catalog file used is: ' + cfile)
		self.flux = fitsio.read(cfile, ext=0)
		self.ivar = fitsio.read(cfile, ext=1)

		# VERY BAD QUASAR
		self.ivar[3576] = 0
		self.ivar[139526] = 0

		print('Number of objects in the catalog are: %d' %(len(self.flux)))

	def set_catalog(self, cfile):
		self.catalog_file = cfile
		load_data(cfile)

	def vcorrect(self, vcorrect_file):
		# Apply variance corrections 
		x, y = np.loadtxt(vcorrect_file)
		f = interp1d(x, y)

		wl_obs = np.array((np.mat(self.wl).T * np.mat(1 + self.zq))).T
		mask = (wl_obs < 3600) | (wl_obs > 7490)

		wl_obs[mask] = np.nan
		corrections = f(wl_obs)

		self.ivar *= corrections
		self.ivar[mask] = 0
		self.vcorrected = True
		print('Applying variance corrections completed')

	def scale_spec(self):
		self.flux /= self.scale[:, None]
		self.ivar *= (self.scale[:, None]) ** 2
		
		ind = np.where(self.scale < 0)[0]
		self.ivar[ind] = 0

		self.scaled = True
