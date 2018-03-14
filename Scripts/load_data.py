#!/bin/python
"""
1. Script that defines a QSO base class
2. Has all the properties as is contained in config_read.py
3. Feel free to add attributes and methods to improve functionality
"""
from configparser import ConfigParser

import json
import numpy as np
import sys

import fitsio
import param_fit


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
        self.scale = self.tb['SCALE']

        X = ConfigMap('make_comp')
        if X == 'FAIL':
            sys.exit("Errors! Couldn't locate section in .ini file")

        # Overall cuts and specific bin selection - 3 * 3
        self.f_cuts = json.loads(X['f_cuts'])

        # Grid for histogram rebinning
        self.n_chop = json.loads(X['n_chop'])

        # SN calculated over the range same as that used for spectral index
        self.chisq_min, self.chisq_max = float(X['chisqr_min']), float(X['chisqr_max'])

        # The reduced chisquares of the respective feature fitting
        # chisq_p1 = tb[features[0] + '_CHISQ_R']
        # chisq_p2 = tb[features[1] + '_CHISQ_R']

        self.wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)

        # Indicator flags
        self.flux = None
        self.ivar = None

        self.vcorrected = False
        self.scaled = False

    def load_data(self, cfile, scaling=True):
        """ load spectra data """
        print('Catalog file used is: ' + cfile)
        self.flux = fitsio.read(cfile, ext=0)
        self.ivar = fitsio.read(cfile, ext=1)
        print('Number of objects in the catalog are: %d \n' %(len(self.flux)))

        if scaling:
            print('Scaling spectra now')

            self.flux /= self.scale[:, None]
            self.ivar *= (self.scale[:, None]) ** 2

            # Remove objects where scale is negative that leads to inverted spectra
            ind = np.where((self.scale < 0) & (np.isnan(self.sn)))[0]
            self.ivar[ind] = 0
            self.scaled = True

    def set_catalog(self, cfile):
        """ set a custom catalog file, need to run load_data"""
        self.catalog_file = cfile
        self.load_data(cfile)

    def get_alpha(self, index):
        """calculate spectral index for a given index"""
        if index == 'all':
            res = param_fit.process_all(self.flux, self.ivar, self.zq, param='alpha')
        else:
            res = param_fit.fit_alpha(self.flux[index], self.ivar[index],
                                      self.zq[index], plotit=True)
        return res

    def get_EW(self, index):
        """calculate C IV equivalent width for a given index"""
        if index == 'all':
            res = param_fit.process_all(self.flux, self.ivar, self.zq, param='EW')
        else:
            res = param_fit.fit_EW(self.flux[index], self.ivar[index], self.zq[index], plotit=True)
        return res

    def get_scale(self):
        """Scaling constant to bring spectra on equal footing"""
        pass
