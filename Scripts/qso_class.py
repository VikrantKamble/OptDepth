#!/bin/python
"""
1. Script that defines a QSO base class
2. Has all the properties as is contained in config_read.py
3. Feel free to add attributes and methods to improve functionality
"""
from configparser import ConfigParser
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import fitsio

# Relative imports
import Scripts
from Scripts import param_fit

PARSER = ConfigParser()
PARSER.read(Scripts.__path__[0] + '/config.ini')

print(PARSER.sections())


# define a helper function to recover all the options in a given section
def config_map(section):
    dict1 = {}
    options = PARSER.options(section)
    for option in options:
        dict1[option] = PARSER.get(section, option)
        if not dict1[option]:
            print('Got an empty string! Please check your .ini file')
            return 'FAIL'
    return dict1


class QSO:
    def __init__(self, data_file=None, attr_file=None):
        G = config_map('GLOBAL')
        if G == 'FAIL':
            sys.exit("Errors! Couldn't locate section in .ini file")

        if data_file is None:
            self.catalog_file = G['catalog']
        else:
            self.catalog_file = data_file

        # Which redshift estimate to use?
        self.drq_z = G['drq_z']

        # tb is the table that has all features of any given spectra
        if attr_file is None:
            attr_file = G['attributes']

        self.tb = fitsio.read(attr_file, ext=-1)
        self.zq, self.sn = self.tb[self.drq_z], self.tb['SN']
        self.scale = self.tb['SCALE']

        X = config_map('make_comp')
        if X == 'FAIL':
            sys.exit("Errors! Couldn't locate section in .ini file")

        # Overall cuts and specific bin selection - 3 * 3
        self.f_cuts = json.loads(X['f_cuts'])

        # Grid for histogram rebinning
        self.n_chop = json.loads(X['n_chop'])

        # SN calculated over the range same as that used for spectral index
        self.chisq_min, self.chisq_max = float(X['chisqr_min']), float(X['chisqr_max'])

        self.wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)

        # Indicator flags
        self.flux = None
        self.ivar = None
        self.vcorrected = False
        self.scaled = False

    def manual_cut(self, ind):
        """
        Manually remove some 'malicious' spectra
        """
        self.ivar[ind] = 0

    def load_data(self, cfile, scaling=True, scale_type='median',
                  col_name='F0_V2'):
        """
        Paramters
        ---------
        scaling   : flag whether to normalize the spectra
        scale_type: type of scaling to use 'median' or 'alpha_fit'
        col_name  : if scale_type == 'alpha_fit', the column name
                    of attr_file to use
        """

        print('Catalog file used is: ' + cfile)
        self.flux = fitsio.read(cfile, ext=0)
        self.ivar = fitsio.read(cfile, ext=1)
        print("Number of objects in the catalog "
              "are: %d \n" %(len(self.flux)))

        if scaling:
            if scale_type == 'median':
                scale = self.scale
            elif scale_type == 'alpha_fit':
                scale = self.tb[col_name]
            else:
                raise ValueError("Scaling type not understood. Must be either"
                                 "'median' or 'alpha_fit'")

            print("Scaling spectra using %s" % scale_type)
            self.flux /= scale[:, None]
            self.ivar *= scale[:, None] ** 2

            # Remove objects where scale is negative or nan that leads to
            # inverted spectra and also where signal-to-noise is nan
            ind = np.where((scale < 0) | (np.isnan(scale)) | (np.isnan(self.sn)))[0]
            self.ivar[ind] = 0
            self.scaled = True

        # Remove bad spectra manually
        bad_spec = [4691]
        self.manual_cut(bad_spec)

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

    def get_ew(self, index, ax=None):
        """calculate C IV equivalent width for a given index"""
        if index == 'all':
            res = param_fit.process_all(self.flux, self.ivar, self.zq, param='EW')
        else:
            if ax is None:
                fig, ax = plt.subplots(1)
            res = param_fit.fit_EW(self.flux[index], self.ivar[index], self.zq[index], plotit=True, ax=ax)
        return res

    def get_scale(self):
        """Scaling constant to bring spectra on equal footing"""
        pass
