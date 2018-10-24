import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# local imports
from Scripts.bin_analyze import analyze


class binObj:
    """ Class to hold information about a particular bin over which
        optical depth is calculated
    """
    ly_line = 1215.67

    input_calib = '/Users/vikrant/Work/MyProject/OptDepth/Data/calib_helion.txt'
    input_var_correct = '/Users/vikrant/Work/MyProject/OptDepth/Data/var_correct.txt'

    def __init__(self, name, qso, parNames, parRanges, snt, preprocess=True,
                 **kwargs):
        print('Building object')

        if not all(x in qso.tb.dtype.names for x in parNames):
            raise ValueError('The selected feature combination not available. \
                Options available are: {}'.format(qso.tb.dtype.names))
        self.name = name
        self.wl = qso.wl
        self.parNames = parNames
        self.parRanges = parRanges

        if len(snt) == 1:
            self.snt_min, self.snt_max = snt, 100
        else:
            self.snt_min, self.snt_max = snt[0], snt[1]

        # set bin data
        self._ixs = np.where((qso.sn > self.snt_min) & (qso.sn <= self.snt_max) &
                             (qso.tb[self.parNames[0]] > self.parRanges[0][0]) &
                             (qso.tb[self.parNames[0]] <= self.parRanges[0][1]) &
                             (qso.tb[self.parNames[1]] > self.parRanges[1][0]) &
                             (qso.tb[self.parNames[1]] <= self.parRanges[1][1]))[0]

        self._flux = qso.flux[self._ixs]
        self._ivar = qso.ivar[self._ixs]
        self._zq = qso.zq[self._ixs]
        self._par1 = qso.tb[self.parNames[0]][self._ixs]
        self._par2 = qso.tb[self.parNames[1]][self._ixs]

        if self.parNames[0] == "ALPHA_V0":
            self._alpha = self._par1
        elif self.parNames[1] == "ALPHA_V0":
            self._alpha = self._par2
        else:
            self._alpha = qso.tb["ALPHA_V0"][self._ixs]

        # positions of lyman alpha absorbers along redshift
        print("Creating observer wavelength matrix")

        self._wAbs = np.array((np.mat(self.wl).T * np.mat(1 + self._zq))).T
        self._zAbs = self._wAbs / self.ly_line - 1

        # histogram rebinning weights
        self._hWeights = None

        self.vcorrected = False
        self.calib_corrected = False

        # Used for fake data analysis
        self.f_noise = None
        self.total_var = None
        self.f_true = None

        if preprocess:
            self.mask_pixels()
            self.mask_calcium()
            self.vcorrect()
            self.calib_correct()
        else:
            print("Be sure to run 'vcorrect' for variance corrections and"
                  "'calib_correct' for \ flux calibration corrections")

        print("The number of objects in this bin are %d" % len(self._zq))

    def bin_info(self, plot_zabs_dist=False, plot_zabs_kwargs=None):
        """ Information about the bin """
        # Add extra functionalities here!

        print("{} : {}".format(self.parNames[0], self.parRanges[0]))
        print("{} : {} \n".format(self.parNames[1], self.parRanges[1]))

        print("Total number of objects in this bin: {} \n".format(self._zq.shape[0]))

    def mask_pixels(self, obs_ranges=[3700, 7000]):
        """ Use this to mask pixels in the observer wavelength frame """
        print("Masking pixels outside the range", obs_ranges)
        mask = (self._wAbs < 3700) | (self._wAbs > 7000)
        self._ivar[mask] = 0
        self._flux[mask] = 0

    def mask_calcium(self):
        mask1 = (self._wAbs > 3925) & (self._wAbs < 3942)
        self._ivar[mask1] = 0
        mask2 = (self._wAbs > 3959) & (self._wAbs < 3976)
        self._ivar[mask2] = 0

    def vcorrect(self, vcorrect_file=input_var_correct):
        """ Apply variance corrections to the ivar vector """
        coeff = np.loadtxt(vcorrect_file)

        eta = np.piecewise(self._wAbs, [self._wAbs < 5850, self._wAbs >= 5850],
                           [np.poly1d(coeff[0:2]), np.poly1d(coeff[2:])])

        self._ivar *= eta
        self.vcorrected = True
        print('Variance corrections applied')

    def calib_correct(self, calib_file=input_calib):
        """ Use this to correct for flux calibrations in BOSS/eBOSS """
        calib_file = np.loadtxt(calib_file)
        interp_func = interp1d(calib_file[:, 0], calib_file[:, 1],
                               bounds_error=False)

        # Corrections over the observer wavelength frame
        interp_mat = interp_func(self._wAbs)

        self._flux /= interp_mat

        # Veto Calcium H and K lines
        h_mask = (self._wAbs > 3927) & (self._wAbs < 3940)
        k_mask = (self._wAbs > 3965) & (self._wAbs < 3974)

        self._ivar[h_mask] = 0
        self._ivar[k_mask] = 0
        print('Flux calibration corrections applied')

    def analyze(self, index, a_kwargs={}, s_kwargs={}):
        """ Optical depth analysis on a given skewer index
            Use bin_analyze directly to be more explicit in the parameters passed
        """
        return analyze(self, skewer_index=index, skewer_kwargs=s_kwargs, **a_kwargs)

    def build_fake(self):
        """ Build fake lyman alpha forest spectra.
        These are not totally correct since we aren't seeding the perturbations
        from a Gaussian field. Instead each pixel is treated independent with
        contributions comming from LSS variance and measurement noise.
        ie. Only xi(r=0) information used.
        """
        # Initialize array for doing tests on fake data
        lnt0_t, gamma_t = -6, 3.5

        print("Creating fake model")

        def Gauss(x, amp, cen, scale):
            return amp * np.exp(-0.5 * (x - cen) ** 2 / scale ** 2)

        C = (self.wl / 1450.) ** -1.8 + Gauss(self.wl, 0.18, 1123, 11) +\
            Gauss(self.wl, 0.22, 1069, 14) + Gauss(self.wl, 0.9, 1215.67, 30)

        T = np.exp(-np.exp(lnt0_t) * (self._zAbs + 1) ** gamma_t)
        self.f_true = C * T

        print("Create noise matrix")
        var_LSS = 0.065 * ((1 + self._zAbs)/3.25) ** 3.8 * self.f_true ** 2
        self.total_var = var_LSS + 1.0 / self._ivar

        # Sanity checks
        rInd = np.random.randint(len(self._zq), size=10)

        plt.figure()
        plt.plot(self.wl, self.f_true[rInd].T, lw=0.3)
        plt.show()

    def gen_fake(self, seed=0):
        """ Generate fake lyman alpha forest spectra """
        np.random.seed(seed)

        # Sample the noise
        sigmas = np.random.normal(0, np.sqrt(self.total_var))

        # noisy flux values
        self.f_noise = self.f_true + sigmas

        # Sanity checks
        rInd = np.random.randint(len(self._zq), size=10)

        plt.figure()
        plt.plot(self.wl, self.f_noise[rInd].T, lw=0.3)
        plt.show()

    def test_fake(self, index, a_kwargs={}, s_kwargs={}):
        return analyze(self, test_fake=True, distort=False,
                       skewer_index=index, skewer_kwargs=s_kwargs, **a_kwargs)
