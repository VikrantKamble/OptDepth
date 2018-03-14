import numpy as np
from bin_analyze import analyze
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class binObj:
    """ Class to hold information about a particular bin over which
        optical depth is calculated
    """
    ly_line = 1215.67

    def __init__(self, qso, parNames, parRanges, snt=[2, 50], **kwargs):
        print('Building object')

        if not all(x in qso.tb.dtype.names for x in parNames):
            raise ValueError('The selected feature combination not available. \
                Options available are: {}'.format(qso.tb.dtype.names))

        self.wl = qso.wl
        self.parNames = parNames
        self.parRanges = parRanges

        if len(snt) == 1:
            self.snt_min, self.snt_max = snt, 100
        else:
            self.snt_min, self.snt_max = snt[0], snt[1]

        # set bin data
        _ixs = np.where((qso.sn > self.snt_min) & (qso.sn <= self.snt_max) &
                        (qso.tb[self.parNames[0]] > self.parRanges[0][0]) &
                        (qso.tb[self.parNames[0]] <= self.parRanges[0][1]) &
                        (qso.tb[self.parNames[1]] > self.parRanges[1][0]) &
                        (qso.tb[self.parNames[1]] <= self.parRanges[1][1]))[0]

        self._flux = qso.flux[_ixs]
        self._ivar = qso.ivar[_ixs]
        self._zq = qso.zq[_ixs]
        self._par1 = qso.tb[self.parNames[0]][_ixs]
        self._par2 = qso.tb[self.parNames[0]][_ixs]

        if self.parNames[0] == "ALPHA_V0":
            self._alpha = self._par1
        elif self.parNames[1] == "ALPHA_V0":
            self._alpha = self._par2
        else:
            self._alpha = qso.tb["ALPHA_V0"][_ixs]

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

        print("Be sure to run 'vcorrect' for variance corrections and"
              "'calib_correct' for \ flux calibration corrections")

    def bin_info(self, plot_zabs_dist=False, plot_zabs_kwargs=None):
        """ Information about the bin """

        print("{} : {}".format(self.parNames[0], self.parRanges[0]))
        print("{} : {} \n".format(self.parNames[1], self.parRanges[1]))

        print("Total number of objects in this bin: {} \n".format(self._zq.shape[0]))

        # if plot_zabs_dist:
        #     plotter._hist1d(self.zAbs, **plot_zabs_kwargs)

    def mask_pixels(self, obs_ranges=[3700, 7000]):
        print("Masking pixels outside the range", obs_ranges)
        mask = (self._wAbs < 3700) | (self._wAbs > 7000)
        self._ivar[mask] = 0
        self._flux[mask] = 0

    def vcorrect(self, vcorrect_file):
        # Apply variance corrections
        coeff = np.loadtxt(vcorrect_file)

        eta = np.piecewise(self._wAbs, [self._wAbs < 5850, self._wAbs >= 5850],
                           [np.poly1d(coeff[0:2]), np.poly1d(coeff[2:])])

        self._ivar *= eta
        self.vcorrected = True
        print('Variance corrections applied')

    def calib_correct(self, calib_file='../Data/calib_helion.txt'):
        # Use pixels above 3700 A - 7000 A in observer wavelength
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
        """ Optical depth analysis on a given skewer index"""
        return analyze(self, skewer_index=index, skewer_kwargs=s_kwargs, **a_kwargs)

    def build_fake(self):
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
