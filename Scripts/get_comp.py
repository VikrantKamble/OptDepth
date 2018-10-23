import numpy as np
from scipy.stats import binned_statistic_2d


def get_comp(flux, ivar, zLyf, wl, frange, nboot=200):
    # Select the forest
    ixs = (wl > frange[0]) & (wl < frange[1])
    loc_flux, loc_ivar, loc_zLyf = flux[:, ixs], ivar[:, ixs], zLyf[:, ixs]

    n_qso, n_skewers = loc_flux.shape

    # We are going to bin in bins of uniform width in the
    # lyman-alpha forest redshift space
    biny = np.arange(2.0, 4.61, 0.2)

    # define a mask to select pixels that provide information
    mask = loc_ivar > 0

    def sig_func(a, threshold=30):
        if len(a) < threshold:
            return np.inf
        else:
            """ Bootstrap and return the approximate error on the mean """
            a_boot = np.random.choice(a, replace=True, size=(nboot, len(a)))
            return np.std(np.mean(a_boot, axis=1))

    # Each skewer is its own bin
    x = np.arange(n_skewers)[np.newaxis, :]
    x = x.repeat(repeats=n_qso, axis=0)

    binx = np.arange(n_skewers + 1)

    # Applying the bin_statistic with above user-defined functions
    # Default statistic is mean
    mu = binned_statistic_2d(x[mask], loc_zLyf[mask], loc_flux[mask],
                             bins=[binx, biny])

    sig = binned_statistic_2d(x[mask], loc_zLyf[mask], loc_flux[mask],
                              statistic=sig_func, bins=[binx, biny])

    return mu.statistic, sig.statistic
