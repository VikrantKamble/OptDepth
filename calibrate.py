from __future__  import division
from func_defs import compute_alpha
import fitsio
import numpy as np

# AIM: TO STUDY DEVIATIONS FROM POWER LAW - ESTIMATE CALIBRATION SYSTEMATICS

# DEFINE MASTER COMPOSITE AND NORMALIZE IT CORRECTLY
m_comp = fitsio.read('../Composites/comp_V0_44_21_equalobj/comp_V0_44_21_equalobj.fits')[16]
m_ivar = fitsio.read('../Composites/comp_V0_44_21_equalobj/comp_V0_44_21_equalobj.fits', ext=1)[16]

# Use the range where there are no emission lines
ratio_range = [1250, 1480]


alpha, amp = tb['ALPHA_V0'], tb['AMP_V0']
z = tb[drq_z]

# Get low-z stuff
lowz_ind = np.where((z > 1.6) & (z < 2.3) & (alpha > -2) & (alpha < -1))[0]
lowz_spec, lowz_ivar, lowz = spec[lowz_ind], ivar[lowz_ind], z[lowz_ind]
lowz_amp, lowz_alpha = amp[lowz_ind], alpha[lowz_ind]

# The range over which to normalize 
# # HAS TO BE PUT BY HAND TO CHOOSE EMISSION FREE RANGE
# wlInd = np.where((wl > 2100) & (wl < 2140))[0]

# # Normalize the spectra and the variance 
# scale   = np.median(lowz_spec[:,wlInd], axis=1)
# lowz_spec  = lowz_spec / scale[:, None]
# lowz_ivar  = lowz_ivar * (scale[:,None])**2

# Normalize in amplitude
lowz_spec = lowz_spec / lowz_amp[:, None]
lowz_ivar = lowz_ivar * (lowz_amp[:, None])**2

# Normalize in alpha
lowz_spec = lowz_spec * (wl/1450)**(-1.5 - lowz_alpha[:,None])

# Normalize master composite
m_param = compute_alpha(wl, m_comp, m_ivar, [[1280,1290],[1320,1330],[1345,1360],[1440,1480]], 3)
m_comp = m_comp * (wl/1450)**(-1.5 - m_param[1])


lam_ind = np.where((wl > ratio_range[0]) & (wl <= ratio_range[1]))[0]

vals, weights = lowz_spec[:,lam_ind] / m_comp[lam_ind], lowz_ivar[:,lam_ind]

# Mapping rest-frame pixel to observer frame
lam_obs = np.array(np.mat(wl[lam_ind]).T * np.mat(1+lowz)).T

mylam, myval, myivar = np.ravel(lam_obs), np.ravel(vals), np.ravel(weights)
divs = np.arange(min(mylam), max(mylam), 4)

Z_val = (divs[1:] + divs[:-1]) / 2
T_val, S_val = np.zeros((2, len(Z_val)))

# Bin values using weighted average statistic
for i in range(len(divs)-1):
	ind = np.where((mylam > divs[i]) & (mylam <= divs[i+1]))
	zeros_ind = np.where((myivar[ind] > 0))[0]
	if (len(zeros_ind) > 0):
		T_val[i] = np.median(myval[ind][zeros_ind])
		S_val[i] = np.std(myval[ind][zeros_ind]) #/len(zeros_ind)
		#T_val[i] = np.average(myval[ind][zeros_ind], weights=myivar[ind][zeros_ind])
		