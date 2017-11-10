
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.stats import chi2

get_ipython().magic('matplotlib inline')
get_ipython().magic("run './plot_setup.py'")


# # Confidence level from n sigma
# ## $$n\ sig \to CI = erf \left(\frac{n}{\sqrt{2}}\right)$$
# 
# ### ppf gives the inverse cumulative distribution, and supplying it an argument (volume) gives the location that encloses that volume

# In[2]:

"""
Input parameters:
    pos = Mean
    cov = Covariance matrix
    nsig =  the n'th standard deviation to plot
    ax = Axes instance to stick the ellipse tp

"""
def plot_ellipse(pos, cov, nsig, ax, fc='none', ec='k', a=1, lw=2, ls='-'):
    var_x, var_y = cov[0,0], cov[1,1]
    covar_xy = cov[0,1]
    
    sig1 = np.sqrt((var_x + var_y) / 2. + np.sqrt( (var_x - var_y) ** 2 / 4. + covar_xy ** 2))
    sig2 = np.sqrt((var_x + var_y) / 2. - np.sqrt( (var_x - var_y) ** 2 / 4. + covar_xy ** 2))
    
    angle = 0.5 * np.arctan2(2 * covar_xy, (var_x - var_y))
    
    # Plot the results
    from scipy.special import erf

    kwargs = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw, 'linestyle':ls}
    
    for ele in nsig:
        scale = np.sqrt(chi2.ppf(erf(ele / np.sqrt(2)), df=2))
        width = 2 * sig1 * scale
        height = 2 * sig2 * scale
        
        e = Ellipse(xy=pos, width=width, height=height, angle=angle * 180. / np.pi, **kwargs)
        ax.add_patch(e)
        ax.set_xlim(pos[0] - 2 * np.sqrt(var_x), pos[0] + 2 * np.sqrt(var_x))
        ax.set_ylim(pos[1] - 2 * np.sqrt(var_y), pos[1] + 2 * np.sqrt(var_y))



