from __future__ import print_function
from __future__ import unicode_literals

# from builtins import range
import sys, os, glob
import logging
import numpy as np 
from scipy import interpolate
import pylab as plt 
import matplotlib.ticker as ticker
# import matplotlib.mlab as mlab

from asamba import star, utils
from asamba import machine_learning as ml

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

marg_contour_levels = 11
logD_ticks = ['None', 'Low', 'Med', 'High', 'Max']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def corner(self, figure_name):
  """
  Create a corner plot (similar in idea to https://pypi.python.org/pypi/corner/1.0.0) to demonstrate
  the 1D and 2D marginalized posterior probabilities
  """
  names = self.get('feature_names')
  n     = len(names)
  fig, axes = plt.subplots(n, n, figsize=(6, 6), tight_layout=False)

  # Put 1D marginalized probability distributions on the diagonal panels
  if not self.marginalize_done: self.marginalize()
  marg_1d = self.get('marginal_results')

  # skip the upper rectangle
  fig.patch.set_visible(False)
  for iax in range(n-1):
    for jax in range(iax+1, n):
      ax    = axes[iax, jax]
      # ax.patch.set_visible(False)
      ax.axis('off')

  # Plot 1D posteriors on the diagonal of the figure
  list_xlim = []
  for iax in range(n):
    var   = names[iax]
    ax    = axes[iax, iax]

    _res  = marg_1d[iax]
    x_1d  = np.array([_tup[0] for _tup in _res])
    y_1d  = np.array([_tup[1] for _tup in _res])
    _n    = len(x_1d)
    widths= np.zeros(_n) 
    widths[:-1] = x_1d[1:] - x_1d[:-1]
    widths[-1]= widths[-2]
    lefts = x_1d - widths / 2.0

    if iax != n-1:
      ax.bar(x_1d, y_1d, widths, align='center', fill=False, ec='b', ls='solid', lw=1)
    else:
      ax.barh(x_1d, y_1d, widths, align='center', fill=False, ec='b', ls='solid', lw=1)
      plt.gca().invert_yaxis()

    ax.set_xticklabels(())
    ax.set_yticklabels(())

    xlims = (min(x_1d) * 0.98, max(x_1d) * 1.02)
    # ax.set_xlim(*xlims)
    list_xlim.append( xlims )

  # Plot 2D posterior maps on the lower panels
  for jax in range(n):
    wrt_x   = names[jax]
    xlims   = list_xlim[jax]
    for iax in range(jax+1, n):
      wrt_y   = names[iax]

      (x, y, xi, yi, zi) = marginal_2D(self, wrt_x, wrt_y, None)
      
      # Plotting
      ax = axes[iax, jax]
      lev= marg_contour_levels
      cf = ax.contourf(xi, yi, zi, lev, zorder=1, cmap=plt.get_cmap('Greys'),
                       norm=plt.Normalize(vmin=0, vmax=abs(zi).max())) 

      ax.scatter(x, y, marker=',', facecolor='k', edgecolor='none', s=1, 
                       alpha=0.5, zorder=2)
      # Axis cosmetics
      if jax == 0:   ax.set_ylabel(utils.feature_name_in_latex(wrt_y))
      if jax > 0:    ax.set_yticklabels(())
      if jax == 0:
        for item in ax.get_yticklabels(): 
          item.set_rotation(45)
          item.set_fontsize('x-small')
      if jax == 0:
        if wrt_y == 'fov': ax.yaxis.set_ticks(np.unique(y)[::2])
        if wrt_y == 'Z':   ax.yaxis.set_ticks(np.unique(y)) 
        if wrt_y == 'Xc':  ax.yaxis.set_ticks(np.linspace(0, 0.71, 5))
        if wrt_y == 'logD':
          ax.yaxis.set_ticks(range(5))
          ax.set_yticklabels(logD_ticks, rotation=45, fontsize='x-small')

      if iax < n-1:  ax.set_xticklabels(())
      if iax == n-1: ax.set_xlabel(utils.feature_name_in_latex(wrt_x))
      if iax == n-1: 
        for item in ax.get_xticklabels(): 
          item.set_rotation(45)
          item.set_fontsize('x-small')
      if iax == n-1: 
        if wrt_x == 'fov': ax.xaxis.set_ticks(np.unique(x)[::2])
        if wrt_x == 'Z':   ax.xaxis.set_ticks(np.unique(x)) 
        if wrt_x == 'Xc':  ax.xaxis.set_ticks(np.linspace(0, 0.71, 5))
        if wrt_x == 'logD':
          ax.xaxis.set_ticks(range(5))
          ax.set_xticklabels(logD_ticks, rotation=45, fontsize='x-small')
      # ax.set_xlim(*xlims)

  plt.savefig(figure_name, transparent=True)
  logger.info('corner: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def marginal_2D(self, wrt_x, wrt_y, figure_name=None):
  """
  This function provides a 2D filled-contour plot of the marginalized posterior probability with 
  respect to two features, passed as wrt_x and wrt_y. 
  """
  res = self.marginalize_wrt_x_y(wrt_x, wrt_y)
  tagx= np.array([tup[0] for tup in res], dtype=np.int16)
  tagy= np.array([tup[1] for tup in res], dtype=np.int16)
  z   = np.array([tup[2] for tup in res], dtype=np.float32)

  if wrt_x == 'logD':
    x = tagx[:]
  else:
    x = self.convert_tags_to_features(tagx, wrt_x)
  if wrt_y == 'logD':
    y = tagy[:]
  else:
    y = self.convert_tags_to_features(tagy, wrt_y)

  nx  = ny = 101
  xi  = np.linspace(min(x), max(x), nx)
  yi  = np.linspace(min(y), max(y), ny)
  xi_ = xi[None,:]
  yi_ = yi[:,None]
  # zi0 = mlab.griddata(x, y, z, xi, yi, interp='linear')
  zi  = interpolate.griddata((x, y), z, (xi_, yi_), method='cubic')
  zi[np.isnan(zi)] = 0.0
  max_zi   = np.max(zi)
  zi_low   = max_zi / 100.0
  ind_zero = zi <= zi_low
  zi[zi <= ind_zero] = 0

  if figure_name is None: return (x, y, xi, yi, zi)

  # Continue making the figure
  fig, ax = plt.subplots(1, figsize=(4,4), dpi=100, tight_layout=True)

  lev = marg_contour_levels
  # c   = ax.contour(xi, yi, zi, lev, linestyle='dotted', linewidth=0.5, color='k')
  cf  = ax.contourf(xi, yi, zi, lev, zorder=1, cmap=plt.get_cmap('Greys'),
                       norm=plt.Normalize(vmin=0, vmax=abs(zi).max())) 
  ax.scatter(x, y, marker=',', facecolor='grey', edgecolor='grey', s=1, zorder=2)
  cb  = fig.colorbar(cf, ax=ax, shrink=1.00)

  ax.set_xlabel(utils.feature_name_in_latex(wrt_x))
  ax.set_ylabel(utils.feature_name_in_latex(wrt_y))

  if wrt_x == 'logD': 
    ax.xaxis.set_ticks(range(5))
    ax.set_xticklabels(logD_ticks, rotation=45, fontsize='small')

  if wrt_y == 'logD': 
    ax.yaxis.set_ticks(range(5))
    ax.set_yticklabels(logD_ticks, rotation=45, fontsize='small')

  if figure_name is not None:
    plt.savefig(figure_name)
    logger.info('marginal_2D: saved {0}'.format(figure_name))
    plt.close()

  return (x, y, xi, yi, zi)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def all_marginal_1D(self, figure_name):
  """
  The method marginalize() of the class ml.neural_net() stores the marginal tuples of all dimensions
  of the problem into self.marginal_results. E.g., if the problem at hand has six dimensions/features
  (like M_ini, fov, ...), then self.marginal_results will contain six tuples for each of the features.
  This routine makes a multi-panel figure showing the feature arrays on the abscissa and their 
  corresponding marginal probabilities on the ordinate.
  """
  if not self.marginalize_done: self.marginalize()
  results = self.get('marginal_results')
  names   = self.get('feature_names')
  n_dim   = len(results)
  n_rows  = 2
  n_cols  = n_dim // n_rows if n_dim % n_rows == 0 else n_dim // n_rows + 1

  fig, tup_ax = plt.subplots(n_rows, n_cols, figsize=(6.5, 5), dpi=100, tight_layout=True)
  arr_ax  = tup_ax.reshape(-1)

  for i_ax in range(n_dim):
    wrt   = names[i_ax]
    ax    = arr_ax[i_ax]
    _res  = results[i_ax]
    x_tag = np.array([_tup[0] for _tup in _res])
    prob  = np.array([_tup[1] for _tup in _res])

    x_marg= self.convert_tags_to_features(x_tag, wrt) if wrt != 'logD' else x_tag

    ax.plot(x_marg, prob, marker='o', linestyle='solid', color='grey', ms=4)

    # ax.set_xlabel(utils.feature_name_in_latex(wrt))
    ax.set_xlabel(utils.feature_name_in_layman(name=wrt, short=False))

    # Cosmetics
    if wrt == 'logD': 
      ax.xaxis.set_ticks(range(5))
      ax.set_xticklabels(logD_ticks, rotation=45, fontsize='small')

    if wrt == 'fov': ax.xaxis.set_ticks(np.unique(x_marg)[::2])
    if wrt == 'Z':   ax.xaxis.set_ticks(np.unique(x_marg)) 
    if wrt == 'Xc':  
      ax.xaxis.set_ticks(np.linspace(0, 0.71, 5))
      for item in ax.get_xticklabels(): item.set_rotation(45)

  if self.exclude_eta_column:
    ax     = arr_ax[-1]
    ax.set_axis_off()

  plt.savefig(figure_name)
  logger.info('all_marginal_1D: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def show_MAP_frequencies(self, figure_name):
  """
  Compare the frequencies of the maximum a posteriori model with the observations. This will graphically
  show how well we are able to reproduce the observations given our fixed grid of models.
  """
  if not self.get('MAP_done'): return

  modes = self.get('modes')  
  obs_freq = np.array([mode.freq for mode in modes]) # unit: per day
  d_freq   = obs_freq[1:] - obs_freq[:-1]

  # MAP_names = self.get('feature_names')
  # MAP_features = self.get('MAP_feature')
  MAP_freq  = self.get('MAP_frequencies')

  fig, ax = plt.subplots(1, figsize=(4,3), tight_layout=True)

  for k, freq in enumerate(obs_freq):
    ax.axvline(x=freq, ymin=0, ymax=1, linestyle='solid', color='r', lw=2)
    ax.axvline(x=MAP_freq[k], ymin=0, ymax=0.8, linestyle='dashed', color='k', lw=1)

  # Axis cosmetics
  ax.set_ylim(0, 1.2)
  ax.set_yticklabels(())
  ax.set_xlabel(r'Frequency (per day)')
  # Legend
  ax.plot([], [], linestyle='solid', lw=4, color='r', label='Observed')
  ax.plot([], [], linestyle='dashed', lw=2, color='k', label='Model')
  leg = ax.legend(loc=1)

  plt.savefig(figure_name)
  logger.info('show_MAP_frequencies: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
