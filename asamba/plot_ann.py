from __future__ import print_function
from __future__ import unicode_literals

# from builtins import range
import sys, os, glob
import logging
import numpy as np 
import pylab as plt 
import matplotlib.mlab as mlab

from asamba import star, utils
from asamba import artificial_neural_network as ann

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def marginal_2D(self, wrt_x, wrt_y, figure_name):
  """
  This function provides a 2D filled-contour plot of the marginalized posterior probability with 
  respect to two features, passed as wrt_x and wrt_y. 
  """
  res = self.marginalize_wrt_x_y(wrt_x, wrt_y)
  # arr = utils.list_to_ndarray(res).astype(np.int16)
  tagx= np.array([tup[0] for tup in res], dtype=np.int16)
  tagy= np.array([tup[1] for tup in res], dtype=np.int16)
  z   = np.array([tup[2] for tup in res], dtype=np.float32)
  # tagx, tagy, z = arr[:,0], arr[:,1], arr[:,2]
  x   = self.convert_tags_to_features(tagx, wrt_x)
  y   = self.convert_tags_to_features(tagy, wrt_y)
  x   = np.array(x)
  y   = np.array(y)

  fig, ax = plt.subplots(1, figsize=(4,4), dpi=100, tight_layout=True)
  # ax[0].tricontourf(x, y, z, N=101, cmap=plt.get_cmap('Greys'), norm=None,)
  # ax[0].scatter(x, y, marker='o', facecolor='k', edgecolor='k', s=2)

  nx  = ny = 101
  xi  = np.linspace(min(x), max(x), nx)
  yi  = np.linspace(min(y), max(y), ny)
  zi  = mlab.griddata(x, y, z, xi, yi, interp='linear')
  cf  = ax.contourf(xi, yi, zi, 15, zorder=1, cmap=plt.get_cmap('Greys'),
                       norm=plt.Normalize(vmin=0, vmax=abs(zi).max())) 
  ax.scatter(x, y, marker='o', facecolor='k', edgecolor='k', s=2, zorder=2)
  cb  = fig.colorbar(cf, ax=ax, shrink=0.95)

  ax.set_xlabel(utils.feature_name_in_latex(wrt_x))
  ax.set_ylabel(utils.feature_name_in_latex(wrt_y))

  plt.savefig(figure_name)
  print('marginal_2D: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def all_marginal_1D(self, figure_name):
  """
  The method marginalize() of the class ann.neural_net() stores the marginal tuples of all dimensions
  of the problem into self.marginal_results. E.g., if the problem at hand has six dimensions/features
  (like M_ini, fov, ...), then self.marginal_results will contain six tuples for each of the features.
  This routine makes a multi-panel figure showing the feature arrays on the abscissa and their 
  corresponding marginal probabilities on the ordinate.
  """
  if not self.marginalize_done: return 
  results = self.get('marginal_results')
  n_dim   = len(results)
  n_rows  = 2
  n_cols  = n_dim // n_rows if n_dim % n_rows == 0 else n_dim // n_rows + 1

  fig, tup_ax = plt.subplots(n_rows, n_cols, figsize=(8,6), dpi=100, tight_layout=True)
  arr_ax  = tup_ax.reshape(-1)

  for i_ax in range(n_dim):
    ax    = arr_ax[i_ax]
    _res  = results[i_ax]
    x_marg= np.array([_tup[0] for _tup in _res])
    y_marg= np.array([_tup[1] for _tup in _res])

    ax.plot(x_marg, y_marg, marker='o', linestyle='solid', color='grey', ms=4)

  plt.savefig(figure_name)
  logger.info('all_marginal_1D: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
