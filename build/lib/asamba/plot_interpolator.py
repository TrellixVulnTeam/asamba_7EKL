from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
import sys, os, glob
import logging
import numpy as np 
import pylab as plt 

from asamba import star, interpolator

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def input_frequencies_wrt(self, wrt, figure_name):
  """
  Show the coarse frequencies over the input points (i.e. coming from the database), before doing the
  interpolation

  @param self: an instance of the interpolation() class
  @type self: obj
  @param figure_name: full path to save the resulting figure
  @type figure_name: str
  """
  if not self.interp_inputs_OK:
    logger.error('input_frequencies_wrt: interp_inputs_OK is False!')
    return False

  names    = self.get('interp_param_names')
  if wrt not in names:
    logger.error('input_frequencies_wrt: wrt="{0}" is not among the interpolation attributes'.format(wrt))
    return False 

  features = self.get('input_features')
  freqs    = self.get('input_frequencies')
  
  # Deduce the parameter values for the anchor model
  anc_names= self.get('anchor_param_names')
  anc_vals = self.get('anchor_param_values')
  anc_freqs= self.get('anchor_frequencies')

  i_M_ini  = anc_names.index('M_ini')
  i_fov    = anc_names.index('fov')
  i_Z      = anc_names.index('Z')
  i_logD   = anc_names.index('logD')
  i_Xc     = anc_names.index('Xc')
  i_eta    = 0
  anc_M_ini= anc_vals[i_M_ini]
  anc_fov  = anc_vals[i_fov]
  anc_Z    = anc_vals[i_Z]
  anc_logD = anc_vals[i_logD]
  anc_Xc   = anc_vals[i_Xc]
  if self.exclude_eta_column:
    i_eta  = -1
    anc_eta= 0.0
  else:
    i_eta  = anc_names.index('eta')
    anc_eta= anc_vals[i_eta]

  # Find indexes of all models that have attribute values (independently) almost equal to that of the anchor
  tol      = 1e-4
  n_rows   = features.shape[0]
  _0       = np.zeros(n_rows)
  _1       = np.ones(n_rows)
  ind_M_ini= np.where(np.abs(features[:,i_M_ini] - anc_M_ini) < tol, _1, _0)
  ind_fov  = np.where(np.abs(features[:,i_fov] - anc_fov) < tol, _1, _0)
  ind_Z    = np.where(np.abs(features[:,i_Z] - anc_Z) < tol, _1, _0)
  ind_logD = np.where(np.abs(features[:,i_logD] - anc_logD) < tol, _1, _0)
  ind_Xc   = np.where(np.abs(features[:,i_Xc] - anc_Xc) < tol, _1, _0)
  
  # Do an AND operation by multiplication to find indexes of models which have identical attributes to anchor
  # The "wrt" is excluded here, so, we must get more than one parameter 
  ind_prod = np.ones(n_rows)
  for name in names:
    if name == wrt: continue
    if name == 'M_ini': ind_prod *= ind_M_ini
    if name == 'fov':   ind_prod *= ind_fov
    if name == 'Z':     ind_prod *= ind_Z
    if name == 'logD':  ind_prod *= ind_logD
    # if name == 'Xc':    ind_prod *= ind_Xc

  # eta needs special treatment to handle the rotating vs. non-rotating cases
  if not self.exclude_eta_column:
    ind_eta  = np.where(np.abs(features[:,i_eta] - anc_eta) < tol, _1, _0)
    ind_prod *= ind_eta

  # now, convert the 0/1 values in the ind_prod to an index array
  ind      = np.where(ind_prod == 1)[0]

  n_ind    = len(ind)
  if n_ind == 0:
    logger.error('input_frequencies_wrt: No matching rows found!')
    return False
  features = features[ind]
  freqs    = freqs[ind]
  print('shape freqs for wrt={0}: {1}'.format(wrt, freqs.shape))

  i_wrt    = names.index(wrt)
  arr_wrt  = features[:, i_wrt]

  fig, ax  = plt.subplots(1, figsize=(6,4))
  plt.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.12)

  num_modes= freqs.shape[-1]
  for i in range(num_modes):
    xvals  = arr_wrt
    yvals  = freqs[:, i]

    ax.scatter(xvals, yvals, marker='o', s=4, edgecolor='none', facecolor='grey', zorder=1)

  # stack the anchor frequencies on top of other points too for comparison
  n_freq   = len(anc_freqs)
  anc_wrt  = anc_vals[anc_names.index(wrt)]
  xvals    = np.ones(n_freq) * anc_wrt
  yvals    = anc_freqs
  ax.scatter(xvals, yvals, marker='o', s=10, edgecolor='k', facecolor='none', zorder=2)

  xlabel   = wrt if '_' not in wrt else wrt.replace('_', '')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(r'Frequency [d$^{-1}$]')

  plt.savefig(figure_name)
  logger.info('input_frequencies_wrt: saved: {0}'.format(figure_name))
  plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
