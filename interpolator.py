
"""
This module provides interpolation between frequencies over a give range of input stellar parameters. With this tool, it is no longer needed to compute too highly-resolved grids around the best asteroseismic models. Instead, the resolved models are prepared by the interpolation in between the grid points from the coarse model.
"""

import sys, os, glob
import logging
import numpy as np
from scipy.interpolate import griddata

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    ###  ### ######  ###     ###   #####
    #    #    #    #  #     #  #       #   #     #
    #     #   #    #  #     #  #       #  #
    #    #    #    #  ######   #       #  #
    #####     #    #  #     #  #       #  #
    #         #    #  #     #  #    #  #   #     #
    #          ####   ######  ####### ###   #####


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class interpolation(object):
  """
  The base class for internal interpolation means.

  If interp_... is True, then that parameter will be interpolated
  from interp_..._from to interp_..._to, in interp_..._steps number of 
  meshpoints, including the last point (i.e. interp_..._to).
  """
  def __init__(self):

    #.............................
    # The Sampled data         
    #.............................
    # Matrix of features, similar to sampling.learning_x
    self.original_x = []
    # Matrix of frequencies, similar to sampling.learning_y
    self.original_y = []
    # Matrix of radial orders, similar to sampling.learning_radial_orders
    self.original_n_pg = []
    # Matrix of mode types, similar to sampling.learning_mode_types
    self.original_mode_types = []

    #.............................
    # Anchor (or the best) model. One GYRE ouput model
    # already exists for the anchor model. 
    #.............................
    # Names of the parameters/features
    # similar to sampling.feature_names
    self.anchor_param_names = []
    # Values for the parameters of anchor model
    # similar to neural_net.marginal_features
    self.anchor_param_values = []
    # Frequencies corresponding to the anchor parameters,
    # coming from neural_net.MAP_frequencies
    self.anchor_frequencies = []
    # Radial orders corresponding to each anchor frequency
    self.anchor_radial_orders = []
    # Mode identification corresponding to each anchor
    # frequency, as defined in grid.sql schema file
    self.anchor_mode_types = []

    #.............................
    # How to collect inputs?
    #.............................
    # Query around the anchor model
    self.inputs_around_anchor = False
    # if True, then, specify n number of points to 
    # each side of the anchor model, thus ending with 2n+1 points
    self.inputs_around_anchor_M_ini_n = 0
    self.inputs_around_anchor_fov_n   = 0
    self.inputs_around_anchor_Z_n     = 0
    self.inputs_around_anchor_logD_n  = 0
    self.inputs_around_anchor_Xc_n    = 0
    self.inputs_around_anchor_eta_n   = 0

    # Query in a range of models
    self.inputs_by_range = False

    #.............................
    # Input features and frequencies
    # from the grid (actual GYRE outputs)
    # m: number of rows
    # n: number of features
    # K: number of frequencies per row
    #.............................
    # The ndarray of input features; shape: (m, n)
    self.input_features     = []
    # The ndarray of input frequencies; shape: (m, K)
    self.input_frequencies  = []

    #.............................
    # Specifications for interpolation
    #.............................
    # Parameter ranges, and stepsizes
    self.interp_M_ini       = False
    self.interp_M_ini_from  = 0
    self.interp_M_ini_to    = 0
    self.interp_M_ini_steps = 0

    self.interp_fov         = False
    self.interp_fov_from    = 0
    self.interp_fov_to      = 0
    self.interp_fov_steps   = 0

    self.interp_Z           = False
    self.interp_Z_from      = 0
    self.interp_Z_to        = 0
    self.interp_Z_steps     = 0

    self.interp_logD        = False
    self.interp_logD_from   = 0
    self.interp_logD_to     = 0
    self.interp_logD_steps  = 0

    self.interp_Xc          = False
    self.interp_Xc_from     = 0
    self.interp_Xc_to       = 0
    self.interp_Xc_steps    = 0

    self.interp_eta         = False
    self.interp_eta_from    = 0
    self.interp_eta_to      = 0
    self.interp_eta_steps   = 0

    #.............................
    # Bookkeeping of the process
    #.............................
    # The status of the preparation
    self.interp_prepare_OK  = False
    # Effective parameters used for interpolation
    self.interp_param_names = ['']
    # Number of multi-D dimensions of the interpolant
    self.interp_n_dim       = 0
    # List of slice objects (Python built-in) for np.mgrid
    self.interp_slices      = []
    # List of 1D ndarrays for all interpolation dimensions
    self.interp_1d_points   = []
    # Count the number of prepared points: 
    # n=Prod(n_k), n_k=len(points_k), for the k-th dimension
    self.interp_n_points    = 0
    # The status of calling numpy.mgrid, and building the meshgrid
    self.interp_meshgrid_OK = False
    # The resulting (interp_n_dim) tuple of meshgrids, all with
    # identical shape
    self.interp_meshgrid    = []
    # The shape of the resulting meshgrid
    self.interp_meshgrid_shape = tuple()


  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    if not hasattr(self, attr):
      logger.error('interpolation: set: Attribute "{0}" is unavailable.')
      sys.exit(1)

    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('interpolation: get: Attribute "{0}" is unavailable.')
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def do_interpolate(self):
    """
    This routine carries out the interpolation of frequencies over non-uniformly 
    gridded background layout of data points (attributes like M_ini, Z, etc).
    """
    _do_interpolate(self)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    #####    ###  ###  ###    ###    ###########  ######### 
    #    #   #    #    #    #    #     # #    #    #    #  #       #
    #     #  #     #   #    #    #     # #         #       #
    #    #   #    #    #     #  #     #   #        #       ####
    #####    #####     #     #  #     #####        #       #
    #        #  #      #      #      #     #       #       #       #
    #        #    #   ###    ###    ###   ###      #       #########


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _do_interpolate(self):
  """
  Refer to the documentation of the do_interpolate() method for detailed information.
  """
  _prepare(self)
  if not self.interp_prepare_OK:
    return False

  _build_meshgrid(self)

  _collect_inputs(self)

  _check_inputs(self)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs_around_anchor(self):
  """
  Query the database for fixed points around the anchor point.
  """
  M_ini_n = self.get('inputs_around_anchor_M_ini_n')
  fov_n   = self.get('inputs_around_anchor_fov_n')
  Z_n     = self.get('inputs_around_anchor_Z_n')
  logD_n  = self.get('inputs_around_anchor_logD_n')
  Xc_n    = self.get('inputs_around_anchor_Xc_n')
  eta_n   = self.get('inputs_around_anchor_eta_n')
  if np.sum(np.array([M_ini_n, fov_n, Z_n, logD_n, Xc_n, eta_n])) == 0:
    logger.error('_collect_inputs_around_anchor: Set inputs_around_anchor_..._n > 0')
    sys.exit(1)

  names   = self.get('anchor_param_names')
  vals    = self.get('anchor_param_values')
  freqs   = self.get('anchor_frequencies')
  ords    = self.get('anchor_radial_orders')
  types   = self.get('anchor_mode_types')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs_by_range(self):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs(self):
  """
  This routine collects the inputs from the database by quering it. There are two possibilities:
  - querying around the anchor model (check out self.anchor_param_values)
  - querying for a range of input parameters, e.g. M_ini: [2 - 5], etc

  """
  flags = np.array([self.inputs_around_anchor, self.inputs_by_range])
  if all(flags) or not all(flags):
    logger.error('_collect_inputs: Set only inputs_around_anchor or inputs_by_range to True')
    sys.exit(1)

  if self.inputs_around_anchor:
    _collect_inputs_around_anchor(self)

  if self.inputs_by_range:
    _collect_inputs_by_range(self)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _check_inputs(self):
  """
  For interpolation, you deinitely need an input, which must be compatible with the number of interpolation
  parameters, i.e. self.interp_n_dim. This routine provides two ndarrays, one for the x and one for y.
  The input x is a multi-dimensional ndarray of shape say (n, D), and the y is an ndarray of shape (n, ).
  """
  shape_x = self.input_features.shape 
  shape_y = self.input_frequencies.shape
  # the number of rows of the two must match
  if shape_x[0] != shape_y[0]:
    logger.error('_check_inputs: The input features and frequencies have different number of row')
    sys.exit(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _build_meshgrid(self):
  """
  Build the underlying meshgrid on-top-of-which the interpolation will be carried out. The size of
  this each of the meshgrids might become huge, specifically when requiring too many interpolation 
  points along each of the parameter dimensions. Thus, care must be practiced here to ensure all 
  needed intermediate matirxes fit properly into the memory of the computing hardware/node.
  """
  if not self.interp_prepare_OK:
    return False

  slices = self.get('interp_slices')

  try:
    msh  = np.mgrid[[the_slice for the_slice in slices]] 
    self.set('interp_meshgrid_OK', True)
    self.set('interp_meshgrid', msh)
    self.set('interp_meshgrid_shape', msh.shape)
  else:
    self.set('interp_meshgrid_OK', False)
    self.set('interp_meshgrid', [])
    self.set('interp_meshgrid_shape', (0, ))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _prepare(self):
  """
  Prepare the variables needed for the multi-D interpolation  
  """
  if self.interp_Xc:    names.append('Xc')
  if self.interp_eta:   names.append('eta')

  names  = []
  slices = []
  points = []
  n_pts  = 1
  if self.interp_M_ini:
    names.append('M_ini')
    slices.append(slice(self.interp_M_ini_from, self.interp_M_ini_to, complex(0, self.interp_M_ini_steps)))
    points.append(np.linspace(self.interp_M_ini_from, self.interp_M_ini_to, self.interp_M_ini_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_fov:
    names.append('fov')
    slices.append(slice(self.interp_fov_from, self.interp_fov_to, complex(0, self.interp_fov_steps)))
    points.append(np.linspace(self.interp_fov_from, self.interp_fov_to, self.interp_fov_steps))
    n_pts *= len(points[-1])

  if self.interp_Z:
    names.append('Z')
    slices.append(slice(self.interp_Z_from, self.interp_Z_to, complex(0, self.interp_Z_steps)))
    points.append(np.linspace(self.interp_Z_from, self. interp_Z_to, self.interp_Z_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_logD:
    names.append('logD')
    slices.append(slice(self.interp_logD_from, self.interp_logD_to, complex(0, self.interp_logD_steps)))
    points.append(np.linspace(self.interp_logD_from, self.interp_logD_to, self.interp_logD_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_Xc:
    names.append('Xc')
    slices.append(slice(self.interp_Xc_from, self.interp_Xc_to, complex(0, self.interp_Xc_steps)))
    points.append(np.linspace(self.interp_Xc_from, self.interp_Xc_to, self.interp_Xc_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_eta:
    names.append('eta')
    slices.append(slice(self.interp_eta_from, self.interp_eta_to, complex(0, self.interp_eta_steps)))
    points.append(np.linspace(self.interp_eta_from, self.interp_eta_to, self.interp_eta_steps, endpoint=True))
    n_pts *= len(points[-1])

  n = len(names)
  if n == 0:
    logger.error('_prepare: You must specify at least one parameter for interpolation')
    self.set('interp_prepare_OK', False)
  else:
    self.set('interp_prepare_OK', True)

  self.set('interp_param_names', names)
  self.set('interp_n_dim', n)
  self.set('interp_slices', slices)
  self.set('interp_1d_points', points)
  self.set('interp_n_points', n_pts)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
