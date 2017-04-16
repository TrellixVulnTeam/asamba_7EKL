
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
  The base class for internal interpolation means
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
    # Specifications for interpolation
    #.............................
    # Parameter ranges, and stepsizes
    self.interp_M_ini      = False
    self.interp_M_ini_from = 0
    self.interp_M_ini_to   = 0
    self.interp_M_ini_step = 0

    self.interp_fov        = False
    self.interp_fov_from   = 0
    self.interp_fov_to     = 0
    self.interp_fov_step   = 0

    self.interp_Z          = False
    self.interp_Z_from     = 0
    self.interp_Z_to       = 0
    self.interp_Z_step     = 0

    self.interp_logD       = False
    self.interp_logD_from  = 0
    self.interp_logD_to    = 0
    self.interp_logD_step  = 0

    self.interp_Xc         = False
    self.interp_Xc_from    = 0
    self.interp_Xc_to      = 0
    self.interp_Xc_step    = 0

    self.interp_eta        = False
    self.interp_eta_from   = 0
    self.interp_eta_to     = 0
    self.interp_eta_step   = 0


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




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    #####    ###  ###  ###    ###    ###########  ######### 
    #    #   #    #    #    #    #     # #    #    #    #  #       #
    #     #  #     #   #    #    #     # #         #       #
    #    #   #    #    #     #  #     #   #        #       ####
    #####    #####     #     #  #     #####        #       #
    #        #  #      #      #      #     #       #       #       #
    #        #    #   ###    ###    ###   ###      #       #########


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
