
"""
This module provides various functionalities for carrying out Artificial Neural Network (ANN)
analysis (modelling) using the asteroseismic database, and a given set of observations. This
module builds heavily on the "sampler" module.
"""

import sys, os, glob
import logging
import numpy as np 

import sampling

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class neural_net(object):
  """

  """
  def __init__(self):

    #.............................
    # Inheriting from sampler.sampling()
    #.............................
    self.sampling = sampler.sampling()

    #.............................
    # Normal Equation
    #.............................
    self.theta_normal_equation = 0

  # Setter
  def setter(self, attr, val):
    if not hasattr(self, attr):
      logger.error('neural_net: setter: Attribute "{0}" is unavailable.')
      sys.exit(1)

    setattr(self, attr, val)

  # Getter
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('neural_net: get: Attribute "{0}" is unavailable.')
      sys.exit(1)

    return getattr(self, val)

  # Methods
  def solve_normal_equation(self):
    """

    """
    _solve_normal_equation(self)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _solve_normal_equation(self):
  """
  Refer to the documentation of solve_normal_equation() for further details.
  """
  # Get THE instance of the sampling class already stored in neural_net object 
  sample = self.get('sampling') 
  # Check if the sampling is already done or not
  if not sample.learning_done:
    logger.error('_solve_normal_equation: The sampling is not done yet. Call sampler.sampling.build_sampling_sets() first')
    sys.exit(1)

  x = sample.learning_x
  y = sample.learning_y

  return ((x.T * x) * x.T) * y



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

