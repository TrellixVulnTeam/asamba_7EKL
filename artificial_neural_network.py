
"""
This module provides various functionalities for carrying out Artificial Neural Network (ANN)
analysis (modelling) using the asteroseismic database, and a given set of observations. This
module builds heavily on the "sampler" module.
"""

import sys, os, glob
import logging
import numpy as np 

import star, sampler

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
    self.sampling = None # sampler.sampling()

    #.............................
    # Normal Equation
    #.............................
    # Analytic solution to the coefficients
    self.normal_equation_theta = 0
    # Solution of the features given analytic theta, and the observed y
    self.normal_equation_features = 0
    # The cost from the optimized theta and features
    self.normal_equation_cost = 0

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

    return getattr(self, attr)

  # Methods
  def solve_normal_equation(self):
    """
    Find the analytic solution for the unknown hypothesis coefficients \f$\theta\f$, which minimizes the
    cost function \f$ J(\theta) \f$ as defined below.
    
    \f[ J(\theta)= \frac{1}{2m} (\theta^T X-y)^T \cdot (\theta^T X-y) \f]
    
    For more information refer to: 
    <a href="http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression">Click to Open</a> 
    Consequently, the analytic solution to \f$\theta\f$ is:
    
    \f[ \theta_0 = (X^T \cdot X)^{-1} \cdot X^{-1} \cdot y. \f]

    Once \f$\theta_0\f$ is analytically derived, then the cost function has obtained its minimum value. If we assume
    this set of coefficients make the cost function approach zero \f$J(\theta_0)\approx 0\f$, intuitively 
    \f$ \theta_0^T\cdot X \approx y \f$. 

    One can immediately solve for the unknown feature vector \f$ X \f$, which reproduces the observations \f$ y_0\f$, 
    given the corresponding coefficients \f$ \theta_0 \f$. To that end, we multiply both sides of the last equation 
    by \f$ \theta \f$, followed by a multiplication with \f$ (\theta_0 \cdot \theta_0^T)^{-1} \f$ to yield \f$ X \f$:
    
    \f[ X_0 \approx (\theta_0 \cdot \theta_0^T)^{-1} \cdot (\theta \cdot y_0) \f]
    
    Notes:
    - The resulting coefficients are saved as the following attribute self.normal_equation_theta, and the resulting
      feature vector \f$ X_0 \f$ is stored as the attribute self.normal_equation_features.
    - The model frequencies \f$ y \f$ and the observed frequencies \f$ y_0 \f$ are converted to the per day 
      (\f$ d^{-1} \f$) unit for a fair comparison.

    @param self: instance of the neural_net class
    @type self: object
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

  x = sample.learning_x                # (m, n)
  x = _prepend_with_column_1(x)        # (m, n+1)
  y = sample.learning_y                # (m, K)

  a = np.dot(x.T, x)                   # (n+1, n+1)
  b = np.linalg.inv(a)                 # (n+1, n+1)
  c = np.dot(x.T, y)
  d = np.dot(b, c)

  theta = d[:]                         # (n+1, K)
  self.setter('normal_equation_theta', theta)

  # observed frequency from list of modes
  modes = sample.star.modes

  # Now, solve for the "best-value" features using theta from above
  freqs = np.array([ mode.freq for mode in modes ]).T # (K, 1)

  e = np.linalg.inv(np.dot(theta, theta.T)) # (n+1, n+1)
  f = np.dot(theta, freqs)             # (n+1, K) x (K, 1) == (n+1, 1)
  g = np.dot(e, f)                     # (n+1, n+1) x (n+1, 1)  == (n+1, 1)

  self.setter('normal_equation_features', g)

  J = np.sum(np.dot(g.T, theta) - freqs) / (2 * len(freqs))

  self.setter('normal_equation_cost', J)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _prepend_with_column_1(matrix):
  """
  Add a column of ones to the m-by-n matrix, so that the result is a m-by-n+1 matrix
  @param matrix: The general matrix of any arbitrary size with m rows and n columns
  @type matrix: np.ndarray
  @return: a matrix of m rows and n+1 columns where the 0-th column is all one.
  @rtype: np.ndarray
  """
  if not len(matrix.shape) == 2:
    print matrix.shape
    print len(matrix.shape)
    logger.error('_prepend_with_column_1: Only 2D arrays are currently supported')
    sys.exit(1)

  col_1 = np.ones(( matrix.shape[0], 1 )) 

  return np.concatenate([col_1, matrix], axis=1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
