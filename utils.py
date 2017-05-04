
"""
This module provides some generic utilities to facilitate working with different datatypes
and input/outputs conveniently.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div

import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger  = logging.getLogger(__name__)
is_py3x = sys.version_info[0] >= 3

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def list_to_recarray(list_input, dtype):
  """
  Convert a list of tuples to a numpy recordarray. Each tuple is one retrieved row of data from calling
  the SQL queries, and fetching them through e.g. db.fetch_all() method.

  @param list_input: The inputs to be converted to numpy record array. 
  @type list_input: list
  @return: recarray with the number of rows equal to the number of tuples in the input list, and the 
        number of columns equal to the number of items in each tuple. The dtype for each column is 
        passed as an input argument by the user.
  @rtype: np.recarray
  """
  if not is_py3x:
    names = [tup[0] for tup in dtype]
    types = [tup[1] for tup in dtype]
    dtype = [(name.encode('ascii'), dtp) for name, dtp in zip(names, types)]

  try:
    a = np.core.records.fromarrays(np.array(list_input).T, dtype=dtype)
    # a = np.core.records.fromarrays(list_to_ndarray(list_input), dtype=dtype)
  except Exception as xpt:
    print('error occured:', xpt)
  else:
    return a

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def list_to_ndarray(list_input):
  """
  Convert a list of tuples to a numpy ndarray. 

  @param list_input: The inputs to be converted to numpy recorda array
  @type list_input: list
  @return: recarray with the number of rows equal to the number of tuples in the input list, and the 
        number of columns equal to the number of items in each tuple.
  @rtype: np.recarray
  """
  return np.stack(list_input, axis=0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ndarray_to_recarray(arr, dtype):
  """
  Convert a numpy ndarray to a numpy record array

  @param arr: the input array
  @param dtype: the list of np.dtype for all columns/attributes
  @return: a corresponding record array
  """
  if not is_py3x:
    names = [tup[0] for tup in dtype]
    types = [tup[1] for tup in dtype]
    dtype = [(name.encode('ascii'), dtp) for name, dtp in zip(names, types)]

  return np.core.records.fromarrays(arr.T, dtype=dtype)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def recarray_to_ndarray(rec):
  """
  Convert a numpy record array to a matrix ndarray

  @param rec: numpy record array
  @return: ndarray
  """
  
  return rec.view(np.float32).reshape(rec.shape + (-1, ))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepend_with_column_1(matrix):
  """
  Add a column of ones to the m-by-n matrix, so that the result is a m-by-n+1 matrix
  @param matrix: The general matrix of any arbitrary size with m rows and n columns
  @type matrix: np.ndarray
  @return: a matrix of m rows and n+1 columns where the 0-th column is all one.
  @rtype: np.ndarray
  """
  if not len(matrix.shape) == 2:
    print(matrix.shape)
    print(len(matrix.shape))
    logger.error('prepend_with_column_1: Only 2D arrays are currently supported')
    sys.exit(1)

  col_1 = np.ones(( matrix.shape[0], 1 )) 

  return np.concatenate([col_1, matrix], axis=1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gaussian(x, mu, sigma):
  """
  Return the Normal (Gaussian) probability distribution function g(x) for x with mean mu and standard
  deviation sigma, following the definition 

  \f[
      N(x,\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right].
  \f]

  @param x: array or value of the input 
  @type x: ndarray or float
  @param mu: the mean of the population
  @type mu: float
  @param sigma: the standard deviation of the population around the mean
  @type sigma: npdarray or float
  @return: The probability of x being between the interval x and x+epsilon, where epsilon goes to zero
  @rtype: ndarray or float
  """
  if isinstance(x, np.ndarray) and isinstance(sigma, np.ndarray):
    try:
      assert len(x) == len(sigma)
    except AssertionError:
      logger.error('gaussian: the size of input arrays "x" and "sigma" must be identical')
      sys.exit(1)

  scale = old_div(1.,(np.sqrt(2*np.pi)*sigma))
  arg   = -0.5 * (old_div((x - mu)**2,sigma**2))
  
  return  scale * np.exp(arg)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
