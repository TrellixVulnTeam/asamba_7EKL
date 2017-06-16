
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
def feature_name_in_layman(name, short=True):
  """
  Return the layman-term representation of the feature names. This is majorly used for setting the axis 
  names in plotting routines.
  
  @param name: one of the names from self.feature_names
  @type name: str
  @return: raw string in LaTeX format, e.g. r'$Z$' if name=='Z'
  @rtype: str
  """
  if name == 'M_ini':
    latex = r'Mass' if short else r'Initial Mass'
  elif name == 'fov':
    latex = r'Overshoot'if short else r'Core Overshooting'
  elif name == 'Z':
    latex = 'Metallicity'
  elif name == 'logD':
    latex = r'Extra Mixing' if short else r'$\log_{10}$(Extra Mixing)'
  elif name == 'Xc':
    latex = r'Core H' if short else r'Core Hydrogen [$\%$]'
  elif name == 'eta':
    latex = r'Spin Rate' if short else r'Rotation Rate [$\%$]'
  else:
    logger.warning('feature_name_in_layman: name:"{0}" is invalid'.format(name))
    latex = None 

  return latex

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def feature_name_in_latex(name):
  """
  Return the LaTeX representation of the feature names. This is majorly used for setting the axis 
  names in plotting routines.
  
  @param name: one of the names from self.feature_names
  @type name: str
  @return: raw string in LaTeX format, e.g. r'$Z$' if name=='Z'
  @rtype: str
  """
  if name == 'M_ini':
    latex = r'$M_{\rm ini}$ [M$_\odot$]'
  elif name == 'fov':
    latex = r'$f_{\rm ov}$'
  elif name == 'Z':
    latex = r'$Z$'
  elif name == 'logD':
    # latex = r'$\log D_{\rm mix}$ (cm$^2$ sec$^{-1}$)'
    # latex = r'$\log D_{\rm mix}$'
    latex = r'Extra Mixing'
  elif name == 'Xc':
    latex = r'$X_{\rm c}$'
  elif name == 'eta':
    latex = r'$\eta_{\rm rot}$ [$\%$]'
  else:
    logger.warning('feature_name_in_latex: name:"{0}" is invalid'.format(name))
    latex = None 

  return latex

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def intert_dic_key_value(dic):
  """
  Invert the key/value order of the dictionary, so that {'kye':value} becomes {'(value, )':key}.
  If the value in the original dictionary is a single integer/float, then the associated key in the
  inverted dictionary will be a single-element tuple made from that, i.e. (value, ).
  @param dic: The dictinary whose key/value pairs to be inverted
  @type dic: dict 
  @return: The inverted dictionary with the key/value pairs swapped. Note that for single value 
        dictionaries, the new keys will be basically the single-element tuples of the former values
  @rtype: dict
  """
  keys     = list(dic.keys())
  vals     = list(dic.values())
  int_vals = isinstance(vals[0], int)
  flt_vals = isinstance(vals[0], float)
  inv      = dict()
  if int_vals or flt_vals:
    _keys  = [(key, ) for key in vals]
    _vals  = [val for val in keys]
  for key, val in zip(_keys, _vals): inv[key] = val 

  return inv

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def has_nan(arr):
  """
  This routine tries to find a NaN in any of the elements of the passed array. If no NaN is found, 
  then it returns True, but if at least one NaN value is found, then it returns False
  """

  return np.sum(np.isnan(arr) * 1) > 0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def substitute_inf(arr):
  """
  The sys.float_info.max = 1.7976931348623157e+308 (hereafter max_float) is the maximum double precision 
  floating point number that Python handles. If the absolute value of any number exceeds this limit, 
  then it is assigned as inf.
  This routine substitutes the +inf elements with the max_float, the -inf with -max_float.

  @param arr: any n-dimensional array of numbers
  @type arr: numpy.ndarray
  @return: the same array with the +/- inf values and NaN values substituted as explained above.
  @rtype: numpy.ndarray 
  """
  inf       = float('inf')
  max_float = sys.float_info.max / 2
  
  i_inf_pos = np.where(arr == inf)[0]
  arr[i_inf_pos] = max_float

  i_inf_neg = np.where(arr == -inf)[0]
  arr[i_inf_neg] = -max_float

  return arr

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
def prepend_with_0(vector):
  """
  Add a float of value zero, 0.0, to the input vector of length m, so that the return vector will
  eventually have m+1 elements, witht the zeroth element being now 0.0
  @param vector: The generic ndarray vector of any arbitrary size 
  @type vector: np.ndarray
  @return: a ndarray vector of m+1 elements with the zeroth element set to zero
  @rtype: np.ndarray
  """
  return np.concatenate([np.zeros(1), vector])
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepend_with_1(vector):
  """
  Add a float of value unity, 1.0, to the input vector of length m, so that the return vector will
  eventually have m+1 elements, witht the zeroth element being now 1.0
  @param vector: The generic ndarray vector of any arbitrary size 
  @type vector: np.ndarray
  @return: a ndarray vector of m+1 elements with the zeroth element set to unity
  @rtype: np.ndarray
  """
  return np.concatenate([np.ones(1), vector])

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
