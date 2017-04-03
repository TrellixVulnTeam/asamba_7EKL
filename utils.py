
"""
This module provides some generic utilities to facilitate working with different datatypes
and input/outputs conveniently.
"""

import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def list_to_recarray(list_input, dtype):
  """
  Convert a list of tuples to a numpy recordarray. Each tuple is one retrieved row of data from calling
  the SQL queries, and fetching them through e.g. db.fetch_all() method.

  @param list_input: The inputs to be converted to numpy record array. They are 
  """

  return np.core.records.fromarrays(np.array(list_input).T, dtype=dtype)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def recarray_to_ndarray(rec):
  """
  Convert a numpy record array to a matrix ndarray

  @param rec: numpy record array
  @return: ndarray
  """
  
  return rec.view(np.float32).reshape(rec.shape + (-1, ))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
