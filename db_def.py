
"""
Define the basic class objects of the database (db), and some basic functionalities
"""

import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class X:
  """
  The class to interact with the grid, and execute SQL commands/querries
  """
  def __init__(self, dbname='grid'):
    """
    The constructor of the class

    @param dbname: the name of the running database server. By default, it is called "grid" too.
    @type dbname: string
    """
    self.dbname = dbname

  # Setters
  def set_dbname(self, dbname):
    self.dbname = dbname

  # Getters
  def get_dbname(self, dbname):
    return self.dbname

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
