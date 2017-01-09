
"""
Define the basic class objects of the database (db), and some basic functionalities
"""

import sys, os, glob
import logging
import numpy as np 
import psycopg2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
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

    connection  = psycopg2.connect('dbname={0}'.format(dbname))
    cursor      = conn.cursor()

    self.connection = connection
    self.cursor     = cursor

  # Setters
  def set_dbname(self, dbname):
    self.dbname = dbname

  # Getters
  def get_dbname(self):
    return self.dbname

  def get_connection(self):
    return self.connection

  def get_cursor(self):
    return self.cursor

  # Methods
  def commit(self):
    """
    Wrapper around the psycopg2.cursor.commit()
    """
    self.cursor.commit()

  def execute_one(self, cmnd):
    """
    Execute one SQL command on the cursor, passed by the "cmnd"
    """
    self.cursor.execute(cmnd)
    self.commit()

  def execute_many(self, list_cmnds):
    """
    Execute many (a list of) SQL commands on the cursor, passed by the "list_cmnds", one comand per
    item
    @param list_cmnds: list of SQL commands to execute
    @type list_cmnds: list
    """
    n_cmnds = len(list_cmnds)
    if n_cmnds == 0:
      logger.error('execute_many: the list of commands is empty')
      sys.exit(1)

    for i, cmnd in enumerate(list_cmnds):
      self.cursor.execute(cmnd)
    self.commit()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
