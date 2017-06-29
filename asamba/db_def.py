
"""
Define the basic class objects of the database (db), and some basic functionalities.
Some of the methods are wrappers around similar methods in psycopg2, with an additional
underscore "_" in the name; e.g. 
- grid_db.execute_one() wraps around psycopg2.execute()
- grid_db.execute_many() wraps around psycopg2.executemany()
- grid_db.fetch_one() wraps around psycopg2.fetchone()
- grid_db.fetch_many() wraps around psycopg2.fetchmany()
"""
from __future__ import print_function
from __future__ import unicode_literals

from builtins import object
import sys, os, glob
import logging
import numpy as np 
import subprocess
import psycopg2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger  = logging.getLogger(__name__)
is_py3x = sys.version_info[0] >= 3

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class grid_db(object):
  """
  The class to interact with the grid, and execute SQL commands/querries
  """
  # ...................................
  def __init__(self, location):
    """
    The constructor of the class. Example of use:

    >>>my_job = db_def.grid_db('ivs')
    >>>cursor = my_job.get_cursor()

    @param location: the location where the database server is setup and is running. 
                 Note: this "location" is different from "dbaname" used by psycopg2 to connect to 
                 the database. The dbname is set internally, based on the provided location for 
                 the database. Currently, the available names are the following:
                 - laptop: which connects to a copy of the database on my own laptop
                 - ivs: which connects to a copy of the databse at the Institute of Astronomy, 
                        KULeuven
                 - https: which connects to a copy of the database openly hosted online
    @type location: string
    """
    if not is_py3x: loc = loc.encode('ascii') # to avoid unicode conflicts
    self.location = location
    self.dbname   = ''

    conn_string = self._prepare_connection()
    if conn_string == '':
      logger.error('__init__: Failed to prepare the connection. Check the connection name again')
      sys.exit(1)

    if exists(self.dbname) is False:
      logger.error('__init__: Database "{0}" does not exist'.format(self.dbname))
      sys.exit(1)

    connection  = psycopg2.connect(conn_string)
    cursor      = connection.cursor()

    self.connection = connection
    self.cursor     = cursor

  def __enter__(self):
    return self 

  def __exit__(self, type, value, traceback):
    self.get_cursor().close()
    self.get_connection().close()

  # Setters
  # ...................................
  def set_dbname(self):
    loc      = self.location
    dbname   = assign_dbname(loc)
    self.dbname = dbname

  # Getters
  # ...................................
  def get_dbname(self):
    return self.dbname

  # ...................................
  def get_connection(self):
    return self.connection

  # ...................................
  def get_cursor(self):
    return self.cursor

  # Methods
  # ...................................
  def _prepare_connection(self):
    loc         = self.location
    if loc      == 'laptop':  # must go obsolete after other options come online
      string    = 'dbname=grid'
    elif loc    == 'asamba_dev':
      string    = 'dbname=asamba_dev'
    elif loc    == 'ivs':
      string    = 'dbname=asamba user=asamba host=fs1'
    elif loc    == 'https': # Not ready yet
      string    = ''
    else:
      logger.error('_prepare_connection: location="{0}" is unavailable'.format(loc ))
      string    = ''

    self.set_dbname()

    return string

  # ...................................
  def get_table_columns(self, table):
    cmnd = 'select column_name from information_schema.columns where table_name = %s'
    self.execute_one(cmnd, (table, ))
    return [tup[0] for tup in self.fetch_all()]

  # ...................................
  def has_table(self, table):
    cmnd = 'select exists(select * from information_schema.tables where table_name=%s)'
    self.execute_one(cmnd, (table,))
    return self.fetch_one()[0]

  # ...................................
  def has_function(self, function_name):
    cmnd = 'select exists(select * from pg_proc where proname=%s)'
    self.execute_one(cmnd, (function_name, ))
    return self.fetch_one()[0]

  # ...................................
  def has_prepared_statement(self, statement):
    cmnd = 'select exists(select * from pg_prepared_statements where name=%s)'
    self.execute_one(cmnd, (statement, ))
    return self.fetch_one()[0]

  # ...................................
  def commit(self):
    """
    Wrapper around the psycopg2.cursor.commit()
    """
    self.connection.commit()

  # ...................................
  def execute_one(self, cmnd, value, commit=True):
    """
    **Execute AND commit** one SQL command on the cursor, passed by the "cmnd"
    """
    result = self.cursor.execute(cmnd, value)
    if result is not None:
      logger.error('execute_one failed')
      sys.exit(1)
    if commit: self.commit()

  # ...................................
  def execute_many(self, cmnd, values, commit=True):
    """
    **Execute AND commit** many (a list of) SQL commands on the cursor.
    The command is passed by the "cmnd", and the corresponding values are passed
    by the "values" tuple. This function is very useful for inserting data into
    the database.

    @param cmnd: A general command to execute many. E.g., cmnd can look like the 
           the following: cmnd = 'insert into table_name (var1, var2) values (?, ?)'
           This ensures that the execute/commit process is protected against possible
           Injection Attacks. 
    @type cmnd: string
    @param values: A list of tuples to execute the command. For every execute/commit
          transaction, one tuple must be in this list. The order of the quantities 
          in each tuple must match the order of the parameters in the command.
    @type values: list of tuples
    """
    if not isinstance(values, list):
      logger.error('execute_many: the 2nd argument (i.e. values) must be a list of tuples.')
      sys.exit(1)

    n_vals  = len(values)
    if n_vals == 0:
      logger.error('execute_many: the list of values is empty')
      sys.exit(1)

    cursor  = self.get_cursor()
    try:
      result  = cursor.executemany(cmnd, values)
    except psycopg2.Error as err:
      print('error has occured:', err)
    if result is not None:
      logger.error('execute_many failed')
      sys.exit(1)
    if commit == True: self.commit()

  # ...................................
  def fetch_one(self):
    """
    A wrapper around the psycopg2.fetchone() method
    """
    return self.get_cursor().fetchone() 

  # ...................................
  def fetch_many(self):
    """
    A wrapper around psycopg2.fetchmany()
    """
    return self.get_cursor().fetchmany()

  # ...................................
  def fetch_all(self):
    """
    A wrapper around psycopg2.fetchall()
    """
    return self.get_cursor().fetchall()

  # ...................................
  def get_mode_types(self):
    """
    Retrieve the contents of the "mode_types" table
    """      
    cmnd = 'select * from mode_types'
    self.execute_one(cmnd, None)
    out  = self.fetch_all()

    return out

  # ...................................
  def get_rotation_rates(self):
    """
    Retrieve the contents of the "rotation_rates" table
    """
    cmnd = 'select * from rotation_rates'
    self.execute_one(cmnd, None)
    out = self.fetch_all()

    return out
  # ...................................
  # ...................................

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def assign_dbname(location):
  """
  Assign a unique (hard-coded) name to the database, based on where it is hosted
  This function can be used by external calls, or by any instance of the grid_db()
  class.
  @param location: The location where the database is hosted, e.g. 'laptop'
  @type location: str
  @return: the assigned name of the database
  @rtype: str
  """
  loc = location  
  if loc   == 'laptop': 
    dbname = 'asamba_dev'
  elif loc == 'ivs':
    dbname = 'asamba'
  elif loc == 'https':
    dbname = ''
  else:
    logger.error('assign_dbname: location="{0}" is undefined'.format(loc))
    sys.exit(1)

  return dbname

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def exists(dbname):
  """
  Check if the database already exists.
  Returns True if the database exists, and False otherwise.
  """
  cstr   = '-d asamba -U asamba -h fs1' if dbname == 'asamba' else ''
  cmnd   = 'psql -lqt {0} | cut -d \| -f 1 | grep -w {1}'.format(cstr, dbname)
  exe    = subprocess.Popen(cmnd, shell=True, universal_newlines=True, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  err    = exe.returncode

  if err is not None:
    logger.info('grid_db.exists(): Command failed: "{0}"'.format(cmnd))
    return False
  else:
    stdout = exe.stdout.read().rstrip('\r\n').strip()
    stderr = exe.stderr.read().rstrip('\r\n').strip()
    try:
      assert stdout == dbname
      return True
    except AssertionError:
      return False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

