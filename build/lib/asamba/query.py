
"""
This module offers pre-composed queries to retrieve data from the database for a suite of 
recurring queries.
The outcome of most (if not all) of the routines in this module is basically the SQL query in 
string format. Therefore, external routines that call these functions just need to execute these
SQL querries.
"""
from __future__ import absolute_import
from __future__ import unicode_literals

import sys, os, glob
import logging
import time
import itertools
import numpy as np 

from asamba import db_def

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# G E N E R I C   R O U T I N E S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def without_constraint(dbname, table, returned_columns=[]):
  """
  Prepare a query to retrieve specific columns (from "returned_columns" argument). This is a generic
  routine that can be used on any table to retrive all rows from the table. A subset of columns can 
  be selected for retrieval through the "returned_columns" argument. 

  In fact, all this routine does is similar to the following generic SQL query:

    "SELECT * FROM table"

  where the "*" can be optionally replaced with a user-specified list of strings giving the desired
  column names of the table to retrieve.

  @param dbname: the name of the database to connect to
  @type dbname: str
  @param table: The name of the table where the query is going to be prepaired, and will be imosed on.
  @type table: str
  @param returned_columns: the list of column names that we require the values for in the output.
         E.g. one can set returned_columns = ['id', 'id_track', 'model_number'] to get the model id,
         the id of the evolutionary track that the model comes from, and the model_number of the model
         when this snapshot was stored by MESA. We check the requested column names exist as the 
         keys of the "models" table
  @type returned_columns: list of strings
  @return: The query string which is ready to be used
  @rtype: str
  """
  n_cols     = len(returned_columns)

  if n_cols == 0:
    str_cols = ' * '
  else:
    str_cols = ','.join(returned_columns)

  with db_def.grid_db(dbname=dbname) as the_db:
    # check if the table exists in the database
    if not the_db.has_table(table):
      logger.error('without_constraint: Database "{0}" does not have the \
                    table "{1}"'.format(dbname, table))
      sys.exit(1)

    # check that requested column names exist among the table attributes
    attrs   = the_db.get_table_columns(table) 
    if n_cols > 0:
      for attr in returned_columns:
        if attr.lower() not in attrs:
          logger.error('without_constraint: Attribute "{0}" is not a \
                        valid "models" key'.format(attr))
          sys.exit(1)

  str_cols  = ','.join(returned_columns)

  the_query = 'select {0} from {1}'.format(str_cols, table)

  return the_query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def with_constraints(dbname, table, returned_columns=[], constraints_keys=[], constraints_ranges=[]):
  """
  Prepare a query to retrieve specific columns (from "returned_columns" argument) subject to the 
  "WHERE" constraints specified by the "constraints_keys" argument, within the ranges specified in
  "constraints_ranges" list. Therefore, this is a general-purpose routine that can be used on any 
  database, with any table therein.

  E.g. 
  >>>my_query = with_constraints(dbname = 'grid', table = 'models',
                        returned_columns = ['id', 'Xc'], 
                        constraints_keys = ['log_Teff', 'log_g'], 
                        constraints_ranges = [[4,4.1], [3.4, 3.5]])
  >>>with db_def.grid_db(dbname) as my_db: 
  >>>  my_db.execute_one(my_query, None)
  >>>  result = my_db.fetch_all()
  >>>print len(results) > 0

  Note: The order of items in the input arguments "constraints_keys" and "constraints_ranges" must 
  match, because they are merged into a single SQL query in the same order as passed.

  @param dbname: the name of the database to connect to
  @type dbname: str
  @param table: The name of the table where the query is going to be prepaired, and will be imosed on.
  @type table: str
  @param returned_columns: the list of column names that we require the values for in the output.
         E.g. one can set returned_columns = ['id', 'id_track', 'model_number'] to get the model id,
         the id of the evolutionary track that the model comes from, and the model_number of the model
         when this snapshot was stored by MESA. We check the requested column names exist as the 
         keys of the "models" table
  @type returned_columns: list of strings
  @param constraints_keys: the list of keys for which we impose constraints during the querying.
         E.g. constraints_keys = ['age', 'mass_conv_core']. Indeed, we check internally if the requested
         key is valid.
  @type constrains_keys: list of strings   
  @param constraints_ranges: The list which provides the list/tuple for the lower and higher range
         for each of the keys in the previous argument "constraints_keys". Note that the ranges are
         applied in the same order as the keys are passed, so that first key goes with first range,
         second key goes with the second range, and so on.
         E.g. for the two keys mentioned above, one may pass: 
         constraints_ranges = [(1e6, 5e6), (0.3, 0.5)]
  @return: The query string which is ready to be used
  @rtype: str
  """
  if not returned_columns:
    logger.error('with_constraints: Provide non-empty list of return columns \
                  from the "{0}" attributes'.format(table))
    sys.exit(1)
  if not constraints_keys:
    logger.error('with_constraints: Provide non-empty list of constraint keys \
                  from the "{0}" attributes'.format(table))
    sys.exit(1)
  if not constraints_ranges:
    logger.error('with_constraints: Provide non-empty list of constraint ranges \
                  for each "{0}" attributes in the constraints_keys list'.format(table))
    sys.exit(1)

  n_cols    = len(returned_columns)
  n_keys    = len(constraints_keys)
  n_ranges  = len(constraints_ranges)

  if n_cols == 0:
    logger.error('with_constraints: the "returned_columns" must be non-empty list')
    sys.exit(1)

  if n_keys != n_ranges:
    logger.error('with_constraints: Mismatch between the number of keys and ranges')
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as the_db:
    # check if the table exists in the database
    if not the_db.has_table(table):
      logger.error('with_constraints: Database "{0}" does not have the \
                    table "{1}"'.format(dbname, table))
      sys.exit(1)

    # check that requested column names exist among the table attributes
    attrs   = the_db.get_table_columns(table) 
    for attr in returned_columns + constraints_keys:
      if attr.lower() not in attrs:
        logger.error('with_constraints: Attribute "{0}" is not a \
                      valid "models" key'.format(attr))
        sys.exit(1)

  str_cols  = ','.join(returned_columns)
  str_where = []
  for k, key in enumerate(constraints_keys):
    low     = constraints_ranges[k][0]
    high    = constraints_ranges[k][1]

    try:
      assert low <= high
    except AssertionError:
      logger.error('with_constraints: for key: "{0}", lower > higher! Revert the range.')
      sys.exit(1)

    str_where.append( '({0} between {1} and {2})'.format(key, low, high) )

  # put "AND" between each of the where statements
  str_where = ' and '.join(str_where)

  the_query = 'select {0} from {1} where {2}'.format(str_cols, table, str_where)

  return the_query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Q U E R Y I N G   T H E  T R A C K S   T A B L E 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_tracks_distinct_M_ini_logD():
  """
  In the database, the range of \\f$log D\f$ values were selected as a function on initial mass, so that 
  \f$\log D\f$ ranges between 0.0 and some \f$max(\log D)\f$ value in 5 discrete values; here, 
  \f$ max(\log D)\f$ is a linear function of initial mass, as:

  \f[ max(\log D_{\rm mix}) = \rm{offset} + \rm{slope}\, \times\, \log_{10}(M_{\rm ini}). 
  \f]
  where slope=\f$(6.5 - 2.5)/(\log_{10}(35) - \log_{10}(1.4))\f$, and the 
  offset=\f$6.5-{\rm slope}\times\log_{10}(35)\f$, with 1.4 and 35 \f$M_{\odot}\f$ bening the lowest
  and highest masses in the database, and \f$\log(D)=2.5\f$ and \f$\log(D)=6.5\f$ being the maximum logarithm
  of diffusive mixing for the lowest and highest masses in the database, respectively.

  This routine, prepares a simple query to retrieve all combinations of (M_ini, logD) for all tracks.
  """
  return 'select distinct on (M_ini, logD) M_ini, logD from tracks group by M_ini, logD'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Q U E R Y I N G   T H E   M O D E L S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_models_id_from_M_ini_fov_Z_logD_Xc(M_ini_range, fov_range, Z_range, logD_range, Xc_range):
  """
  Retrieve the basic model/track attributes from the "tracks" and "models" tables in the database,
  by providing the ranges in M_ini, fov, Z, logD and Xc, respectively.  

  @param _range: the lower and upper range for searching the database for each parameter (both extremes
         inclusive).
  @type _range: 2-element list of floats
  @return: a SQL search/query string
  @rtype: str
  """
  try:
    assert len(M_ini_range) == 2
    assert M_ini_range[0] <= M_ini_range[1]

    assert len(fov_range) == 2
    assert fov_range[0] <= fov_range[1]

    assert len(Z_range) == 2
    assert Z_range[0] <= Z_range[1]

    assert len(logD_range) == 2
    assert logD_range[0] <= logD_range[1]

    assert len(Xc_range) == 2
    assert Xc_range[0] <= Xc_range[1]

  except AssertionError:
    logger.error('get_models_id_from_M_ini_fov_Z_logD_Xc: Input range arguments must have length=2')
    sys.exit(1)

  where_M_ini = '(M_ini between {0} and {1})'.format(M_ini_range[0], M_ini_range[1])
  where_fov   = '(fov between {0} and {1})'.format(fov_range[0], fov_range[1])
  where_Z     = '(Z between {0} and {1})'.format(Z_range[0], Z_range[1])
  where_logD  = '(logD between {0} and {1})'.format(logD_range[0], logD_range[1])
  where_Xc    = '(Xc between {0} and {1})'.format(Xc_range[0], Xc_range[1])
  where_      = ' and '.join([where_M_ini, where_fov, where_Z, where_logD, where_Xc])

  the_query   = 'select models.id, M_ini, fov, Z, logD, Xc from models, tracks \
                 where (models.id_track=tracks.id) and {0}'.format(where_)
  return the_query
                 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_M_ini_fov_Z_logD_Xc_from_models_id(models_ids):
  """
  Retrieve the basic model/track attributes from the "tracks" and "models" tables in the database,
  using the list of models.id attribute (input argument).

  @param models_ids: an array of models ids
  @type models_ids: list/tuple/ndarray of type np.int32 
  @return: the query
  @rtype: str
  """
  n_ids     = len(models_ids)
  if n_ids  == 0:
    logger.error('get_M_ini_fov_Z_logD_Xc_from_models_id: The input list is empty')
    sys.exit(1)

  str_ids   = ','.join(['{0}'.format(mid) for mid in models_ids])

  the_query = 'select models.id, M_ini, fov, Z, logD, Xc from models, tracks \
       where (models.id_track=tracks.id) and (models.id in ({0}))'.format(str_ids)

  return the_query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_log_Teff_log_g_from_models_id(id_model):
  """
  Return the followng fixed query statement, ready to be executed:
  @param id_model: The id of the model
  @type id_model: integer
  @return: The following SQL query is returned: "select log_Teff, log_g from models where id=?", where 
        the "?" is replaced internally by the id of the desired model.
  @rtype: str
  """
  if not isinstance(id_model, int):
    logger.error('get_log_Teff_log_g_from_models_id: The input "id_model" must be integer')
    sys.exit(1)

  return 'select log_Teff, log_g from models where id={0}'.format(id_model)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Q U E R Y I N G   T H E   R O T A T I O N _ R A T E S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Q U E R Y I N G   T H E   M O D E S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def modes_from_fixed_id_model_id_rot(id_model, id_rot, id_type=[], freq_range=[]):
  """
  Return the query statement to fetch mode information (i.e. id_model, id_rot, n, id_type, freq)
  from the "modes" table given a fixed id_model and id_rot. This means, we basically deduce mode info
  from one GYRE output file which now sits in the database.

  @param id_model: the modes.id_model attribute from the "modes" table
  @type id_model: int
  @param id_rot: the modes.id_rot attribute from the "modes" table
  @type id_rot: int
  @param id_type: the list of mode identification types which are pre-defined in the grid.sql schema;
         e.g. radial modes (l, m) = (0, 0) correspond to id_type = [0], and radial prograde modes 
         correspond to [1]. Therefore, to inquire searching for radial AND dipole prograde modes, you
         must set id_types = [0, 1]
  @type id_types: list of integer
  @param freq_range: the lower and upper frequency range (Hz) to scan for, regardless of the id_type
  @type freq_range: list
  @return: the SQL query statement. the return rows from executing this statement will give the following
         output tuple per row: (id_model, id_rot, n, id_type, freq)
  @rtype: str
  """
  if not isinstance(id_model, int):
    logger.error('modes_from_fixed_id_model_id_rot: id_model must be an integer')
    sys.exit(1)
  if not isinstance(id_rot, int):
    logger.error('modes_from_fixed_id_model_id_rot: id_rot must be an integer')
    sys.exit(1)

  str_id_type   = ''
  if id_type:
    for val in id_type:
      if not isinstance(val, int):
        logger.error('modes_from_fixed_id_model_id_rot: id_type accepts only list of integers!')
        sys.exit(1)

    str_id_type = ','.join( ['{0}'.format(val) for val in id_type] )
    q_id_types  = ' and (id_type in ({0}))'.format(str_id_type)

  q_freq        = ''
  if freq_range:
    freq_lo     = freq_range[0]
    freq_hi     = freq_range[1]
    try:
      assert freq_lo <= freq_hi
    except AssertionError:
      logger.error('modes_from_fixed_id_model_id_rot: Assert that: freq_range[0] >= freq_range[1]')
    q_freq      = ' and (freq between {0} and {1})'.format(freq_lo, freq_hi)

  the_query = 'select id_model, id_rot, n, id_type, freq from modes where \
               id_model={0} and id_rot={1} {2} {3}'.format(id_model, id_rot, q_id_types, q_freq)

  return the_query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def modes_from_fixed_id_model_id_rot_prepared_statement(statement, id_type=[], freq_range=[]):
  """
  Return the "prepared" query statement to fetch mode information (i.e. id_model, id_rot, n, id_type, 
  freq) from the "modes" table given a fixed id_model and id_rot. This means, we basically deduce mode info
  from one GYRE output file which now sits in the database.

  @param statement: the name of the prepared statement, which is also used to execute the statement
  @type statement: str
  @param id_type: the list of mode identification types which are pre-defined in the grid.sql schema;
         e.g. radial modes (l, m) = (0, 0) correspond to id_type = [0], and radial prograde modes 
         correspond to [1]. Therefore, to inquire searching for radial AND dipole prograde modes, you
         must set id_types = [0, 1]
  @type id_types: list of integer
  @param freq_range: the lower and upper frequency range (Hz) to scan for, regardless of the id_type
  @type freq_range: list
  @return: the prepared statement and the execute statment, as a tuple of two elements
         The return rows from executing this statement will give the following
         output tuple per row: (id_model, id_rot, n, id_type, freq)
  @rtype: tuple
  """
  i_arg        = 2  # args $1 and $2 are reserved for id_model and id_rot, respectively
  dtypes       = ['int', 'int']
  q_id_types   = ''
  if id_type:
    for val in id_type:
      if not isinstance(val, int):
        logger.error('modes_from_fixed_id_model_id_rot_prepared_statement: \
                      id_type accepts only list of integers!')
        sys.exit(1)

    list_str_type = []
    for k, arg in enumerate(id_type):
      i_arg     += 1
      list_str_type.append( '${0}'.format(i_arg) )
      dtypes.append('int')

    str_id_type = ','.join( list_str_type )

    q_id_types  = ' and (id_type in ({0}))'.format(str_id_type)

  q_freq        = ''
  if freq_range:
    freq_lo     = freq_range[0]
    freq_hi     = freq_range[1]
    try:
      assert freq_lo <= freq_hi
    except AssertionError:
      logger.error('modes_from_fixed_id_model_id_rot_prepared_statement: Assert that: \
                    freq_range[0] >= freq_range[1]')
    q_freq      = ' and (freq between ${0} and ${1})'.format(i_arg+1, i_arg+2)
    dtypes.append('real')
    dtypes.append('real')
    i_arg       = i_arg + 2

  dtypes    = ','.join(dtypes)
  the_query =  'prepare {0} ({1}) as '.format(statement, dtypes)
  the_query += 'select id_model, id_rot, n, id_type, freq from modes where'
  the_query += ' id_model=$1 and id_rot=$2 {0} {1}'.format(q_id_types, q_freq)
 
  str_exec  = ','.join(['%s'] * (i_arg))
  the_exec  = 'execute {0} ({1})'.format(statement, str_exec)

  return (the_query, the_exec)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
