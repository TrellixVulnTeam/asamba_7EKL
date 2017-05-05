from __future__ import unicode_literals

from builtins import zip
from builtins import range
import sys, os, glob
import logging
import numpy as np 
import psycopg2

from asamba import db_def, query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E _ T Y P E S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dic_look_up_mode_types_id(dbname_or_dbobj):
  """
  Create a look up dictionary for the "mode_types" table, to speed up fetching the mode types  ids 
  through dictionary look up.
  E.g. to retrieve the type id for the radial modes (l, m) = (0, 0), we do the following:
  
  >>>from asamba import db_lib
  >>>dic_mode_type = db_lib.get_dic_look_up_mode_types_id('grid')
  >>>print dic_mode_type[(0,0)]
  >>>0

  @param dbname_or_dbobj: The first argument of this function can have two possible types. The reason 
        is that Python does not really support function overloading. Instead, it is careless about the
        type of the input argument, which we benefit from here. The reason behind this choice of 
        development is to avoid creating/closing a connection/cursor to the database everytime one 
        freaking model ID needs be fetched. This avoids connection overheads when thousands to 
        millions of track IDs need be retrieved.
        The two possible inputs are:
        - dbname: string which specifies the name of the dataase. This is used to instantiate the 
                  db_def.grid_db(dbname) object. 
        - dbobj:  An instance of the db_def.grid_db class. 
  @type dbname_or_dbobj: string or db_def.grid_db object
  @return: a look up dictionary that contains the mode_type tuples as keys, and the mode_type "id"s
        as values. 
  @rtype: dict
  """
  # fetch the "mode_types" table
  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      mode_types = the_db.get_mode_types()
  #
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    mode_types   = dbname_or_dbobj.get_mode_types()
  #
  else:
    logger.error('get_dic_look_up_mode_types_id: Input type not string or db_def.grid_db!')
    sys.exit(1)

  if not isinstance(mode_types, list):
    logger.error('get_dic_look_up_mode_types_id: failed')
    sys.exit(1)

  n   = len(mode_types)
  if n == 0:
    logger.error('get_dic_look_up_mode_types_id: the result list is empty')
    sys.exit(1)

  mode_types_id  = [tup[0] for tup in mode_types]
  mode_types_l_m = [(tup[1], tup[2]) for tup in mode_types]
  dic_mode_types = dict()
  for key, val in zip(mode_types_l_m, mode_types_id):
    dic_mode_types[key] = val

  return dic_mode_types

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   R O T A T I O N _ R A T E S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dic_look_up_rotation_rates_id(dbname_or_dbobj):
  """
  Create a look up dictionary for the "rotation_rates" table, to speed up fetching the mode types ids 
  through dictionary look up.
  E.g. to retrieve the id for the rotation rate eta=30.00 percent, we do the following:
  
  >>>from asamba import db_lib
  >>>dic_rot_rates = db_lib.get_dic_look_up_rotation_rates_id('grid')
  >>>eta = 25
  >>>tup_rot = (eta, )
  >>>print dic_rot_rates[tup_rot]
  >>>7

  @param dbname_or_dbobj: The first argument of this function can have two possible types. The reason 
        is that Python does not really support function overloading. Instead, it is careless about the
        type of the input argument, which we benefit from here. The reason behind this choice of 
        development is to avoid creating/closing a connection/cursor to the database everytime one 
        freaking model ID needs be fetched. This avoids connection overheads when thousands to 
        millions of track IDs need be retrieved.
        The two possible inputs are:
        - dbname: string which specifies the name of the dataase. This is used to instantiate the 
                  db_def.grid_db(dbname) object. 
        - dbobj:  An instance of the db_def.grid_db class. 
  @type dbname_or_dbobj: string or db_def.grid_db object
  @return: a look up dictionary that contains the rotation_rate tuples as keys, and the rotation_rate "id"s
        as values. 
  @rtype: dict
  """
  # fetch the "rotation_rates" table
  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      rot_rates = the_db.get_rotation_rates()
  #
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    rot_rates   = dbname_or_dbobj.get_rotation_rates()
  #
  else:
    logger.error('get_dic_look_up_rotation_rates_id: Input type not string or db_def.grid_db!')
    sys.exit(1)

  if not isinstance(rot_rates, list):
    logger.error('get_dic_look_up_rotation_rates_id: failed')
    sys.exit(1)

  n   = len(rot_rates)
  if n == 0:
    logger.error('get_dic_look_up_rotation_rates_id: the result list is empty')
    sys.exit(1)

  eta_ids   = [tup[0] for tup in rot_rates]
  eta_vals  = [(tup[1], ) for tup in rot_rates]
  dic_rot_rates = dict()
  for key, val in zip(eta_vals, eta_ids):
    dic_rot_rates[key] = val

  return dic_rot_rates

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def find_missing_models(dbname, eta):
  """
  For a given rotation rate, eta, find the models.id that do not have a corresponding row in the 
  modes.id_model list. These are models which the GYRE computation has either failed, or for some
  reason, they are not still inserted into the "modes" table yet.
  
  @param dbname: The name of the database, e.g. "grid"
  @type dbname: string
  @return: list of models.id where the GYRE computaiton shall be either repeated, or the data must
        be inserted into the modes table.
  @rtype: list of int
  """
  tup_rot  = (eta, )
  dic_rot  = get_dic_look_up_rotation_rates_id(dbname)
  try: 
    id_rot = dic_rot[tup_rot]
    tup_id_rot = (id_rot, )
    logger.info('find_missing_models: corresponding id_rot is "{0}"'.format(id_rot))
  except:
    logger.error('find_missing_models: eta={0} is invalid, and not supported yet!'.format(eta))
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as the_db:
    cmnd = 'select id from models'
    the_db.execute_one(cmnd, None)
    id_from_models = [tup[0] for tup in the_db.fetch_all()]
    n    = len(id_from_models)

    cmnd = 'select distinct on (id_model) id_model from modes where id_rot=%s group by id_model'
    the_db.execute_one(cmnd, tup_id_rot)
    id_from_modes  = [tup[0] for tup in the_db.fetch_all()]
    m    = len(id_from_modes)

  # Sanity checks
  if n == 0:
    logger.error('find_missing_models: The "models" table is empty!')
    sys.exit(1)
  if m == 0:
    logger.error('find_missing_models: The "modes" table is empty!')
    sys.exit(1)
  if m > n:
    logger.error('find_missing_models: The funny case that we have more input to modes than the input models!')
    sys.exit(1)

  # The safe mode
  if n == m:
    logger.info('find_missing_models: All Input models have a corresponding mode list for eta="{0}"'.format(eta))
    return None
  else:
    missing = set(id_from_models).symmetric_difference(set(id_from_modes))
    n_missing = len(missing)
    logger.info('find_missing_models: Returning missing {0} models.id values'.format(n_missing))
    return sorted(list(missing))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_missing_gyre_task(dbname, eta, h5_prefix='ad-sum'):
  """
  For a given rotation rate, eta, find the names of the GYRE input files, and the missing GYRE output
  files for which the GYRE output file is absent in the database (so, may need to do the GYRE 
  computaitons for these).

  @param dbname: The name of the database to connect to
  @type dbname: string
  @param eta: the rotation rate in percentage
  @type eta: float
  @param h5_prefix: The prefix which is added at the begining of the HDF5 GYRE output files to 
        distinguish the adiabatic/non-adiabatic summary/mode files. Set to '' to ignore it
  @type h5_prefix: str
  @return: two lists of strings are returned. 
        1. The first list is the core name of the GYRE input files, e.g.
           M35.000-ov0.010-Z0.010-logD00.00-MS-Xc0.6092-00983.gyre
        2. The second list is the core name of the GYRE ouput files, e.g.
           ad-sum-M35.000-ov0.010-Z0.010-logD00.00-MS-Xc0.6092-00983-eta50.00.h5
  @rtype: tuple of two lists
  """
  missing = find_missing_models(dbname=dbname, eta=eta)
  # tups    = [(val, ) for val in missing]
  cmnd    = 'select M_ini, fov, Z, logD, model_number, Xc from tracks inner join models on tracks.id = models.id_track where models.id=%s'
  result  = []
  with db_def.grid_db(dbname=dbname) as the_db:
    for i, id_model in enumerate(missing):
      tup = (id_model, )
      the_db.execute_one(cmnd, tup)
      result.append(the_db.fetch_one())

  list_gyre_in  = []
  list_gyre_out = []
  for i, tup in enumerate(result):
    M_ini, fov, Z, logD, model_number, Xc = tup
    core = 'M{0:6.3f}-ov{1:4.3f}-Z{2:4.3f}-logD{3:05.2f}-MS-Xc{4:5.4f}-{5:05d}'.format(
            M_ini, fov, Z, logD, Xc, model_number)
    gyre_in  = 'M{0:06.3f}/gyre_in/{1}.gyre'.format(M_ini, core)
    gyre_out = 'M{0:06.3f}/gyre_out/eta{1:05.2f}/{2}-{3}-eta{4:05.2f}.h5'.format(
                M_ini, eta, h5_prefix, core, eta) 
    
    list_gyre_in.append(gyre_in)
    list_gyre_out.append(gyre_out)

  return list_gyre_in, list_gyre_out

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E L S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dic_look_up_models_id(dbname_or_dbobj):
  """
  Retrieve the id, id_track and model_number from the entire "models" table, and construct a look up
  dictionary with the keys as the (id_track, model_number) tuple, and the values as the id. 

  @param dbname_or_dbobj: The first argument of this function can have two possible types. The reason 
        is that Python does not really support function overloading. Instead, it is careless about the
        type of the input argument, which we benefit from here. The reason behind this choice of 
        development is to avoid creating/closing a connection/cursor to the database everytime one 
        freaking model ID needs be fetched. This avoids connection overheads when thousands to 
        millions of track IDs need be retrieved.
        The two possible inputs are:
        - dbname: string which specifies the name of the dataase. This is used to instantiate the 
                  db_def.grid_db(dbname) object. 
        - dbobj:  An instance of the db_def.grid_db class. 
  @type dbname_or_dbobj: string or db_def.grid_db object
  @return: look up dictinary with keys as a tuple with the two elements "(id_track, model_number)" and 
        the value as the "models.id"
  @rtype: dict
  """
  cmnd = 'select id, id_track, model_number from models'

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd, None)
      result = the_db.fetch_all()
  #
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd, None)
    result   = dbname_or_dbobj.fetch_all()
  #
  else:
    logger.error('get_dic_look_up_models_id: Input type not string or db_def.grid_db! It is: {0}'.format(type(dbname)))
    sys.exit(1)

  if not isinstance(result, list):
    logger.error('get_dic_look_up_models_id: failed')
    sys.exit(1)

  n   = len(result)
  if n == 0:
    logger.error('get_dic_look_up_models_id: the result list is empty')
    sys.exit(1)

  list_id  = np.array([ result[k][0] for k in range(n) ])
  list_tup = [ (result[k][1], result[k][2]) for k in range(n) ]
  dic = dict()
  for key, val in zip(list_tup, list_id): dic[key] = val

  logger.info('get_dic_look_up_models_id: Successfully returning "{0}" records'.format(n))

  return dic

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_models_id_by_id_tracks_and_model_number(dbname_or_dbobj, id_track, model_number):
  """
  @param dbname_or_dbobj: The first argument of this function can have two possible types. The reason 
        is that Python does not really support function overloading. Instead, it is careless about the
        type of the input argument, which we benefit from here. The reason behind this choice of 
        development is to avoid creating/closing a connection/cursor to the database everytime one 
        freaking model ID needs be fetched. This avoids connection overheads when thousands to 
        millions of track IDs need be retrieved.
        The two possible inputs are:
        - dbname: string which specifies the name of the dataase. This is used to instantiate the 
                  db_def.grid_db(dbname) object. 
        - dbobj:  An instance of the db_def.grid_db class. 
  @type dbname_or_dbobj: string or db_def.grid_db object
  @param id_track: the track id of the model. This must be already provided by calling e.g. the 
         db_lib.get_track_id() routine. For that, we must provide the four track attributes (knowing 
         them by heart! or from the model filename).
  @type id_track: int
  @param model_number: The model_number is present in the GYRE input/output filename.
  @type model_number: int
  @return: the id of the models from the "models" table. If the operation fails, or the model id is 
         not found (for any awkward reason), then an exception is raised.
  @rtype: int
  """
  cmnd_min = 'select min(id) from models'
  cmnd_max = 'select max(id) from models'
  cmnd_id  = 'select id from models where id_track=%s and model_number=%s'
  tup      = (id_track, model_number)

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd_min, None)
      min_id = the_db.fetch_one()[0]
      the_db.execute_one(cmnd_max, None)
      max_id = the_db.fetch_one()[0]

      the_db.execute_one(cmnd_id, tup)
      result = the_db.fetch_one()
  #
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd_min, None)
    min_id   = dbname_or_dbobj.fetch_one()[0]
    dbname_or_dbobj.execute_one(cmnd_max, None)
    max_id   = dbname_or_dbobj.fetch_one()[0]

    dbname_or_dbobj.execute_one(cmnd_id, tup)
    result   = dbname_or_dbobj.fetch_one()
  #
  else:
    logger.error('get_track_id: Input type not string or db_def.grid_db! It is: {0}'.format(type(dbname)))
    sys.exit(1)

  if isinstance(result, type(None)):
    logger.error('get_track_id: failed. id_track={0}, model_number={1}'.format(id_track, model_number))
    sys.exit(1)
  else:
    id = result[0]

  if not isinstance(id, int):
    logger.error('get_models_id_by_id_tracks_and_model_number: returned non-integer id!')
    sys.exit(1)

  if id < min_id:
    logger.error('get_models_id_by_id_tracks_and_model_number: id < min_id')
    sys.exit(1)

  if id > max_id:
    logger.error('get_models_id_by_id_tracks_and_model_number: id > min_id')
    sys.exit(1)

  return id

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_models_log_Teff_log_g_by_id(dbname, list_ids):
  """
  TBD ...
  """
  result = []
  n_ids  = len(list_ids)
  if n_ids == 0:
    logger.error('get_models_log_Teff_log_g_by_id: The input "list_ids" is empty')
    sys.exit(1)
  
  with db_def.grid_db(dbname=dbname) as the_db:
    q_id  = query


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   T R A C K S   T A B L E 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dic_look_up_track_id(dbname_or_dbobj):
  """
  Retrieve the id, M_ini, fov, Z, and logD from the entire "models" table, and construct a look up
  dictionary with the keys as the (M_ini, fov, Z, logD) tuple, and the values as the id. This gives
  a mapping of track ids to their corresponding attributes, which is very useful for the fastest 
  way to retrieve track ids by their attributes.

  @param dbname_or_dbobj: The first argument of this function can have two possible types. The reason 
        is that Python does not really support function overloading. Instead, it is careless about the
        type of the input argument, which we benefit from here. The reason behind this choice of 
        development is to avoid creating/closing a connection/cursor to the database everytime one 
        freaking model ID needs be fetched. This avoids connection overheads when thousands to 
        millions of track IDs need be retrieved.
        The two possible inputs are:
        - dbname: string which specifies the name of the dataase. This is used to instantiate the 
                  db_def.grid_db(dbname) object. 
        - dbobj:  An instance of the db_def.grid_db class. 
  @type dbname_or_dbobj: string or db_def.grid_db object

  """
  cmnd = 'select id, M_ini, fov, Z, logD from tracks;'

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd, None)
      result = the_db.fetch_all()
  #
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd, None)
    result   = dbname_or_dbobj.fetch_all()
  #
  else:
    logger.error('get_dic_look_up_track_id: Input type not string or db_def.grid_db! It is: {0}'.format(type(dbname)))
    sys.exit(1)

  if not isinstance(result, list):
    logger.error('get_dic_look_up_track_id: failed')
    sys.exit(1)

  n   = len(result)
  if n == 0:
    logger.error('get_dic_look_up_track_id: the result list is empty')
    sys.exit(1)

  list_id  = np.array([ result[k][0] for k in range(n) ])
  list_tup = [ (result[k][1], result[k][2], result[k][3], result[k][4]) for k in range(n) ]
  dic = dict()
  for key, val in zip(list_tup, list_id): dic[key] = val

  logger.info('get_dic_look_up_track_id: Successfully returning "{0}" records'.format(n))

  return dic


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_track_by_id(dbname, id):
  """
  Retrieve the four basic track attributes, M_ini, fov, Z, logD, respectively by the requested id.
  if the id exceeds the minimum and maximum id range in the database, an exception is raised, and
  the function terminates.

  @param dbname: database name, used to instantiate the db_def.grid_db(dbname) object
  @type dbname: string
  @param id: the unique id of the grid.tracks table to fetch the corresponding row
  @type id: integer
  @return: a tuple with (M_ini, fov, Z, logD), respectively
  @rtype: tuple
  """
  with db_def.grid_db(dbname=dbname) as the_db:

    cmnd = 'select %s between (select min(id) from tracks) and (select max(id) from tracks)'
    the_db.execute_one(cmnd, (id, ))
    if the_db.fetch_one() is False:
      logger.error('get_track_by_id: id={0} exceeds the available tracks.id range')
      sys.exit(1)

    cmnd = 'select M_ini, fov, Z, logD from tracks where id=%s'
    the_db.execute_one(cmnd, (id, ))
    result = the_db.fetch_one()

  return result

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_track_id(dbname_or_dbobj, M_ini, fov, Z, logD):
  """
  Retrieve the id for a track given the four basic parameters (attributes) the distinguish the track.

  @param dbname_or_dbobj: The first argument of this function can have two possible types. The reason 
        is that Python does not really support function overloading. Instead, it is careless about the
        type of the input argument, which we benefit from here. The reason behind this choice of 
        development is to avoid creating/closing a connection/cursor to the database everytime one 
        freaking track ID needs be fetched. This gives a nice speedup when thousands to millions of 
        track IDs need be retrieved.
        The two possible inputs are:
        - dbname: string which specifies the name of the dataase. This is used to instantiate the 
                  db_def.grid_db(dbname) object. 
        - dbobj:  An instance of the db_def.grid_db class. 
  @type dbname_or_dbobj: string or db_def.grid_db object
  @param M_ini: initial mass (in solar mass)
  @type M_ini: float
  @param fov: exponential overshoot parameter
  @type fov: float
  @param Z: initial metallicity
  @type Z: float
  @param logD: the logarithm of the diffusive mixing coefficient
  @type logD: float
  @return: the id of the corresponding row, if the row exists, and if the query succeeds.
        In case of a failure, we return False
  @rtype: integer
  """
  cmnd = 'select id from tracks where M_ini~%s and fov~%s and Z~%s and logD~%s'
  tup  = (M_ini, fov, Z, logD)

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd, tup)
      result = the_db.fetch_one()
  #
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd, tup)
    result   = dbname_or_dbobj.fetch_one()
  #
  else:
    logger.error('get_track_id: Input type not string or db_def.grid_db! It is: {0}'.format(type(dbname)))
    sys.exit(1)

  if result is None:
    logger.warning('get_track_id: failed: %s' % tup)
    return False
  else:
    return result[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  R O U T I N E S   F O R    G E N E R A L   U S E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_tables_info(dbname):
  """
  Retrieve the information of the tables in the database passed by its name (as dbname). The following
  informations are retrieved, and used as the key of the returned dictionary:
  - user_name
  - schema_name
  - table_name
  - index_name
  - is_unique
  - is_primary
  - index_type
  - indkey
  - index_keys
  - is_functional
  - is_partial

  Note that the value corresponding to each key is a list of strings, and the length of all these returned
  lists are identical.

  @param dbname: the database name
  @type dbname: string
  @return: a dictionary with the entire information, accessed through 11 keys listed above. The associated
      value of each key is a list of strings
  @rtype: dict
  """
  cmnd = 'SELECT \
          U.usename                AS user_name, \
          ns.nspname               AS schema_name, \
          idx.indrelid :: REGCLASS AS table_name, \
          i.relname                AS index_name,\
          idx.indisunique          AS is_unique, \
          idx.indisprimary         AS is_primary, \
          am.amname                AS index_type, \
          idx.indkey, \
               ARRAY( \
                   SELECT pg_get_indexdef(idx.indexrelid, k + 1, TRUE) \
                   FROM \
                     generate_subscripts(idx.indkey, 1) AS k \
                   ORDER BY k \
               ) AS index_keys, \
          (idx.indexprs IS NOT NULL) OR (idx.indkey::int[] @> array[0]) AS is_functional, \
          idx.indpred IS NOT NULL AS is_partial \
        FROM pg_index AS idx \
          JOIN pg_class AS i \
            ON i.oid = idx.indexrelid \
          JOIN pg_am AS am \
            ON i.relam = am.oid \
          JOIN pg_namespace AS NS ON i.relnamespace = NS.OID \
          JOIN pg_user AS U ON i.relowner = U.usesysid \
        WHERE NOT nspname LIKE %s; -- Excluding system tables'
  val  = ('pg%', )

  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, val)
    result = the_db.fetch_all()

  if result is None:
    logger.error('get_tables_info: failed')
    sys.exit(1)
  n       = len(result)

  # arrange all info as a dictionary
  dic     = dict()
  dic['user_name']     = [tup[0] for tup in result]
  dic['schema_name']   = [tup[1] for tup in result]
  dic['table_name']    = [tup[2] for tup in result]
  dic['index_name']    = [tup[3] for tup in result]
  dic['is_unique']     = [tup[4] for tup in result]
  dic['is_primary']    = [tup[5] for tup in result]
  dic['index_type']    = [tup[6] for tup in result]
  dic['indkey']        = [tup[7] for tup in result]
  dic['index_keys']    = [tup[8] for tup in result]
  dic['is_functional'] = [tup[9] for tup in result]
  dic['is_partial']    = [tup[10] for tup in result]

  return dic

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
