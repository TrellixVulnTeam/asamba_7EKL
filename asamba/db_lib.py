
"""
This module encapsulates a collection of useful interactions with different tables within the 
database, and provides a collection of routines which useful look-up dictionaries or tagging
dictionaries. Functions are grouped based on which table they query from.

In many places across this module, the use of numpy arrays are prohibited (though they provide 
a significant speed up compared to the brute-force Pythonic manipulation with lists and tuples).
There reason for that is a mismatch between the single-precision floating point as returned from
PostgreSQL (through psycopg2), compared to the numpy's pre-built floating precision. E.g. the value
a=0.1234 returned from psycopg2 is represented at a=0.12340000001, which absulutely screws up the
whole taging concept. For that reason, we stick to the slow Pythonic list/tuple comprehension, 
sorting, indexing, etc. But the gain is, the tags come out right!
"""

from __future__ import unicode_literals

from builtins import zip
from builtins import range
import sys, os, glob
import logging
import numpy as np 
import psycopg2

from asamba import db_def, query, utils

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  R O U T I N E S  T H A T   C O M B I N E   S E V E R A L   T A B L E S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dic_tag_Xc(dbname):
  """
  The models in the database start from ZAMS (Xc~0.7), and are evolved up to TAMS (Xc~0.002). However,
  the timesteps in MESA are dynamically determined, and are non-uniform. Therefore, no two tracks 
  ncesserily have identical Xc values for their models. As a result of that, it is difficult for some 
  other applications (e.g. interpolation between models, or marginalization of probabilities, etc) to 
  find/provide a logical connection between various models along various tracks. For that, we employ
  the *tagging* concept. See also get_dic_tag_track_attributes() for another case.

  The tagging is carried out simply as the following:
  - First, the unique tracks are found, and for each track, all Xc values are retrieved from the 
    "models" table.
  - All Xc values are sorted in decreasing order, so that the ZAMS model is the first in the row, and
    the TAMS model is the last, the same way as MESA stores the models
  - For an internal loop over all Xc values per one track, a tuple is created with the following order
    in order to use it as a key of the returned dictinary: (M_ini, fov, Z, logD, Xc)
  - Finally, for each model (equivalent to a key) a tag between 0 and N-1 is assigned to, where N is 
    the number of models stored per that specific track. Note that N is not necessarily fixed from one 
    track to another (because MESA timesteps depend on a collection of variables, among which the 
    convergence criteria and the timestep criteria).

  Notes:
  - An alternative to calling this function is to load the Xc tags from e.g. an ASCII file. For example
    of this, see the write.Xc_tags_to_ascii() and read.Xc_tags_from_ascii() functions
  - This operation depends on fetching all Xcs in the models table (over 3.8 million entries), and takes
    roughly 30 sec. So, please be patient.
  - There might be two different tracks with an identical Xc value, e.g. 0.1234. The current tagging 
    scheme may give these two models an identical or different tag. But, we do not care if a single Xc
    value maybe or not tagged differently. The important point to bear in mind is that each Xc has a 
    unique tag along its own specific track
  - The format of the key is fixed, is zero-padded, and can be found in write.Xc_tags_to_ascii()

  To access the Xc tag for a model in the grid, one can do the following:
  >>>from asamba import db_lib
  >>>dic_tag_Xc  = db_lib.get_dic_tag_Xc('grid')
  >>>key_model   = '12.099,0.035,0.014,01.29,0.2314'
  >>>this_Xc_tag = dic_tag_Xc[key_model]

  @param dbname: The name of the database to connect to, and fetch data from
  @type dbname: str
  @return: the tagging dictionary, where the tuple of the attributes of each model (as a dictionary 
      key) is mapped to a unique Xc integer-valued tag (see example above).
  @rtype: dict
  """
  # Get the contents of the tracks table as a record array
  tracks_vals = get_tracks(dbname=dbname)
  tracks_ids  = [tup[0] for tup in tracks_vals]
  arr_ids     = np.array(tracks_ids)  # only for fast indexing

  # Get all Xc values and their corresponding track_id from the models table
  dic_Xc      = get_dic_look_up_Xc(dbname_or_dbobj=dbname)
  Xc_keys     = list(dic_Xc.keys())
  Xc_ids      = np.array([tup[0] for tup in Xc_keys])
  Xc_vals     = [tup[1] for tup in Xc_keys]
  dtype       = [('id_track', np.int32), ('Xc', np.float32)]

  dic_tag       = dict()
  for k, track_id in enumerate(tracks_ids):
    this_track  = tracks_vals[k][1:]
    str_M_ini   = '{0:06.3f}'.format(this_track[0])
    str_fov     = '{0:05.3f}'.format(this_track[1])
    str_Z       = '{0:05.3f}'.format(this_track[2])
    str_logD    = '{0:05.2f}'.format(this_track[3])

    # Get all Xcs corresponding to this specific track, using track_id
    ind       = np.where(Xc_ids == track_id)[0]
    if len(ind) == 0:
      logger.error('get_dic_tag_Xc: Found no Xc for this track: ', tup_attrs)
      sys.exit(1)

    track_Xcs = [ Xc_vals[j] for j in ind ]
    track_Xcs.sort()
    track_Xcs = track_Xcs[::-1] # from ZAMS to TAMS
    for tag_Xc, this_Xc in enumerate(track_Xcs):
      str_Xc  = '{0:06.4f}'.format(this_Xc)
      key     = ','.join([str_M_ini, str_fov, str_Z, str_logD, str_Xc])
      dic_tag[key] = tag_Xc     # 'M_ini,fov,Z,logD,Xc': tag

  logger.info('get_dic_tag_Xc: Returning all tagging dictionaries')

  return dic_tag

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
def get_dic_look_up_Xc(dbname_or_dbobj):
  """
  Retrieve the id, id_track and Xc (core hydrogen mass fraction) from the entire "models" table, and 
  construct a look up dictionary with the keys as the (id_track, Xc) tuple, and the values as the id. 

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
  @return: look up dictinary with keys as a tuple with the two elements "(id_track, Xc)" and the value
        as the "models.id"
  @rtype: dict
  """
  cmnd = 'select id, id_track, Xc from models'

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
    logger.error('get_dic_look_up_Xc: Input type not string or db_def.grid_db! It is: {0}'.format(type(dbname)))
    sys.exit(1)

  if not isinstance(result, list):
    logger.error('get_dic_look_up_Xc: failed')
    sys.exit(1)

  n   = len(result)
  if n == 0:
    logger.error('get_dic_look_up_Xc: the result list is empty')
    sys.exit(1)

  # list_id  = np.array([ result[k][0] for k in range(n) ])
  list_id  = [tup[0] for tup in result]
  # list_tup = [ (result[k][1], result[k][2]) for k in range(n) ]
  list_tup = [ tup[1:] for tup in result ]
  dic = dict()
  for key, val in zip(list_tup, list_id): dic[key] = val

  logger.info('get_dic_look_up_Xc: Successfully returning "{0}" records'.format(n))

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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   T R A C K S   T A B L E 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dics_tag_track_attributes(dbname):
  """
  This routine returns four tagging dictionaries, each for one of the key attributes from the "tracks"
  table in the grid. Each of these dictionaries map a value into a unique integer tag. This allows to 
  track attributes using their tags, rather than their absolute values. Furthermore, the required tags
  are small integers, and are pretty lightweighted. Using these tags also facilitate carrying out 
  derivatives between model properties, and also interpolate in between them. This comes very handy 
  when carrying out integrations over posterior probabilities in order to marginalize w.r.t. to few
  parameters, where instead of values, we now use their tags for integration. For instance

  An important point to keep in mind is that the keys are strings, by rounding off the attribute values
  to a proper number of decimal points which also complies with the design of the grid table attributes.

  >>>from asamba import db_lib
  >>>dics_for_tags = db_lib.get_dics_tag_track_attributes('grid')
  >>>dic_tag_fov   = dics_for_tags[1]
  >>>key           = '0.025'
  >>>tag_fov_025   = dic_tag_fov[key]
  
  >>>dic_tag_logD  = dics_for_tags[3]
  >>>key           = '03.130,02.63'
  >>>tag_logD_2p63 = dic_tag_logD[key]

  @param dbname: the name of the database to connect to, and fetch information from
  @type dbname: str
  @return: four dictionaries, each providing a tagging facility to tag the features. Below, we provide
      the key: value example to retrieve data from each dictionary for each feature
      - '{0:06.3f}'.format(M_ini) --> tag
      - '{0:05.3f}'.format(fov)   --> tag
      - '{0:05.3f}'.format(Z)     --> tag
      - '{0:06.3f},{1:05.2f}'.format(M_ini, logD) --> tag
      The key in each dictionary is a tuple of one of the values of that quantity, and the returned 
      value is a uniqu integer-valued tag
  @type: tuple of dics
  """
  # Get the contents of the tracks table as a record array
  tracks_vals = get_tracks_as_recarray(dbname)

  # Find unique values for each column of the tracks table
  uniq_track_id  = tracks_vals.id
  uniq_M_ini  = np.unique(tracks_vals.M_ini)
  uniq_fov    = np.unique(tracks_vals.fov)
  uniq_Z      = np.unique(tracks_vals.Z)
  uniq_logD   = np.unique(tracks_vals.logD)

  # Assign a tag to each unique feature (M_ini, fov, Z, logD) and construct a dictionary, where the
  # key is the unique identify from the feature column, and the value of the dictionary is an integer 
  # tag. Note that for logD, this is tricky, whose value is mass dependent, and whose tag is always 
  # between 0 and 4, and requires a correct mapping with initial mass 
  #.........................
  def gen_dic(arr, code):
    """
    Returns a dictionary with the key as the tuple of each element "(key,)" and value an index 
    which starts from 0 and ends with N-1, for an input array of length N.
    """
    dic = dict()
    for k, key in enumerate(np.sort(arr)): 
      if code == 1:   # M_ini
        _key = '{0:06.3f}'.format(key) 
      elif code == 2 or code == 3: # fov and Z
        _key = '{0:05.3f}'.format(key) 
      elif code == 4: # logD 
        _key = '{0:05.2f}'.format(key)
      else:
        logger.error('get_dics_tag_track_attributes: gen_dic: Wrong code passed')
        sys.exit(1)
      dic[_key] = k
    return dic
  #.........................

  dic_tag_M_ini = gen_dic(arr=uniq_M_ini, code=1)
  dic_tag_fov   = gen_dic(arr=uniq_fov, code=2)
  dic_tag_Z     = gen_dic(arr=uniq_Z, code=2)

  # Now, fix logD tags (i.e. dictionary values) by walking over the uniq masses, and tracks_vals
  dic_tag_logD  = dict()
  for k, this_M in enumerate(uniq_M_ini):
    ind          = np.where(tracks_vals.M_ini == this_M)[0]
    this_logD    = np.sort(np.unique(tracks_vals.logD[ind]))
    if len(this_logD) != 5:
      logger.error('get_dics_tag_track_attributes: the length of the unique logD array != 5, bus is: {0}'.format(len(this_logD)))
      sys.exit(1)
    for tag, logD_key in enumerate(this_logD): 
      _key       = '{0:06.3f},{1:05.2f}'.format(this_M, logD_key)
      dic_tag_logD[_key] = tag # Voila

  logger.info('get_dics_tag_track_attributes: Returning tagging dics for (M_ini, fov, Z, logD)')

  return (dic_tag_M_ini, dic_tag_fov, dic_tag_Z, dic_tag_logD)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_dic_look_up_track_id(dbname_or_dbobj):
  """
  Retrieve the id, M_ini, fov, Z, and logD from the entire "tracks" table, and construct a look up
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
  # list_tup = [ (result[k][1], result[k][2], result[k][3], result[k][4]) for k in range(n) ]
  list_tup = [ result[k][1:] for k in range(n) ]
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
def get_tracks(dbname):
  """
  This function retrieves the entire content of the "tracks" table. Each record in that table is an
  item (tuple) in the returned list.

  @param dbname: name of the database to connect to, and fetch the data from.
  @type dbname: str
  @return: list of tuples where each tuple has this form: (id, M_ini, fov, Z, logD)
  @rtype: list of tuples
  """
  # Construct a recarray of the tracks table with: (id, M_ini, fov, Z, logD) as column names
  tracks_dic  = get_dic_look_up_track_id(dbname_or_dbobj=dbname)
  tracks_keys = list(tracks_dic.keys())
  tracks_id   = list(tracks_dic.values())
  tracks_rows = len(tracks_id)
  tracks_vals = [(tracks_id[k], ) + tracks_keys[k] for k in range(tracks_rows)]

  return tracks_vals

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_tracks_as_recarray(dbname):
  """
  This function retrieves the entire content of the "tracks" table, and returns it back as a numpy
  record array, with the record names being the tracks attributes, i.e. id, M_ini, fov, Z, and logD.
  For example, to find the unique initial masses used in the entire grid, you may do the following:

  >>>from asamba import db_lib
  >>>recarr = db_lib.get_tracks_as_recarray('grid')
  >>>all_masses = recarr.M_ini
  >>>uniq_masses = np.unique(all_masses)

  @param dbname: name of the database to connect to, and fetch the data from.
  @type dbname: str
  @return: named record array with five columns: id, M_ini, fov, Z, logD. Each row in the returned 
        array stands for one track in the grid
  @rtype: np.recarray
  """
  tracks_vals = get_tracks(dbname=dbname)
  tracks_cols = 1 + 4
  f32         = np.float32
  dtype       = [('id', np.int16), ('M_ini', f32), ('fov', f32), ('Z', f32), ('logD', f32)]
  tracks_vals = utils.list_to_recarray(tracks_vals, dtype)

  return tracks_vals

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
