
import sys, os, glob
import logging
import numpy as np 
import psycopg2

from grid import db_def

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E L S   T A B L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
  @return: the id of the corresponding row, if the row exists, and if the querry succeeds.
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
  else:
    return result[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

