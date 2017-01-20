
import sys, os, glob
import logging
import numpy as np 
import psycopg2

from grid import db_def

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
def get_tracks_id(dbname, M_ini, fov, Z, logD):
  """
  Retrieve the id for a track given the four basic parameters (attributes) the distinguish the track.

  @param dbname: database name, used to instantiate the db_def.grid_db(dbname) object
  @type db: string
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
  tup  = (M_ini, fov, Z, logD)
  cmnd = 'select id from tracks where M_ini~%s and fov~%s and Z~%s and logD~%s'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, tup)
    result = the_db.fetch_one()
    id     = result[0]

  return id

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

