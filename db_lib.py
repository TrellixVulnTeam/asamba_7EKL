
import sys, os, glob
import logging
import numpy as np 
import psycopg2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_tracks_id(db, M_ini, fov, Z, logD):
  """
  Retrieve the id for a track given the four basic parameters (attributes) the distinguish the track.

  @param db: an instance of the db_def.grid_db() class
  @type db: class object
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
  # curs = db.get_cursor()
  tup  = (M_ini, fov, Z, logD)
  cmnd = 'select id from grid.tracks where M_ini=%s and fov=%s and Z=%s and logD=%s'
  print cmnd
  with db as the_db:
    id = the_db.execute_one(cmnd, tup)

  print id

  return id

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

