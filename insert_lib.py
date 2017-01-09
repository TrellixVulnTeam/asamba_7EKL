
import sys, os, glob
import logging
import numpy as np 

import var_def 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logger.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_tracks_insert_commands(list_tracks):
  """
  Receive a list of "track" objects, and return a list of SQL insert commands to put the data into the
  "grid.tracks" table.
  @param list_tracks: list of "var_def.track" instances
  @type list_tracks: list
  @return: list of SQL commands, one item to insert one MESA history/track info into the database.
  @rtype: list of strings
  """
  n_tracks = len(list_tracks)
  if n_tracks == 0:
    logger.error('gen_tracks_insert_commands: Input list is empty')
    sys.exit(1)

  list_cmnds = []
  for i, track in enumerate(list_tracks):
    M_ini = track.M_ini
    fov   = track.fov
    Z     = track.Z
    logD  = track.logD
    cmnd  = """insert into tracks (initial_mass, f_overshoot, metalicity, log_D_mix) \
               values ({0}, {1}, {2}, {3})""".format(M_ini, fov, Z, logD)
    list_cmnds.append(cmnd)

  return list_cmnds

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
