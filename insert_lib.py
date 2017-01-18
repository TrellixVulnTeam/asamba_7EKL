
import sys, os, glob
import logging
import numpy as np 

from grid import var_def, db_def

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def insert_tracks_from_models_parameter_file(ascii_in):
  """
  This routine is protected agains *Injection Attacks*.
  """
  if not os.path.exists(ascii_in):
    logger.error('get_insert_commands_from_models_parameter_file: "{0}" does not exist'.format(ascii_in))
    sys.exit(1)

  # get a cursor to the grid
  try:
    the_db = db_def.grid_db()
    # cursor = the_db.get_cursor()
    logger.info('insert_tracks_from_models_parameter_file: grid_db instantiated.')
  except:
    logger.error('insert_tracks_from_models_parameter_file: failed to instantiate grid_db.')

  # enumerate the number of available rows in the file
  # with open(ascii_in, 'r') as handle:
  #   lines  = handle.readlines()
  #   n_rows = len(lines)
  #   lines  = []               # clean up the list by deallocating its content

  # reopen the file, and get the file handle
  handle   = open(ascii_in, 'r')
  # iterate over each row in the file, and construct the insertion command
  # unique   = set()
  tups     = []

  i        = -1
  while True:
    i      += 1
    if i <= 0:     # skip the header
      hdr  = handle.readline()
      continue     

    line   = handle.readline()
    if not line: break      # exit by the End-of-File (EOF)
    row    = line.rstrip('\r\n').split()

    M_ini  = float(row[0])
    fov    = float(row[1])
    Z      = float(row[2])
    logD   = float(row[3])
    tup    = (M_ini, fov, Z, logD)

    # progress bar on the stdout
    sys.stdout.write('\r')
    sys.stdout.write('progress = {0:.2f} % '.format(100. * i/3.845e6))
    sys.stdout.flush()

    # if tup in unique: continue
    tups.append(tup)

  handle.close()
  print

  # pick only unique sets of the track parameters
  unique   = set(tups)
  tups     = list(unique)

  n_values = len(tups)
  cmnd     = 'insert into grid.tracks (M_ini, fov, Z, logD) \
              values (%s, %s, %s, %s)'

  # execure many and commit
  logger.info('insert_tracks_from_models_parameter_file: "{0}" tracks recognized'.format(n_values))
  the_db.execute_many(cmnd=cmnd, values=tups)
  logger.info('insert_tracks_from_models_parameter_file: Insertion completed.')

  return

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
    cmnd  = """insert into tracks (M_ini, fov, Z, logD) \
               values ({0}, {1}, {2}, {3})""".format(M_ini, fov, Z, logD)
    list_cmnds.append(cmnd)

  return list_cmnds

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
