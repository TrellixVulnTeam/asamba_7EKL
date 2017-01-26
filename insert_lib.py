
import sys, os, glob
import logging
import numpy as np 

from grid import var_def, db_def, db_lib

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   T O   I N T E R A C T   W I T H   T H E   D A T A B A S E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepare_insert_tracks(dbname_or_dbobj, include_id=False):
  """
  Execute the SQL "Prepare" statement in order to prepare for inserting rows into the tracks
  table
  """
  if include_id:
    cmnd   = 'prepare prepare_insert_tracks (integer, real, real, real, real) as \
              insert into %s.tracks (id, M_ini, fov, Z, logD) values \
              ($1, $2, $3, $4, $5)'
  else:
    cmnd   = 'prepare prepare_insert_tracks (real, real, real, real) as \
              insert into %s.tracks (M_ini, fov, Z, logD) values \
              ($1, $2, $3, $4)'

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd, (the_db.dbname, ))
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd, (dbname_or_dbobj.dbname, ))
  else:
    logger.error('prepare_insert_tracks: Input argument has unsupported type')
    sys.exit(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def execute_insert_into_tracks(dbname_or_dbobj, id, M_ini, fov, Z, logD):
  """
  Execute the prepared insertion statement. This implies that the function prepare_insert_tracks()
  is already executed, and the prepare statment is already activated in the open session.
  """
  if isinstance(id, type(None)):
    cmnd = 'execute prepare_insert_tracks (%s, %s, %s, %s)'
    tup  = (M_ini, fov, Z, logD)
  elif isinstance(id, int):
    cmnd = 'execute prepare_insert_tracks (%s, %s, %s, %s, %s)'
    tup  = (id, M_ini, fov, Z, logD)

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd, tup)
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd, tup)
  else:
    logger.error('execute_insert_into_tracks: Input argument dbname_or_dbobj has a wrong type!')
    sys.exit(1)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def insert_row_into_tracks(dbname_or_dbobj, id, M_ini, fov, Z, logD):
  """
  Inset one row into the "tracks" table of the database
  """
  if isinstance(id, type(None)):
    cmnd = 'insert into tracks (M_ini, fov, Z, logD) values (%s, %s, %s, %s)'
    tup  = (M_ini, fov, Z, logD)
  elif isinstance(id, int):
    cmnd = 'insert into tracks (id, M_ini, fov, Z, logD) values (%s, %s, %s, %s, %s)'
    tup  = (id, M_ini, fov, Z, logD)
  else:
    logger.error('insert_row_into_tracks: Input argument id has unexpected type')
    sys.exit(1)

  if isinstance(dbname_or_dbobj, str):
    with db_def.grid_db(dbname=dbname_or_dbobj) as the_db:
      the_db.execute_one(cmnd, tup)
  elif isinstance(dbname_or_dbobj, db_def.grid_db):
    dbname_or_dbobj.execute_one(cmnd, tup)
  else:
    logger.error('insert_row_into_tracks: Input argument dbname_or_dbobj has a wrong type!')
    sys.exit(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   T O   W O R K   W I T H   A N   I N P U T   A S C I I   F I L E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def insert_models_from_models_parameter_file(ascii_in):
  """

  """
  if not os.path.exists(ascii_in):
    logger.error('insert_models_from_models_parameter_file: "{0}" does not exist'.format(ascii_in))
    sys.exit(1)
  else:     # get the file handle
    handle = open(ascii_in, 'r')

  # walk over the input file, and insert each row one after the other
  i        = -1
  with db_def.grid_db() as the_db:
    while  True:
      i      += 1
      if i <= 0:      # skip the header
        hdr  = handle.readline()
        continue

      line   = handle.readline()
      if line == '': break

      model  = model_line_to_model_object(line)

      id_track = db_lib.get_track_id(M_ini=M_ini, fov=fov, Z=Z, logD=logD)
      print id_track

      i      += 1

      if i == 10: break

  handle.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def insert_tracks_from_models_parameter_file(dbname, ascii_in):
  """
  This routine is protected agains *Injection Attacks*.
  """
  if not os.path.exists(ascii_in):
    logger.error('get_insert_commands_from_models_parameter_file: "{0}" does not exist'.format(ascii_in))
    sys.exit(1)

 try:
    assert db_def.exists(dbname=dbname) == True
    # the_db = db_def.grid_db()
    logger.info('insert_tracks_from_models_parameter_file: database "{0}" exists.'.format(dbname))
  except:
    logger.error('insert_tracks_from_models_parameter_file: failed to instantiate database "{0}".'.format(dbname))
    sys.exit(1)


  # open the file, and get the file handle
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
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.prepare_insert_tracks(include_id=False)
    # execute many and commit
    logger.info('insert_tracks_from_models_parameter_file: "{0}" tracks recognized'.format(n_values))
    cmnd   = 'execute prepare_insert_tracks ({0})'.format(','.join(['?']*4))
    the_db.execute_many(cmnd=cmnd, values=tups)
    logger.info('insert_tracks_from_models_parameter_file: Insertion completed.')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# A U X I L A R Y    F U N C T I O N S
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
    cmnd  = """insert into tracks (M_ini, fov, Z, logD) \
               values ({0}, {1}, {2}, {3})""".format(M_ini, fov, Z, logD)
    list_cmnds.append(cmnd)

  return list_cmnds

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def model_line_to_model_object(line):
  """
  Convert one row of the ascii parameter file, into an instance of the var_def.model class, and store
  the line values into those of the corresponding attribute. The returned object is useful to insert
  into the SQL database

  @param line: one line of the parameter file, which contains numbers only
  @type line: string
  @return: an instance of the var_def.model class, with the attributes all set from the input line.
  @rtype: object
  """
  if not isinstance(line, str):
    logger.error('model_line_to_model_object: The input is not a string')
    sys.exit(1)
  line        = line.rstrip('\r\n').split()

  model       = var_def.model()
  # model_attrs = dir(model)
  
  # exclude     = ['__doc__', '__init__', '__enter__', '__exit__', '__del__', '__module__', 
  #                'filename', 'track', 'set_by_dic', 
  #                'set_filename', 'set_track', 'get']
  # model_attrs = [attr for attr in model_attrs if attr not in exclude]
  # basic_attrs = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number'] # treated manually below
  # other_attrs = [attr for attr in model_attrs if attr not in basic_attrs]
  # color_attrs = set(['U_B', 'B_V', 'V_R', 'V_I', 'V_K', 'R_I', 'I_K', 'J_H', 'H_K', 'K_L', 'J_K',
  #                    'J_L', 'J_Lp', 'K_M'])

  avail_attrs = dir(model)
  exclude     = set(['__init__', '__doc__', '__module__', 'filename', 'track'])
  avail_attrs = [attr for attr in avail_attrs if attr not in exclude]
  avail_attrs = [attr for attr in avail_attrs if 'set' not in attr and 'get' not in attr]
  key_attrs   = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number']
  other_attrs = [attr for attr in avail_attrs if attr not in key_attrs]

  dic         = dict()
  for i, attr in enumerate(key_attrs):
    val       = line[i]
    conv      = float
    model[attr]
    if attr == 'model_number':
      conv    = int
    setattr(model, attr, val)

  return model

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







