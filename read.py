
import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_mesa_ascii(filename):
  """
  Read an history or profile ascii output from MESA.
  @param filename: full path to the input ascii file
  @type filename: string
  @param dtype: numpy-compatible dtype object. if it is not provided, it will be retrieved from read.get_dtype()
  @type dtype: list of tuples
  @return dictionary of the header of the file, and the record array for the data block. It can be called like this
     >>> header, data = read.read_mesa_ascii('filename')
  @rtype: dictionary and numpy record array
  """
  if not os.path.isfile(filename):
    logger.error('read_mesa_ascii: {0} does not exist'.format(filename))

  with open(filename, 'r') as r: lines = r.readlines()
  logger.info('read_mesa_ascii: {0} successfully read'.format(filename))

  skip          = lines.pop(0)
  header_names  = lines.pop(0).rstrip('\r\n').split()
  header_vals   = lines.pop(0).rstrip('\r\n').split()
  temp          = np.array([header_vals], float).T
  header        = np.core.records.fromarrays(temp, names=header_names)
  skip          = lines.pop(0)
  skip          = lines.pop(0)

  col_names     = lines.pop(0).rstrip('\r\n').split()
  n_cols        = len(col_names)

  int_columns   = [ 'mix_type_1', 'mix_type_2', 'mix_type_3', 'mix_type_4', 'mix_type_5', 'mix_type_6',             # hist
                    'burn_type_1', 'burn_type_2', 'burn_type_3', 'burn_type_4', 'burn_type_5', 'burn_type_6',         # hist
                    'mixing_type', 'mlt_mixing_type', 'sch_stable', 'ledoux_stable', 'stability_type',                # prof
                    'num_zones', 'cz_zone', 'cz_top_zone', 'num_backups', 'num_retries', 'zone',
                    'model_number', 'version_number', 'nse_fraction' ]

  dtypes        = []
  for col in col_names:
    if col in int_columns:
      dtypes.append( (col, int) )
    else:
      dtypes.append( (col, float) )

  data          = []
  for i_line, line in enumerate(lines):
    if not line.rstrip('\r\n').split(): continue  # skip empty lines
    data.append(line.rstrip('\r\n').split())

  data = np.core.records.fromarrays(np.array(data, float).transpose(), dtype=dtypes)

  return header, data

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_tracks_parameters_from_ascii(ascii_in):
  """
  This routine reads the contents of an ascii file which tabulates the track parameters, and returns
  a list of "var_def.track()" objects, one per each row in the file. The list can be used later on
  for any manipulation (plotting, inserting into the database, etc). Note that we skip the first row
  as the header.

  @param ascii_in: the full path to the already-available ascii file that contains the entire (or part)
         of the tracks parameters. This file can be generated by first calling the function  
         write_tracks_parameters_to_ascii().
  @type ascii_out: string
  @return: list of instances of var_def.track() class objects, one object per each row (i.e. track).
  @rtype: list
  """
  if not os.path.exists(ascii_in):
    logger.error('read_tracks_parameters_from_ascii: {0} does not exist'.format(ascii_in))

  with open(ascii_in, 'r') as r: lines = r.readlines()
  header  = lines.pop(0)
  n_lines = len(header)
  list_tracks = []

  for i, line in enumerate(lines):
    row   = line.rstrip('\r\n').split(' ')
    M_ini = float(row[0])
    fov   = float(row[1])
    Z     = float(row[2])
    logD  = float(row[3])

    a_track = var_def.track(M_ini=M_ini, fov=fov, Z=Z, logD=logD)
    list_tracks.append(a_track)

  logger.info('read_tracks_parameters_from_ascii exited successfully')
  print ' - read: read_tracks_parameters_from_ascii exited successfully'

  return list_tracks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

