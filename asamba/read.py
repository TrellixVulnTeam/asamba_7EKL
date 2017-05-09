
"""
This module provides basic functionalities to read a variety of data e.g. in ASCII 
and HDF5 etc. formats. The highlight of the module is the "read_mesa_ascii()" function
which can read MESA history or profile files.
"""
from __future__ import print_function
from __future__ import unicode_literals

from builtins import zip
from builtins import range
import sys, os, glob
import logging
import numpy as np 
import h5py

from asamba import var_def

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_inlist(filename):
  """
  This function reads an ASCII file which specifies any set of options as a list of tuples (attr, val) for 
  valid entries in the file. It follows the same idea as the widely-used Fortran inlists.

  The user comments, specified using "#" are trimmed off from anywhere in the lines, so that one may comment 
  the inlist file in a line before the attr = val set or after it. E.g., the following two options are both 
  valid:

     # Here, I specify the name of my star
     name = 'beta Cephei'
  
  or 
  
     name = 'beta Cephei'  # The name of my star

  if the val is 'True', it is set to boolean True, if it is 'False', it is set to boolean False, if it contains
  '.', 'e+' or 'e-', it is interpreted as a fload, and otherwise, it is converted to integer. So, a great caution
  has to be practiced when assigning values to the attributes in the inlist files.

  @param filename: full path to the inlist file
  @type filename: str
  @return: a list of (attr, val) tuples, where 
  """
  if not os.path.exists(filename):
    logger.error('read_inlist: The input file "{0}" not found'.format(filename))
    sys.exit(1)

  with open(filename, 'r') as r: lines = r.readlines()

  options  = []
  numbers  = '+-0123456789'
  for k, line in enumerate(lines):
    line   = line.strip() # rstrip('\r\n')
    if '#' in line:
      ind  = line.find('#')
      line = line[:ind]
    if not line: continue # skip empty lines
    if '=' not in line: continue
    attr, val = line.split('=')
    attr   = attr.strip() # remove spaces
    val    = val.strip()  # remove spaces
   
    if '"' in val:      # look after string inputs
      if val.count('"') == 2:
        continue        # string input detected
      else:
        logger.error('read_inlist: Ambiguous string detected in line: {0} of file: {1}'.format(line, filename))
        sys.exit(1)
    elif "'" in val:
      if val.count("'") == 2:
        continue        # string input detected
      else:
        logger.error('read_inlist: Ambiguous string detected in line: {0} of file: {1}'.format(line, filename))
        sys.exit(1)
    elif val == 'True': # look after boolean inputs 
      val = True 
    elif val == 'False': 
      val = False
    elif '.' in val:    # look after float inputs
      val = float(val)
    elif 'e' in val:
      i_e = val.index('e')
      before = val[i_e-1] in numbers
      after  = val[i_e+1] in numbers
      if all([before, after]):
        val  = float(val)
      else:
        logger.error('read_inlist: Failed to interpret line: {0} in file: {1}'.format(line, filename))
        sys.exit(1)
    else:               # default: set to integer
      val = int(val)

    options.append((attr, val))
  
  logger.info('read_inlist: Successfully read file: "{0}"'.format(filename))

  return options

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gyre_h5(filename):
  """
  Read the GYRE output HDF5 file in full detail, and return an instance of the var_def.modes() with
  relevant attributes filled up. Thus, this routine reads the summary file or the eigenfunction file
  conveniently. Example of use:

  >>>from asamba import read
  >>>gyre_file = '/home/user/projects/gyre/beta_Cep.h5'
  >>>mode_list = gyre_h5(gyre_file)
  >>>freq      = np.real( mode_list.freq )

  @param filename: full path to the output GYRE HDF5 file
  @type filename: string
  @return: an instance of the var_def.modes() class
  @rtype: object
  """
  if not os.path.exists(filename):
    logger.error('gyre_h5: "{0}" does not exist'.format(filename))
    sys.exit(1)

  complex_dtype = np.dtype([('re', '<f8'), ('im', '<f8')])

  with h5py.File(filename, 'r') as h5:
    with var_def.modes() as modes:
      for attr_key, attr_val in zip(list(h5.attrs.keys()), list(h5.attrs.values())):
        modes.set(attr_key, attr_val)
      for column_key in list(h5.keys()):
        if h5[column_key].dtype == complex_dtype:
          column_val = h5[column_key][...]['re'] + 1j * h5[column_key][...]['im']
        else:
          column_val = h5[column_key][...]
        modes.set(column_key, column_val)

  return modes

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_mesa_ascii(filename):
  """
  Read a history or profile ascii output from MESA.
  An example of using this function to read the file "input_file" is the following

  >>> input_file = '/home/user/my-files/The_Sun/LOGS/history.data'
  >>> header, data = read.read_mesa_ascii(input_file)

  @param filename: full path to the input ascii file
  @type filename: string
  @return dictionary of the header of the file, and the record array for the data block. 
  @rtype: dictionary and numpy record array
  """
  if not os.path.isfile(filename):
    logger.error('read_mesa_ascii: {0} does not exist'.format(filename))
    sys.exit(1)

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

  int_columns   = [ 'model_number', 'version_number', 'sch_stable', 'ledoux_stable', 
                    'stability_type', 'num_zones', 'cz_zone', 'cz_top_zone', 
                    'num_backups', 'num_retries', 'zone', 'nse_fraction' ]

  dtypes        = []
  for col in col_names:
    if '_type' in col:
      dtypes.append( (col, int) )
    elif col in int_columns:
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
def read_models_parameters_from_ascii(ascii_in):
  """
  Warning: If the size of the input ascii is too large (which is practically the case), then this 
  function crashes raising a MemoryError exception.

  Read the contents of the input ASCII file containing the whole grid models data.

  @param ascii_in: full path to the ASCII file to be read
  @type ascii_in: string
  @return: array containing the whole data. Each field can be accessed using the same attributes of
           the var_def.model class object.
  @rtype: numpy record array
  """
  if not os.path.exists(ascii_in):
    logger.error('read_models_parameters_from_ascii: {0} does not exist'.format(ascii_in))
    sys.exit(1)

  # First, prepare the columns names, similar to the way the ASCII file is written
  # The following block is adopted from write.write_model_parameters_to_ascii()
  a_model     = var_def.model()
  model_attrs = dir(a_model)
  exclude     = ['__doc__', '__init__', '__enter__', '__exit__', '__del__', '__module__', 
                 'filename', 'track', 'set_by_dic', 
                 'set_filename', 'set_track', 'get']
  model_attrs = [attr for attr in model_attrs if attr not in exclude]
  basic_attrs = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number'] # treated manually below
  other_attrs = [attr for attr in model_attrs if attr not in basic_attrs]
  color_attrs = set(['U_B', 'B_V', 'V_R', 'V_I', 'V_K', 'R_I', 'I_K', 'J_H', 'H_K', 'K_L', 'J_K',
                     'J_L', 'J_Lp', 'K_M'])

  # prepare the column dtypes for the numpy recarray
  dtypes      = []
  for attr in basic_attrs + other_attrs:
    if attr   == 'model_number':
      dtypes.append( (attr, np.int16) )
    else:
      dtypes.append( (attr, np.float32) )
  n_cols      = len(dtypes)

  # read/load the file, and count the number of lines
  with open(ascii_in, 'r') as file: lines = file.readlines()
  n_rows      = len(lines)   # including the one line of the header
  lines       = []           # to garbage the contents of this list, and free RAM memory

  # get the file handle again, and read each line of the file iteratively to minimize RAM
  handle      = open(ascii_in, 'r')

  # iterate over the lines list, and fill up the record array
  rows        = []
  for i_row in range(n_rows):
    if i_row  == 0: 
      header  = handle.readline()
      continue
    else:
      line    = handle.readline()
      row     = line.rstrip('\r\n').split()
      rows.append(row)

  handle.close()

  # create the record array
  rec         = np.core.records.fromarrays(np.array(rows, float).transpose(), dtype=dtypes)

  logger.info('read_models_parameters_from_ascii: returned recarry of file: "{0}"'.format(ascii_in))

  return rec

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
    sys.exit(1)

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
  print(' - read: read_tracks_parameters_from_ascii exited successfully')

  return list_tracks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

