
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

from asamba import var_def, utils

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def convert_val(str_val):
  """
  This function receives an integer, float or a boolean variable in a string representation, identifies
  the correct type, and returns the variable in the expected/appropriate python native data type. 
  E.g. '1' --> 1, and 'True' --> True

  @param str_val: 
  """
  if not isinstance(str_val, str):
    logger.error('convert_val: Input: "{0}" is not a string variable'.format(str_val))
    sys.exit(1)

  numbers  = '+-0123456789'
  if str_val.lower() == 'true': # look after boolean inputs 
    val = True 
  elif str_val.lower() == 'false': 
    val = False
  elif '.' in str_val:    # look after float inputs
    val = float(str_val)
  elif 'e' in str_val:
    i_e = str_val.index('e')
    before = str_val[i_e-1] in numbers
    after  = str_val[i_e+1] in numbers
    if all([before, after]):
      val  = float(str_val)
    else:
      logger.error('convert_val: Failed to interpret line: {0} in file: {1}'.format(line, filename))
      sys.exit(1)
  else:               # default: set to integer
    val = int(str_val)

  return val

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

  As a nice feature, the user can even toss in a list/tuple of values, e.g. var = [1, 2.3, True]. Each element inside
  the list/tuple will be split (comma as a delimiter), and converted to the correct datatype by calling the 
  function convert_val().

  @param filename: full path to the inlist file
  @type filename: str
  @return: a list of (attr, val) tuples, where 
  """
  if not os.path.exists(filename):
    logger.error('read_inlist: The input file "{0}" not found'.format(filename))
    sys.exit(1)

  with open(filename, 'r') as r: lines = r.readlines()

  options  = []
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
        pass        # string input detected
      else:
        logger.error('read_inlist: Ambiguous string detected in line: {0} in file: {1}'.format(line, filename))
        sys.exit(1)
    elif "'" in val:

      if val.count("'") == 2:
        pass        # string input detected
      else:
        logger.error('read_inlist: Ambiguous string detected in line: {0} in file: {1}'.format(line, filename))
        sys.exit(1)
    elif '[' in val and ']' in val:
      i_l  = val.index('[')
      i_r  = val.index(']')
      vals = val[i_l+1 : i_r].split(',')
      val  = [convert_val(v) for v in vals]
    elif '(' in val and ')' in val:
      i_l  = val.index('(')
      i_r  = val.index(')')
      vals = val[i_l+1 : i_r].split(',')
      val  = [convert_val(v) for v in vals]
    else:
      val  = convert_val(val)

    # print (k, attr, val)
    options.append((attr, val))
  
  logger.info('read_inlist: Successfully read file: "{0}"'.format(filename))

  return options

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def modes_from_file(filename, delimiter=''):
  """
  Load a file, and insert all meaningful columns in the file (i.e. those that have the same header name
  as those of the "modes" class) into the "mode" attribute. It returns a list of modes, where each mode
  corresponds to one line in the file. The columns are delimited based on the passed delimiter

  Strict formatting of the input file:
  - The file has two headers as the first two lines:
    + line 1: the name of each column
    + line 2: the Python-intrinsic format of each line, e.g. int, float, boolean
  - The header names must be identical to the attributes of the mode object
  - If one attribute is unknown for all modes of the same star, that column would better be omitted.
    E.g. if for a star we do not know the modes form a frequency spacing or not, we leave this column
    off, instead of setting it off for all modes.
  - if a value for a mode is unknown, the column must read None or none. Never leave an unknown column
    empty.
  - For boolean columns, "1" means True, and "0" means False.

  @param filename: the full path where the 
  @type filename: str
  @param delimiter: the delimiting character between the columns, e.g. ',', or space, etc. Default: ''
  @type delimiter: str
  @return: list of modes
  @rtype: list
  """
  from asamba.star import mode
  
  if not os.path.exists(filename):
    logger.error('modes_from_file: The file "{0}" does not exist'.format(filename))
    sys.exit(1)

  with open(filename, 'r') as r: lines = r.readlines()
  n_lines = len(lines)
  if n_lines <= 2:
    logger.error('modes_from_file: Input file must have two lines of header and at least one mode line')
    sys.exit(1)

  header  = lines.pop(0).rstrip('\r\n').split(delimiter)
  header  = [val.strip() for val in header]
  n_hdr   = len(header)
  if n_hdr < 2:
    logger.error('modes_from_file: There must be at least two columns in the file')
    sys.exit(1)

  types   = lines.pop(0).rstrip('\r\n').split(delimiter)
  types   = [val.strip() for val in types]
  n_types = len(types)
  if n_types != n_hdr:
    logger.error('modes_from_file: The 1st and 2nd line must have identical number of columns!')
    sys.exit(1)
  conv    = []
  for k, t in enumerate(types):
    t     = t.lower()
    if t == 'int':
      conv.append(int)
    elif t == 'float':
      conv.append(float)
    elif t == 'boolean' or t == 'bool':
      conv.append(bool)
    elif t == 'str':
      conv.append(str)
    else:
       print('t is: {0}'.format(t), k)
       logger.error('modes_from_file: 2nd line can only have "int", "float", "str" or "bool".') 
       sys.exit(1)

  # iteratively load a mode and store in a list
  loaded  = []
  for k, line in enumerate(lines):
    line   = line.rstrip('\r\n').split(delimiter)
    a_mode = mode()

    for j, attr in enumerate(header):
      val  = line[j].strip()
      if val.lower == 'none':
        a_mode.setattr(attr, None)
      else:
        func = conv[j]
        val  = func(val)
        try:
          a_mode.set(attr, val)
        except:
          logger.error('modes_from_file: Unrecognized attribute found:')
          print(k, j, func, line[j], attr, val, hasattr(a_mode, attr))
          sys.exit(1)

    loaded.append(a_mode)
  
  # check frequency monotonicity
  freqs  = np.array([this.freq for this in loaded])
  d_freq = freqs[1:] - freqs[:-1]
  if not all(df > 0 for df in d_freq):
    logger.error('modes_from_file: The mode frequencies must be strictly increasing')
    sys.exit(1)

  return loaded

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
def read_rotation_frequencies_from_ascii(ascii_in):
  """
  This routine reads the ASCII file that contains the rotation frequency and break up frequency of 
  the whole database. It is useful for reading and COPYing the contents of the file to the columns
  of the "rotation_frequencies" table. The ASCII file is created by calling the function: 
  write.write_rotation_frequencies_to_ascii()

  @param ascii_in: full path to the ASCII file to be read
  @type ascii_in: str
  @return: array containing the whole data; the column names are taken from the file header, and are
         id_model, id_rot, freq_crit, and freq_rot. The last two frequency values are in Hz.
  @rtype: np.recarray
  """
  if not os.path.exists(ascii_in):
    logger.error('read_rotation_frequencies_from_ascii: {0} does not exist'.format(ascii_in))
    sys.exit(1)

  with open(ascii_in, 'r') as r: lines = r.readlines()
  n_lines = len(lines)
  header  = lines.pop(0)
  header  = header.rstrip('\r\n').split(',')
  types   = [int, np.int8, np.float32, np.float32]
  dtypes  = list(zip(header, types))

  rows    = []
  for k, line in enumerate(lines):
    line  = line.rstrip('\r\n').split(',')
    id_model  = int(line[0])
    id_rot    = int(line[1])
    freq_crit = float(line[2])
    freq_rot  = float(line[3])
    row       = (id_model, id_rot, freq_crit, freq_rot)
    rows.append(row)

  return utils.list_to_recarray(rows, dtypes) 

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
def sampling_from_h5(filename, dset_name):
  """
  This function reads the learning set from an HDF5 file. It is called by sampler.read_sample_from_hdf5()
  and sampler.load_sample_from_hdf5() methods, too. Example of use:

  >>>from asamba import read
  >>>h5_file = '/home/user/asamba_project/my_star_sampling.h5'
  >>>(learning_x, datatype) = read.sampling_from_h5(filename=h5_file, dset_name='learning_x')
  >>>print(learning_x.shape)

  Notes:
  - To write the sampling data in HDF5 format, one may call the function write.write_samplint_to_h5().
  - The dataset has a fixed name, *learning_set*, which is used internally to access it.
  - Depending on the choice made during the writing of this file, the number of returned columns 
    is not fixed, depending on whether or not the periods were also stored to the file.

  @param filename: full path to an already-available HDF5 file
  @type filename: str
  @param dset_name: name of the target dataset to be read from the HDF5 file. 
  @type dset_name: str
  @return: a tuple with the following two elements:
      - dataset: a numpy.ndarray with two dimensions (matrix) of shape m x (n+K), containing m rows 
        of examples, and n+K columns, where n=6 is the number of features (e.g. M_ini, fov, ...), and 
        K is the number of observed frequencies. If (K) periods were also saved along, then n+2K 
        columns will be returned.
      - datatype: a list of tuples specifying the numpy.dtype for all columns
  @rtype: tuple  
  """
  if not os.path.exists(filename):
    logger.error('sampling_from_h5: The input file {0} does not exist'.format(filename))
    sys.exit(1)

  with h5py.File(filename, 'r') as h5: 
    dset_names = h5.keys()
    if dset_name not in dset_names:
      logger.error('sampling_from_h5: Dataset name: {0} is unavailable in the HDF5 file'.format(dset_name))
      sys.exit(1)

    dset   = h5[dset_name]
    data   = h5[dset_name].value
    dtp    = dset.dtype
    nrows  = dset.attrs['num_rows']
    ncols  = dset.attrs['num_columns']
    cols   = dset.attrs['column_names']
    if not isinstance(cols, list): cols = list(cols) # for those datasets with only one column

  dtype  = [(col, dtp) for col in cols]

  logger.info('sampling_from_h5: Done. "{0}" size: {1} x {2} = {3} elements'.format(
              dset_name, nrows, ncols, data.size))

  return (data, dtype)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Xc_tags_from_ascii(filename='data/tags/Xc-tags.txt'):
  """
  This routine reads in the Xc tags from an ASCII file, as provided through the filename, and returns
  a tuple with a list of model attributes (M_ini, fov, Z, logD, Xc), and the corresponding Xc tag.
  Note that the user of numpy.ndarrays are prohibited here, because it messes up the limited floating
  point precision of the attributes, as designed in the grid. To remedy this, we round each attribute
  to a fixed number of decimal points, using the built-in round() method for floats. The tags are, 
  indeed, integers. Another possibility was to use the "decimal" class, but we avoid complicating the 
  datatypes used under the hood further (what we already deal with between Python, pscyopg2 and numpy 
  is already enough).

  Note:
   - Using ASCII seems so far the most robust way to retrieve the taggings, but come with a cost of 
     filesize and being slow.
   - The input file is roughly 125 MB.
   - Because every value per each line has to be rounded to a definite decimal point, this reading
     operation is pretty slow (~ a minute). So, be patient with it.
   - The keys are comma-separated values, rounded up to a certain number of decimals which complies 
     with the design/schema of the database in PostgreSQL. E.g. a key looks like the following:
     "12.345,0.015,0.018,0.00,0.5683" 

  @param filename: full path to where the ASCII file is stored.
  @type filename: str
  @return: a dictionary with the following key/value design -- identical to the output of the function 
        db_lib.get_dic_tag_Xc():
        - Keys: are attribute tuples, in the format (M_ini, fov, Z, logD, Xc). The number of decimals
          for M_ini is 3, for fov is 3, for Z is 3, for logD is 2 and for Xc is 4.
          Note again: the keys are comma-separated strings
        - Values: list of Xc tags, where all values are basically integers.
  @rtype: tuple
  """
  targz        = filename.replace('.txt', '.tar.gz')
  txt_exists   = os.path.exists(filename)
  targz_exists = os.path.exists(targz)
  extract      = (not txt_exists) and targz_exists
  stop         = (not txt_exists) and (not targz_exists)
  if txt_exists:
    pass
  elif stop:
    logger.error('Xc_tags_from_ascii: Neither the .txt or .tar.gz file "{0}" exist'.format(filename))
    sys.exit(1)
  elif extract:
    import tarfile
    with tarfile.open(targz) as this_tar:
      logger.info('\nXc_tags_from_ascii: Extracting the file: {0}'.format(targz))
      this_tar.extractall(path=os.path.dirname(filename))
  else:
    logger.error('Xc_tags_from_ascii: Unexpected situation occured')
    sys.exit(1)


  with open(filename, 'r') as r: lines = r.readlines()
  header = lines.pop(0).rstrip('\r\n').split(',')
  if len(header) != 6:
    logger.error('Xc_tags_from_ascii: Wrong number of header columns!')
    sys.exit(1)
  first  = lines[0]
  comma  = first.rfind(',')
  first  = first.rstrip('\r\n').split(',')
  if len(first) != 6:
    logger.error('Xc_tags_from_ascii: Wrong number of data columns!')
    sys.exit(1)

  dic    = dict()
  for k, line in enumerate(lines):
    str_attr  = line[:comma]
    tag       = int(line[comma+1:])
    dic[str_attr] = tag

  logger.info('Xc_tags_from_ascii: Finished reading the file {0}\n'.format(filename))

  return dic 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Xc_tags_from_h5(filename):
  """
  This routine reads in the Xc tags from an HDF5 file, and has a similar functionality as the function
  Xc_tags_from_ascii(). Please refer to the documentation there for further details.
  """
  if not os.path.exists(filename):
    logger.error('Xc_tags_from_h5: The HDF5 file "{0}" does not exist'.format(filename))
    sys.exit(1)

  with h5py.File(filename, 'r') as h5: dset = h5['Xc_tags'].value
  m, n   = dset.shape
  dtype  = dset.dtype

  logger.info('Xc_tags_from_h5: Done. Dataset has {0} rows, {1} columns, and {2} elements'.format(
              m, n, dset.size))

  attrs  = []
  tags   = []
  dic    = dict()
  dec_M  = 3
  dec_fov= 3
  dec_Z  = 3
  dec_logD = 2
  dec_Xc = 4

  M_ini  = np.round(dset[:,0], dec_M)  
  fov    = np.round(dset[:,1], dec_fov)
  Z      = np.round(dset[:,2], dec_Z)
  logD   = np.round(dset[:,3], dec_logD)
  Xc     = np.round(dset[:,4], dec_Xc)
  tags   = np.int16(dset[:,5])

  n, m   = dset.shape
  attrs  = ['{0:0.3f},{1:0.3f},{2:0.3f},{3:0.2f},{4:0.4f}'.format(
             M_ini[k], fov[k], Z[k], logD[k], Xc[k]) for k in range(n)]
  tags   = [tags[k] for k in range(n)]

  for key, val in zip(attrs, tags): dic[key] = val

  return dic

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


