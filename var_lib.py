
"""
This module provides auxilary functinalities to work with the database, in reading, writng and 
manipulating the grid data (tracks, models, modes, etc) into a proper format. 
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys, os, glob
import logging
import numpy as np 

from asamba import read
from asamba import var_def 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E S   O B J E C T S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E L   O B J E C T S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_attrs():
  """
  Get the entire list of attributes for an instance of the var_def.model class. The attributes are 
  sorted in the way they are defined in the class (which might be different from the order in which 
  they are stored/allocated in memory!). The returned list is a concatenation of the two following 
  lists:
  - basic attributes, retrieved by calling get_model_basic_attrs()
  - other attributes, retrieved by calling get_model_other_attrs()

  @return: the full list of var_def.model attribute names 
  @rtype: list of strings
  """

  return get_model_basic_attrs() + get_model_other_attrs()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_basic_attrs():
  """
  Get the basic attributes of a var_def.model() object. These six attributes distinguish a MESA 
  model from another.

  @return: list of basic attribute names, i.e. 'M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number'
  @rtype: list of strings
  """
  return get_track_attrs() + ['Xc', 'model_number']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_other_attrs():
  """
  Get the other attribute names of the var_def.model() object. These attributes are retrieved from
  MESA history information, and are in the same order as they appear as history column names. This
  choice facilitates much faster conversion of history rows/columns and writing them down into an 
  ASCII file. The same is true when reading the file. For other purposes, the ordering is not 
  important.

  @return: list of other attribute names 
  @rtype: list of strings
  """
  str_attrs = [# fundamental parameters
               'star_mass', 'star_age', 'log_abs_mdot', 'mass_conv_core', 
               # timescales
               'dynamic_timescale', 'kh_timescale', 'nuc_timescale', 
               # global parameters
               'log_Teff', 'log_L', 'radius', 'log_g', 'log_Ledd',               
               # core conditions
               'log_center_T', 'log_center_Rho', 'log_center_P', 
               # core abundances
               'center_h1', 'center_h2', 'center_he3', 'center_he4', 'center_c12', 
               'center_c13', 'center_n14', 'center_n15', 'center_o16', 'center_o18', 
               'center_ne20', 'center_ne22', 'center_mg24', 
               # surface abundances
               'surface_h1', 'surface_h2', 'surface_he3', 'surface_he4', 'surface_c12',
               'surface_c13', 'surface_n14', 'surface_n15', 'surface_o16', 'surface_o18',
               'surface_ne20', 'surface_ne22', 'surface_mg24',
               # asteroseismic global/scaling parameters
               'delta_nu', 'nu_max', 'acoustic_cutoff', 'delta_Pg',
               # color indexes
               'Mbol', 'bcv', 'U_B', 'B_V', 'V_R', 'V_I', 'V_K', 'R_I', 'I_K', 'J_H',
               'H_K', 'K_L', 'J_K', 'J_L', 'J_Lp', 'K_M'
              ] 

  return str_attrs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_color_attrs():
  """
  Get the attributes of the model, corresponding to the color indixes.

  @return: list of color attribute names
  @rtype: list of string
  """
  clr_attrs = ['Mbol', 'bcv', 'U_B', 'B_V', 'V_R', 'V_I', 'V_K', 'R_I', 'I_K', 'J_H',
               'H_K', 'K_L', 'J_K', 'J_L', 'J_Lp', 'K_M']

  return clr_attrs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_list_models_from_hist_and_gyre_in_files(self_tracks):
  """
  Extract the data for all GYRE input models in the repository, using their associated line in the 
  MESA history file.
  """
  # collect necessary info
  st = self_tracks
  dir_repos = st.dir_repos
  n_dirs_M_ini = st.n_dirs_M_ini
  list_dirs_M_ini = st.list_dirs_M_ini
  n_tracks = st.n_tracks
  list_tracks = st.list_tracks

  if n_tracks == 0:
    logger.error('get_list_models_from_hist_and_gyre_in_files: the "tracks" object has no tracks strored in it')
    sys.exit(1)

  n_models    = 0
  list_models = []

  # make a list of attributes in the "model" object
  a_model     = var_def.model()
  model_attrs = dir(a_model)
  exclude     = ['__doc__', '__init__', '__module__', 'filename', 'track', 'set_by_dic', 
                 'set_filename', 'set_track', 'get']
  model_attrs = [attr for attr in model_attrs if attr not in exclude]
  exclude     = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number']
  other_attrs = [attr for attr in model_attrs if attr not in exclude]
  color_attrs = set(['U_B', 'B_V', 'V_R', 'V_I', 'V_K', 'R_I', 'I_K', 'J_H', 'H_K', 'K_L', 'J_K',
                     'J_L', 'J_Lp', 'K_M'])

  # iterate on all tracks and collect their corresponding models
  for i, track in enumerate(list_tracks):
    # locate and read the history file
    hist_file = track.filename
    if not os.path.exists(hist_file):
      logger.error('get_list_models_from_hist_and_gyre_in_files: "{0}" does not exist'.format(hist_file))
      sys.exit(1)

    # instantiate a track from filename parameters
    tup_hist_par   = get_track_parameters_from_hist_filename(hist_file)
    M_ini          = tup_hist_par[0]
    fov            = tup_hist_par[1]
    Z              = tup_hist_par[2]
    logD           = tup_hist_par[3]

    a_track        = var_def.track(M_ini=M_ini, fov=fov, Z=Z, logD=logD)

    try:
      header, hist = read.read_mesa_ascii(hist_file)
    except:
      logger.error('get_list_models_from_hist_and_gyre_in_files: read_mesa_ascii failed to read "{0}"'.format(hist_file))
      sys.exit(1)

    # convert hist path to gyre_in search string
    gyre_in_search_pattern = get_gyre_in_search_pattern_from_hist(dir_repos, hist_file)
    print(gyre_in_search_pattern)

    # instantiate models
    models = var_def.models(dir_repos=dir_repos)
    models.set_model_search_pattern(gyre_in_search_pattern)

    # get available gyre_in files associated with this track
    models.find_list_filenames()
    list_gyre_in_filenames = models.get_list_filenames()
    n_models   += models.get_n_models()
    if n_models == 0:
      logger.error('get_list_models_from_hist_and_gyre_in_files: Found no gyre_in model for this track!')
      sys.exit(1)

    # get a list of model numbers for all stored model associated with this track
    # arr_model_numbers = np.array([ get_model_number_from_gyre_in_filename(f) for f in list_gyre_in_filenames ])
    # arr_model_numbers = np.sort(arr_model_numbers)

    hist_model_numbers= hist['model_number']

    list_rows         = []
    for k, gyre_in_filename in enumerate(list_gyre_in_filenames):

      # instantiate a model
      a_model = var_def.model()
      a_model.set_filename(gyre_in_filename)
      a_model.set_track(track)

      # get attributes from gyre_in filename
      tup_gyre_in_par  = get_model_parameters_from_gyre_in_filename(gyre_in_filename)
      M_ini            = tup_gyre_in_par[0]
      fov              = tup_gyre_in_par[1]
      Z                = tup_gyre_in_par[2]
      logD             = tup_gyre_in_par[3]
      evol_state       = tup_gyre_in_par[4]
      Xc               = tup_gyre_in_par[5]
      model_number     = tup_gyre_in_par[6]

      # manually, insert the 6 above attributes to the model
      setattr(a_model, 'M_ini', M_ini)
      setattr(a_model, 'fov', fov)
      setattr(a_model, 'Z', Z)
      setattr(a_model, 'logD', logD)
      setattr(a_model, 'Xc', Xc)
      setattr(a_model, 'model_number', model_number)

      # set the rest of the attributes from the history row
      ind_row = np.where(model_number == hist_model_numbers)[0]
      row     = hist[ind_row]
      
      for attr in other_attrs: 
        key = attr
        if key in color_attrs: 
          key = key.replace('_', '-')
        setattr(a_model, attr, row[key])

      list_models.append(a_model)

  logger.info('get_list_models_from_hist_and_gyre_in_files: Returned a list of "{0}" model objects'.format(n_models))

  return list_models

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_parameters_from_gyre_out_filename(filename):
  """
  Extract all parameters in the MESA output/GYRE input file, and return them as a tuple. A random file
  may look like the following:
  /home/user/my_grid/M12.345/gyre_out/eta25.00/ad-sum-M12.345-ov0.012-Z0.014-logD02.50-MS-Xc0.3217-00312-eta25.00.h5

  @param filename: The full path to the GYRE output file
  @type filename: string
  @return: a tuple with 7 parameters of the model as float or integer values. The order of the 
           output is the following:
           - 0: M_ini, initial mass
           - 1: fov, exponential overshoot parameter
           - 2: Z, initial metallicity
           - 3: logD: the logarithm of the extra diffusive mixing
           - 4: Xc: the core hydrogen mass fraction
           - 5: model_number: integer, giving MESA step number
           - 6: eta: rotation rate in percentage w.r.t. the break up rotation rate
  @rtype: tuple
  """
  ind_slash = filename.rfind('/')
  ind_point = filename.rfind('.')
  corename  = filename[ind_slash+1 : ind_point].split('-')

  n_prefix  = 2
  pos_M     = n_prefix
  pos_fov   = pos_M + 1
  pos_Z     = pos_fov + 1
  pos_logD  = pos_Z + 1
  pos_evol  = pos_logD + 1
  pos_Xc    = pos_evol + 1
  pos_mod_n = pos_Xc + 1
  pos_eta   = pos_mod_n + 1

  print(pos_M)
  print(corename)
  sys.exit()

  M_ini     = float(corename[pos_M][1:])
  fov       = float(corename[pos_fov][2:])
  Z         = float(corename[pos_Z][1:])
  logD      = float(corename[pos_logD][4:])
  evol_state= corename[pos_evol]
  Xc        = float(corename[pos_Xc][2:])
  model_number  = int(corename[pos_mod_n])
  eta       = float(corename[pos_eta][3:])

  return (M_ini, fov, Z, logD, Xc, model_number, eta)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_track_parameters_from_hist_filename(filename):
  """
  Extract the whole parameters in the MESA history filename, and return them as a tuple. The hist 
  file can look like this:
  /home/user/my_grid/M12.345/hist/M12.345-ov0.012-Z0.014-logD02.50.hist
  whic corresponds to the following parameters:
  - M_ini    = 12.345 Msun
  - fov      = 0.012
  - Z        = 0.014
  - logD     = 2.50

  @param filename: full path to the input GYRE filename
  @type filename: string
  @return: tuple with the following items in the order: M_ini, fov, Z, logD
  @rtype: tuple
  """
  ind_slash = filename.rfind('/')
  ind_point = filename.rfind('.')
  corename  = filename[ind_slash+1 : ind_point].split('-')

  M_ini     = float(corename[0][1:])
  fov       = float(corename[1][2:])
  Z         = float(corename[2][1:])
  logD      = float(corename[3][4:])

  return (M_ini, fov, Z, logD)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_parameters_from_gyre_in_filename(filename):
  """
  Extract the whole parameters in the GYRE input filename, and return them as a tuple. The GYRE input
  file can look like this:
  /home/user/my_grid/M12.345/gyre_in/M12.345-ov0.012-Z0.014-logD02.50-MS-Xc0.5432-98765.gyre
  whic corresponds to the following parameters:
  - M_ini    = 12.345 Msun
  - fov      = 0.012
  - Z        = 0.014
  - logD     = 2.50
  - evol_stat = 'MS'
  - Xc       = 0.5432
  - model_number = 98765

  @param filename: full path to the input GYRE filename
  @type filename: string
  @return: tuple with the following items in the order: M_ini, fov, Z, logD, evol_state, Xc, model_number
  @rtype: tuple
  """
  ind_slash = filename.rfind('/')
  ind_point = filename.rfind('.')
  corename  = filename[ind_slash+1 : ind_point].split('-')

  M_ini     = float(corename[0][1:])
  fov       = float(corename[1][2:])
  Z         = float(corename[2][1:])
  logD      = float(corename[3][4:])
  evol_state= corename[4]
  Xc        = float(corename[5][2:])
  model_number  = int(corename[6])

  return (M_ini, fov, Z, logD, evol_state, Xc, model_number)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_model_number_from_gyre_in_filename(filename):
  """
  Extract the MESA evolution model number (when recording the file) from the GYRE input filename.
  E.g. the GYRE input file looks like the following:
  /home/user/my_grid/M01.400/gyre_in/M01.400-ov0.025-Z0.014-logD02.50-MS-Xc0.7075-00107.gyre
  The model number is the integer after the last dash "-" in the filename (easy to extract)

  @param filename: full path to the GYRE input filename
  @type filename: string
  @return: the model number of the file
  @rtype: int
  """
  ind_dash  = filename.rfind('-')
  ind_point = filename.rfind('.')
  str_mod_num = filename[ind_dash + 1 : ind_point]

  return int(str_mod_num)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_gyre_in_search_pattern_from_hist(dir_repos, filename):
  """
  From the full path to the MESA history file, generate a search string for globbing GYRE input files.
  This function replaces the "/hist/" in the input filename with "/gyre_in/", and also replaces the 
  hist suffix e.g. ".hist" with "*".

  @param dir_repos: the full path to the repository, where hist files are stored. Normally, this is 
         available from tracks.dir_repos
  @type dir_repos: string
  @param filename: full path to the history filename. 
  @type filename: string
  @return: regular expression for explicitly searching for gyre_in files that are linked to a specific
        track
  @rtype: string
  """
  if '/hist/' not in filename:
    logger.error('get_gyre_in_search_pattern_from_hist: "/hist/" not in the filename path')
    sys.exit(1)

  if dir_repos in filename:
    filename = filename.replace(dir_repos, '')
  srch = filename.replace('/hist/', '/gyre_in/')
  ind  = srch.rfind('.')
  srch = srch[:ind] + '*'

  return srch

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_histname_from_gyre_in(gyre_in_filename):
  """
  convert the full filename of the gyre_in file to a full path of the hist file, by following these 
  steps:
  1. substitute "gyre_in" with "hist"
  2. strip off the part of the filename after "logDxx.xx" 
  3. append '.hist' at the end of the file
  """
  f = gyre_in_filename
  f = f.replace('gyre_in', 'hist')
  ind_logD = f.rfind('logD')
  ind_keep = ind_logD + 4 + 5 # 4 for logD, 5 for the value
  f = f[:ind_keep] + '.hist'

  return f

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepare_models_data(self_models):
  """
  Obsolete: This routine is a no Go, when dealing with the entire database, because we immediately 
  run out of memory.

  This routine prepares the necessary data needed to fill up all required fields in the "model" objects.
  For that, we use the values from the history filenames, from GYRE input filename, and from the history
  columns, as soon as we match the model_number of the input model with that of the evolution step in
  the history file.
  Note: For large number of input GYRE files, this routine is extremely inefficient, because for every 
  input model, the history file is read one time. A better approach is provided by this routine:
  var_lib.get_hist_and_gyre_in_data().

  @param self_models: an instance of the "var_def.models" class 
  @type self_models: models object

  """
  sm = self_models

  sm.find_list_filenames()
  list_gyre_in = sm.get_list_filenames()

  # fetch "model" attribute names excluding default __doc__, __init__ and __module__
  a_model     = var_def.model()
  model_attrs = dir(a_model)
  exclude     = ['__doc__', '__init__', '__module__']
  model_attrs = [attr for attr in model_attrs if attr not in exclude]
  
  # Collect all models into a list of model objects
  list_models = []
  for i, filename in enumerate(list_gyre_in):
    # get an instance of the model class
    a_model   = var_def.model()
    a_model.set_filename(filename)

    # find the corresponding history file for this model
    histname  = gen_histname_from_gyre_in(filename)
    if not os.path.exists(histname):
      logger.error('prepare_models_data: missing the corresponding hist file {0}'.format(histname))
      sys.exit(1)
    hdr, hist = read.read_mesa_ascii(histname)

    tup_gyre_in_par = get_model_parameters_from_gyre_in_filename(filename)

    M_ini     = tup_gyre_in_par[0]
    fov       = tup_gyre_in_par[1]
    Z         = tup_gyre_in_par[2]
    logD      = tup_gyre_in_par[3]
    evol_state= tup_gyre_in_par[4]
    Xc        = tup_gyre_in_par[5]
    model_number = tup_gyre_in_par[6]

    # get the corresponding row for this model from the hist recarray
    ind_row   = model_number - 1
    if model_number == hist['model_number'][ind_row]:
      pass
    else:
      ind_row = np.where(hist['model_number'] == model_number)[0]
    row     = hist[ind_row]

    # manually, insert the 6 above attributes to the model
    setattr(a_model, 'M_ini', M_ini)
    setattr(a_model, 'fov', fov)
    setattr(a_model, 'Z', Z)
    setattr(a_model, 'logD', logD)
    setattr(a_model, 'Xc', Xc)
    setattr(a_model, 'model_number', model_number)

    for attr in model_attrs:
      if attr in ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number', 'filename', 'set_by_dic', 'track']:
        continue
      else:
        setattr(a_model, attr, row[attr])

    # generate a track object, and insert it into the model
    the_track = var_def.track(M_ini=M_ini, fov=fov, Z=Z, logD=logD)
    # setattr(a_model, 'track', the_track) 
    a_model.set_track(the_track)

    list_models.append(a_model)

  # store the list of model objects into the instance of the "models" class
  sm.set_list_models(list_models)

  logger.info('Done')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   T R A C K   O B J E C T S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_track_attrs():
  """
  Get the attribute names of the var_def.track object. These are the four basic attributes used to 
  define/distinguish an evolutionary track in MESA.

  @return: list of attribute names, i.e. 'M_ini', 'fov', 'Z', 'logD'
  @rtype: list of strings
  """
  return ['M_ini', 'fov', 'Z', 'logD']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
