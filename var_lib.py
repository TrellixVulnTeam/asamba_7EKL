
"""
This module provides auxilary functinalities to work with the grid data, in manipulating 
the grid data (tracks, models, modes, etc) into a proper format. This module complements
the basic data classes defined in var_def; thus, it is critical to understand the different
classes, attributes and methods in the var_def module. Most of the output of the functions 
are stored internally as the attributes of the passed object files. 
"""

import sys, os, glob
import logging
import numpy as np 

import read
import var_def 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   M O D E L S   O B J E C T S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_list_models_from_hist_and_gyre_in_files(self_tracks):
  """
  Extract the data for all GYRE input models in the repository, using their associated line in the 
  MESA history file.

  Turns out, this is a very memory exhaustive routine, and basically inappropriate to use on a single
  typical node. It is possible to easily exceed the available RAM.

  @param self_tracks: an instance of the var_def.tracks class.
  @param self_tracks: object
  @return: list of models, i.e. a list of instances of the var_def.model class, with one object per
         each GYRE model on the disk.
  @rtype: list of objects
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
    print gyre_in_search_pattern

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

    hist_model_numbers= hist['model_number']

    list_rows         = []
    for k, gyre_in_filename in enumerate(list_gyre_in_filenames):

      # instantiate a model
      a_model = var_def.model()
      a_model.set_filename(gyre_in_filename)
      # a_model.set_track(track)

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
    # the_track = var_def.track(M_ini=M_ini, fov=fov, Z=Z, logD=logD)
    # setattr(a_model, 'track', the_track) 
    # a_model.set_track(the_track)

    list_models.append(a_model)

  # store the list of model objects into the instance of the "models" class
  sm.set_list_models(list_models)

  logger.info('Done')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   T R A C K S   O B J E C T S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
