
"""
This module provides auxilary functinalities to work with the grid data, in reading, writng and 
manipulating the grid data (tracks, models, modes, etc) into a proper format. 
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
def get_hist_and_gyre_in_data(self_tracks):
  """

  """
  st = self_tracks
  dir_repos = st.dir_repos
  n_dirs_M_ini = st.n_dirs_M_ini
  list_dirs_M_ini = st.list_dirs_M_ini
  n_tracks = st.n_tracks
  list_tracks = st.list_tracks

  if n_tracks == 0:
    logger.error('get_hist_and_gyre_in_data: the "tracks" object has no tracks strored in it')

  for i, track in enumerate(list_tracks):
    # locate and read the history file
    hist_file = track.filename
    if not os.path.exists(hist_file):
      logger.error('get_hist_and_gyre_in_data: "{0}" does not exist'.format(hist_file))

    try:
      header, hist = read.read_mesa_ascii(hist_file)
    except:
      logger.error('get_hist_and_gyre_in_data: read_mesa_ascii failed to read "{0}"'.format(hist_file))

    # convert hist path to gyre_in search string
    gyre_in_search_pattern = get_gyre_in_search_pattern_from_hist(hist_file)

    # instantiate models
    models = var_def.models(dir_repos=dir_repos)
    models.set_model_search_pattern(gyre_in_search_pattern)

    # get available gyre_in files associated with this track
    models.find_list_filenames()

    if models.n_models == 0:
      logger.error('get_hist_and_gyre_in_data: Found no gyre_in model for this track!')

    arr_model_numbers = np.array([ model.model_number for model in models ])
    print arr_model_numbers

    sys.exit()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_gyre_in_search_pattern_from_hist(filename):
  """
  From the full path to the MESA history file, generate a search string for globbing GYRE input files.
  This function replaces the "/hist/" in the input filename with "/gyre_in/", and also replaces the 
  hist suffix e.g. ".hist" with "*".

  @param filename: full path to the history filename. 
  @type filename: string
  @return: regular expression for explicitly searching for gyre_in files that are linked to a specific
        track
  @rtype: string
  """
  if '/hist/' not in filename:
    logger.error('get_gyre_in_search_pattern_from_hist: "/hist/" not in the filename path')

  srch = filename.replace('/hist/', '/gyre_in/')
  ind  = srch.rfind('.')
  srch = srch[:ind] + '*'

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
    hdr, hist = read.read_mesa_ascii(histname)

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
# R O U T I N E S   F O R   T R A C K S   O B J E C T S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
