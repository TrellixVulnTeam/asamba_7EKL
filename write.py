
"""
This module provides some functionalities for writing a variety of data on disk as e.g. 
ASCII files. The functions typically receive an instance of data classes defined in var_def
module, and exploit the information in the object. It is strongly recommended to read and 
understand the meaning of different classes, their attributes, and methods.
"""

import sys, os, glob
import logging
import numpy as np 

from grid import var_def, var_lib, read

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_model_parameters_to_ascii(self_tracks, ascii_out):
  """
  This routine receives an instance of the var_def.tracks object, which already has the following 
  attributes propertly assigned:
  - n_dirs_M_ini
  - list_dirs_M_ini
  - n_tracks
  - list_tracks
  Then, we march over the history filenames in the "list_tracks", and read the corresponding history
  file. After that, we construct a globbing search regex to find the full path to all GYRE model 
  files corresponding to that track. Based on the model_number tag on the GYRE output filenames, the
  corresponding line in the MESA history file is selected, and the whole necessary information (from
  the var_def.model object) is retrieved (thanks to the matching MESA column name and modde attribute 
  names). Then, for each GYRE model, the whole parameters are written iteratively to an ASCII file.
  An example of how to use this routine is:

  >>> tracks = var_def.tracks('/home/user/projects/mygrid')
  >>> tracks.set_mass_search_pattern('M*')
  >>> tracks.set_hist_search_pattern('/hist/*.hist')
  >>> tracks.set_hist_extension('.hist')
  >>> tracks.set_mass_directories()
  >>> tracks.set_track_parameters()
  >>> write.write_model_parameters_to_ascii(tracks, 'grid_data/models_parameters.txt')

  @param self_tracks: an instance of the var_def.tracks class.
  @type self_tracks: object
  @param ascii_out: full path to the output ascii file
  @type ascii_out: string
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
    logger.error('write_model_parameters_to_ascii: the "tracks" object has no tracks strored in it')
    sys.exit(1)
  
  # make a list of attributes in the "model" object
  a_model     = var_def.model()
  model_attrs = dir(a_model)
  exclude     = ['__doc__', '__init__', '__enter__', '__exit__', '__del__', '__module__', 
                 'filename', 'track', 'set_by_dic', 
                 'set_filename', 'set_track', 'get']
  model_attrs = [attr for attr in model_attrs if attr not in exclude]
  exclude     = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number'] # treated manually below
  other_attrs = [attr for attr in model_attrs if attr not in exclude]
  color_attrs = set(['U_B', 'B_V', 'V_R', 'V_I', 'V_K', 'R_I', 'I_K', 'J_H', 'H_K', 'K_L', 'J_K',
                     'J_L', 'J_Lp', 'K_M'])

  # open the file handler
  if os.path.exists(ascii_out): os.unlink(ascii_out)
  file_handle = open(ascii_out, 'a')

  # collect the header for all columns
  header      = '{0:>6s} {1:>5s} {2:>5s} {3:>5s} {4:>6s} {5:>5s} '.format(
                 'M_ini', 'fov', 'Z', 'logD', 'Xc', 'num')
  for attr in other_attrs:
    header    += '{0:>12s} '.format(attr[:12])
  header      += '\n'

  # write the header
  file_handle.write(header)

  n_models    = 0

  # iterate on all tracks and collect their corresponding models
  for i, track in enumerate(list_tracks):
    # locate and read the history file
    hist_file = track.filename
    if not os.path.exists(hist_file):
      logger.error('write_model_parameters_to_ascii: "{0}" does not exist'.format(hist_file))
      sys.exit(1)

    # instantiate a track from filename parameters
    tup_hist_par   = var_lib.get_track_parameters_from_hist_filename(hist_file)
    M_ini          = tup_hist_par[0]
    fov            = tup_hist_par[1]
    Z              = tup_hist_par[2]
    logD           = tup_hist_par[3]

    try:
      header, hist = read.read_mesa_ascii(hist_file)
    except:
      logger.error('write_model_parameters_to_ascii: read_mesa_ascii failed to read "{0}"'.format(hist_file))
      sys.exit(1)

    # convert hist path to gyre_in search string
    gyre_in_search_pattern = var_lib.get_gyre_in_search_pattern_from_hist(dir_repos, hist_file)

    # instantiate models of this track, and exploit it
    with var_def.models(dir_repos=dir_repos) as models:
      models.set_model_search_pattern(gyre_in_search_pattern)

      # get available gyre_in files associated with this track
      models.find_list_filenames()
      list_gyre_in_filenames = models.get_list_filenames()
      n_models   += models.get_n_models()

      if n_models == 0:
        logger.error('write_model_parameters_to_ascii: Found no gyre_in model for this track!')
        sys.exit(1)

      hist_model_numbers= hist['model_number']

      list_rows         = []
      for k, gyre_in_filename in enumerate(list_gyre_in_filenames):

        # get attributes from gyre_in filename
        tup_gyre_in_par  = var_lib.get_model_parameters_from_gyre_in_filename(gyre_in_filename)
        M_ini            = tup_gyre_in_par[0]
        fov              = tup_gyre_in_par[1]
        Z                = tup_gyre_in_par[2]
        logD             = tup_gyre_in_par[3]
        evol_state       = tup_gyre_in_par[4]
        Xc               = tup_gyre_in_par[5]
        model_number     = tup_gyre_in_par[6]

        # set the rest of the attributes from the history row
        ind_row = np.where(model_number == hist_model_numbers)[0]
        row     = hist[ind_row]
        
        # collect the basic attributes from the filename
        line      = '{0:>06.3f} {1:>05.3f} {2:>05.3f} {3:>05.2f} {4:>06.4f} {5:>05d} '.format(
                     M_ini, fov, Z, logD, Xc, model_number)
        
        # iterate over the rest of the attributes, and convert them to string
        for k, attr in enumerate(other_attrs):
          key     = attr 
          if key in color_attrs:
            key   = key.replace('_', '-') 
          # print k, key, row[key][0], type(row), type(row[key][0])
          # sys.exit(0)
          line += '{0:>12.6e} '.format(row[key][0])
        line += '\n'
        file_handle.write(line)

  # log and release the file handle
  logger.info('write_model_parameters_to_ascii: "{0}" model parameters written to "{1}" \
              '.format(n_models, ascii_out))
  file_handle.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_and_return_model_parameters_to_ascii(self_models, ascii_out):
  """
  Collect, and write down the whole "track" and "model" data from the dist (repository path). The 
  produced ascii file has two purposes: 
  
  1- It stands on its own, to represent the important data per each stored model, 
  2. It can be later uesd to fill up the SQL database with the corresponding attributes in the "tracks"
     and "models" tables.
  
  @param self_models: an instance of the var_def.models class. Note that before passing this object,
         the following methods must have already been applied on it. So, one use example is the following:

         >>> dir_repos = '/home/user/my-grid'
         >>> tracks = var_def.tracks(dir_repos)
         >>> tracks.set_mass_search_pattern('M*')
         >>> tracks.set_hist_search_pattern('/hist/*.hist')
         >>> tracks.set_hist_extension('.hist')

         >>> tracks.set_mass_directories()
         >>> tracks.set_track_parameters()

         >>> list_models = var_lib.get_list_models_from_hist_and_gyre_in_files(tracks)
         >>> n_models    = len(list_models)
         >>> models      = var_def.models(dir_repos)
         >>> models.set_list_models(list_models)
         >>> models.set_n_models(n_models)

         >>> write.write_and_return_model_parameters_to_ascii(models, 'models-parameters.txt')

  @type self_models: class object
  @param ascii_out: full path to the ascii file to be written on disk
  @type ascii_out: string
  """
  sm = self_models
  n_models = sm.get_n_models()
  if n_models == 0:
    logger.error('write_and_return_model_parameters_to_ascii: the passed "models" object has no models inside')
    sys.exit(1)
  list_models = sm.get_list_models()

  # open the file handle
  if os.path.exists(ascii_out): os.unlink(ascii_out)
  try:
    handle    = open(ascii_out, 'a')
  except:
    logger.error('write_and_return_model_parameters_to_ascii: failed to open: "{0}"'.format(ascii_out))
    sys.exit(1)

  # filter the attributes
  first_model = list_models[0]
  avail_attrs = dir(first_model)
  exclude     = set(['__init__', '__doc__', '__module__', 'filename', 'track'])
  avail_attrs = [attr for attr in avail_attrs if attr not in exclude]
  avail_attrs = [attr for attr in avail_attrs if 'set' not in attr and 'get' not in attr]
  
  key_attrs   = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number']
  # key_fmt     = [float, float, float, float, str, float, int]
  other_attrs = [attr for attr in avail_attrs if attr not in key_attrs]

  # collect the header for all columns
  header      = '{0:>6s} {1:>5s} {2:>5s} {3:>5s} {4:>6s} {5:>5s} '.format(
                 'M_ini', 'fov', 'Z', 'logD', 'Xc', 'num')
  for attr in other_attrs:
    header    += '{0:>12s} '.format(attr[:12])
  header      += '\n'

  # write the header
  handle.write(header)

  # collect the line info as lines
  # lines       = [header]

  # iterate over models, and collect data into lines
  for i, model in enumerate(list_models):
    # first, the key attributes
    line      = '{0:>06.3f} {1:>05.3f} {2:>05.3f} {3:>05.2f} {4:>06.4f} {5:>05d} '.format(
                 model.M_ini, model.fov, model.Z, model.logD, model.Xc, model.model_number)
    
    # iterate over the rest of the attributes, and convert them to string
    for k, attr in enumerate(other_attrs): 
      line += '{0:>12.6e} '.format(getattr(model, attr)[0])
    line += '\n'

    # append to the ascii file, and to the output list
    handle.write(line)
    # lines.append(line)

  logger.info('write_and_return_model_parameters_to_ascii: saved "{0}"'.format(ascii_out))
  print ' - grid.write.write_and_return_model_parameters_to_ascii: saved "{0}"'.format(ascii_out)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_tracks_parameters_to_ascii(self_tracks, ascii_out):
  """
  Store the four parameters of the MESA tracks (mass, overshoot, metallicity and extra mixing) as
  an ascii file on the disk. To do so, the var_def.get_track_parameters() method must have already
  been applied on the var_def.tracks() class object. 
  The format of the stored file is the following: the parameters in each row correspond to one track.
  There will be four columns, separated by a single space, and they correspond to the initial mass
  (M_ini), core overshooting parameter (fov), metallicity (Z), and extra diffusive mixing (logD),
  respectively.

  @param self_tracks: an instance of the var_def.tracks()
  @type self_tracks: class object
  @param ascii_out: full path to store the track parameters.
  @type ascii_out: string
  """
  if self_tracks.n_tracks == 0:
    logger.error('write_tracks_parameters_to_ascii: No track data stored. Call get_track_parameters() first')
    sys.exit(1)

  # add a header
  lines       = ['{0:<6s} {1:<5s} {2:<5s} {3:<5s} \n'.format('M_ini', 'fov', 'Z', 'logD')]

  list_tracks = self_tracks.list_tracks
  for i, obj in enumerate(list_tracks):
    str_M_ini = '{0:06.3f}'.format(obj.M_ini)
    str_fov   = '{0:05.3f}'.format(obj.fov)
    str_Z     = '{0:05.3f}'.format(obj.Z)
    str_logD  = '{0:05.2f}'.format(obj.logD)
    line      = '{0} {1} {2} {3} \n'.format(str_M_ini, str_fov, str_Z, str_logD)
    lines.append(line)

  with open(ascii_out, 'w') as w: w.writelines(lines)
  logger.info('write_tracks_parameters_to_ascii saved {0}'.format(ascii_out))
  print ' - write: write_tracks_parameters_to_ascii saved {0}'.format(ascii_out)

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
