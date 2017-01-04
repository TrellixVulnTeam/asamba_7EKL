
import sys, os, glob
import logging
import numpy as np 

import var_def 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_mass_directories(dir_repos):
  """
  Return the list of directories labelled with track initial masses residing in the repository path.
  E.g. the directories have names like "dir_repos/M01.234", "dir_repos/M56.789", and so on.
  @param dir_repos: Full path to the directory where the grid is stored, e.g. 
         /home/user/projects/asamba-grid
  @type dir_repos: string
  @result: list of full paths to the mass directories
  @rtype: list of strings
  """
  dirs   = sorted(glob.glob(dir_repos + 'M*'))
  n_dirs = len(dirs)
  if n_dirs == 0:
    logging.error('insert_lib: get_mass_directories: Found no mass directory in {0}'.format(dir_repos))

  return dirs 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_track_parameters(tracks, dir_repos):
  """
  Glob and find all available tracks that are organized inside the repository (hence dir_repos).
  The tracks are organized based on their initial mass, and lie inside the "hist" subdirectory, e.g.
  "dir_repos/M12.345/hist/M12.345-ov0.010-Z0.018-logD01.23.hist"
  Then, the whole track parameters will be stored into the "tracks" object
  
  @param tracks: an instance of the var_def.tracks() object
  @type tracks: class object
  @param dir_repos: full path to the repository
  @type dir_repos: string
  """
  mass_dirs = get_mass_directories(dir_repos)

  list_track_paths = []

  # Collect all available hist files
  for dr in mass_dirs:
    hist_search = dr + '/hist/M*.hist'
    hists   = glob.glob(hist_search)
    n_hists = len(hists)
    if n_hists == 0:
      logging.error('insert_lib: no history files found in the path: "{0}"'.format(hist_search))
    list_track_paths += hists

  # Extract parameters from history file paths
  # Store info the class objects
  list_dic_tracks  = []
  n_tracks = len(list_track_paths)

  for i, trck in enumerate(list_track_paths):
    ind_slash = trck.rfind('/')
    ind_point = trck.rfind('.hist')
    trck      = trck[ind_slash+1 : ind_point]
    params    = trck.split('-')
    n_params  = len(params)
    if n_params != 4:
      logging.error('insert_lib: the number of retrieved parameters is different from expected')

    M_ini     = float(params[0][1:])
    fov       = float(params[1][2:])
    Z         = float(params[2][1:])
    logD      = float(params[3][4:])

    track     = var_def.track(M_ini=M_ini, Z=Z, fov=fov, logD=logD)
    list_dic_tracks.append(track)

  # Store the data into the "tracks" object
  tracks.set_n_tracks = n_tracks
  tracks.set_list_dic_tracks = list_dic_tracks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def track_parameters_to_ascii(dir_repos, ascii_out):
  return None
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
