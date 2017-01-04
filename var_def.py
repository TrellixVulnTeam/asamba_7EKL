
import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class track:
  """
  Class object that stores the data for MESA tracks
  """
  def __init__(self, M_ini, Z, fov, logD):
    """
    Constructor that stores the mass, metalicity, overshoot and extra diffusive mixing per each 
    track
    """
    self.M_ini = M_ini
    self.Z = Z
    self.fov = fov
    self.logD = logD

    self.hist_string = ''

  def set_M_ini(self, M_ini):
    self.M_ini = M_ini

  def set_Z(self, Z):
    self.Z = Z

  def set_fov(self, fov):
    self.fov = fov

  def set_logD(self, logD):
    self.logD = logD

  def set_hist_string(self, hist_string):
    self.hist_string = hist_string

  def get_M_ini(self):
    return self.M_ini

  def get_Z(self):
    return self.Z

  def get_fov(self):
    return self.fov 

  def get_logD(self):
    return self.logD 

  def get_hist_string(self):
    return self.hist_string

  def get_dic_track_parameters(self):
    return {'M_ini':self.M_ini,
            'Z':self.Z,
            'fov':self.fov,
            'logD':self.logD}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class tracks:
  """
  Class object that agglemerates multiple instances of the "track" object
  """
  def __init__(self, dir_repos):
    """
    @param dir_repos: Full path to the directory where the grid is stored, e.g. 
           /home/user/projects/asamba-grid
    @type dir_repos: string
    """
    # mandatory attribute
    if dir_repos[-1] != '/': dir_repos += '/'
    self.dir_repos = dir_repos

    # extra attributes
    self.n_dirs_M_ini = 0
    self.list_dirs_M_ini = []
    self.n_tracks = 0
    self.list_dic_tracks = []

  # Setters
  def set_dir_repos(self, dir_repos):
    self.dir_repos = dir_repos

  def set_n_dirs_M_ini(self, n_dirs_M_ini):
    self.n_dirs_M_ini = n_dirs_M_ini

  def set_list_dirs_M_ini(self, list_dirs_M_ini):
    self.list_dirs_M_ini = list_dirs_M_ini

  def set_n_tracks(self, n_tracks):
    self.n_tracks = n_tracks

  def set_list_dic_tracks(self, list_dic_tracks):
    self.list_dic_tracks = list_dic_tracks

  # Getters
  def get_dir_repos(self):
    return self.dir_repos

  def get_n_dirs_M_ini(self):
    return self.n_dirs_M_ini

  def get_list_dirs_M_ini(self):
    return self.list_dirs_M_ini

  def get_n_tracks(self):
    return self.n_tracks

  def get_list_dic_tracks(self):
    return self.list_dic_tracks

  def get_mass_directories(self):
    """
    Return the list of directories labelled with track initial masses residing in the repository path.
    E.g. the directories have names like "dir_repos/M01.234", "dir_repos/M56.789", and so on.
    @result: list of full paths to the mass directories
    @rtype: list of strings
    """
    dir_repos = self.get_dir_repos()
    dirs   = sorted(glob.glob(dir_repos + 'M*'))
    n_dirs = len(dirs)
    if n_dirs == 0:
      logging.error('var_def: get_mass_directories: Found no mass directory in {0}'.format(dir_repos))

    self.n_dirs_M_ini = n_dirs
    self.list_dirs_M_ini = dirs

  def get_track_parameters(self):
    """
    Glob and find all available tracks that are organized inside the repository (hence dir_repos).
    The tracks are organized based on their initial mass, and lie inside the "hist" subdirectory, e.g.
    "dir_repos/M12.345/hist/M12.345-ov0.010-Z0.018-logD01.23.hist"
    Then, the whole track parameters will be stored into the "tracks" object
    
    @param self: an instance of the var_def.tracks() object
    @type self: class object
    """
    list_dirs_M_ini = self.get_list_dirs_M_ini()

    list_track_paths = []

    # Collect all available hist files
    for dr in list_dirs_M_ini:
      hist_search = dr + '/hist/M*.hist'
      hists   = glob.glob(hist_search)
      n_hists = len(hists)
      if n_hists == 0:
        logging.error('var_def: no history files found in the path: "{0}"'.format(hist_search))
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
        logging.error('var_def: the number of retrieved parameters is different from expected')

      M_ini     = float(params[0][1:])
      fov       = float(params[1][2:])
      Z         = float(params[2][1:])
      logD      = float(params[3][4:])

      track     = track(M_ini=M_ini, Z=Z, fov=fov, logD=logD)
      list_dic_tracks.append(track)

    # Store the data into the "tracks" object
    self.set_n_tracks = n_tracks
    self.set_list_dic_tracks = list_dic_tracks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

