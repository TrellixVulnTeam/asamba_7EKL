
"""
This module provides the track and tracks class objects and some basic functionalities. The "tracks"
is build based on the "track" object.
"""

import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hist_search_pattern = '/hist/M*.hist'
hist_extension      = '.hist'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class track:
  """
  Class object that stores the data for MESA tracks
  """
  def __init__(self, M_ini, Z, fov, logD):
    """
    Constructor that stores the mass, metalicity, overshoot and extra diffusive mixing per each 
    track. E.g.

    >>>a_track = var_def.track(M_ini=12.0, Z=0.014, fov=0.024, logD=2.25)

    @param M_ini: the initial mass in solar unit
    @type M_ini: float
    @param Z: metallicity (where the solar metallicity from Asplund et al. 2009 is 0.014)
    @type Z: float
    @param fov: the exponential overshoot free parameter (see e.g. Eq. 2 in Moravveji et al.
           2016, ApJ)
    @type fov: float
    @param logD: the (logarithm of the) constant diffusive mixing in the radiative envelope.
           See e.g. Fig. 2a in Moravveji et al. (2016, ApJ)
    @type logD: float
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
    The constructor of the class. E.g.

    >>>some_tracks = var_def.tracks(dir_repos='/home/username/projects/mygrid')
    
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
    self.list_tracks = []

  # Setters
  def set_dir_repos(self, dir_repos):
    self.dir_repos = dir_repos

  def set_n_dirs_M_ini(self, n_dirs_M_ini):
    self.n_dirs_M_ini = n_dirs_M_ini

  def set_list_dirs_M_ini(self, list_dirs_M_ini):
    self.list_dirs_M_ini = list_dirs_M_ini

  def set_n_tracks(self, n_tracks):
    self.n_tracks = n_tracks

  def set_list_tracks(self, list_tracks):
    self.list_tracks = list_tracks

  # Getters
  def get_dir_repos(self):
    return self.dir_repos

  def get_n_dirs_M_ini(self):
    return self.n_dirs_M_ini

  def get_list_dirs_M_ini(self):
    return self.list_dirs_M_ini

  def get_n_tracks(self):
    return self.n_tracks

  def get_list_tracks(self):
    return self.list_tracks

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
      logger.error('var_def: get_mass_directories: Found no mass directory in {0}'.format(dir_repos))

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
      hist_search = dr + hist_search_pattern 
      hists   = glob.glob(hist_search)
      n_hists = len(hists)
      if n_hists == 0:
        logger.error('var_def: no history files found in the path: "{0}"'.format(hist_search))
      list_track_paths += hists[:]

    # Extract parameters from history file paths
    # Store info the class objects
    list_tracks  = []
    n_tracks = len(list_track_paths)

    for i, trck in enumerate(list_track_paths):
      ind_slash = trck.rfind('/')
      ind_point = trck.rfind(hist_extension)
      trck      = trck[ind_slash+1 : ind_point]
      params    = trck.split('-')
      n_params  = len(params)
      if n_params != 4:
        logger.error('var_def: the number of retrieved parameters is different from expected')

      M_ini     = float(params[0][1:])
      fov       = float(params[1][2:])
      Z         = float(params[2][1:])
      logD      = float(params[3][4:])

      one_track = track(M_ini=M_ini, Z=Z, fov=fov, logD=logD)
      list_tracks.append(one_track)

    # Store the data into the "tracks" object
    self.set_n_tracks(n_tracks)
    self.set_list_tracks(list_tracks)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class model:
  """
  The class that encapsulates the properties of each of MESA output model files which serve as inputs
  to GYRE.
  """
  def __init__(self, M_ini, fov, Z, logD, Xc, model_number):
    """
    constructor of the class
    """
    self.M_ini        = M_ini
    self.fov          = fov
    self.Z            = Z
    self.logD         = logD
    self.Xc           = Xc 
    self.model_number = model_number

    self.mass         = 0.
    self.radius       = 0.
    self.Teff         = 0.
    self.log_g        = 0.
    self.log_L        = 0.
    self.log_Ledd     = 0.
    self.log_mdot     = 0.
    self.mass_conv_core = 0.

    self.age          = 0.
    self.tau_dyn      = 0.
    self.tau_kh       = 0.
    self.tau_nuc      = 0.

    self.log_Tc       = 0.
    self.log_Rhoc     = 0.
    self.log_Pc       = 0.

    self.center_h1    = 0.
    self.center_h2    = 0.
    self.center_he3   = 0.
    self.center_he4   = 0.
    self.center_n14   = 0.
    self.center_n15   = 0.
    self.center_o16   = 0.
    self.center_o18   = 0.
    self.center_ne20  = 0.
    self.center_ne22  = 0.
    self.center_mg24  = 0.

    self.surface_h1   = 0.
    self.surface_h2   = 0.
    self.surface_he3  = 0.
    self.surface_he4  = 0.
    self.surface_c12  = 0.
    self.surface_c13  = 0.
    self.surface_n14  = 0.
    self.surface_n15  = 0.
    self.surface_o16  = 0.
    self.surface_o18  = 0.
    self.surface_ne20 = 0.
    self.surface_ne22 = 0.
    self.surface_mg24 = 0.

    self.delta_nu     = 0.
    self.nu_max       = 0.
    self.nu_cutoff    = 0.
    self.delta_P      = 0.

    self.M_bol        = 0.
    self.BC_V         = 0.
    self.U_B          = 0.
    self.B_V          = 0.
    self.V_R          = 0.
    self.V_I          = 0.
    self.V_K          = 0.
    self.R_I          = 0.
    self.I_K          = 0.
    self.J_H          = 0.
    self.H_K          = 0.
    self.K_L          = 0.
    self.J_K          = 0.
    self.J_L          = 0.
    self.J_Lp         = 0.
    self.K_M          = 0.

  # def setters for the most important attributes of the class
  def set_M_ini(self, M_ini):
    self.M_ini = M_ini

  def set_fov(self, fov):
    self.fov = fov 

  def set_Z(self, Z):
    self.Z = Z 

  def set_logD(self, logD):
    self.logD = logD

  def set_Xc(self, Xc):
    self.Xc = Xc 

  def set_model_number(self, model_number):
    self.model_number = model_number

  # setter (by dictionary) for the rest of the class attribute
  def set_by_dic(self, dic):
    """
    Since the "model" class has many attributes, instead of writing a setter for all 
    attributes manually (exhaustive), we pass the attribute values through a dictionary.
    This is a general-purpose interface to set the "canonical" attributes of the "model"
    class. E.g. 

    >>> a_model.set_by_dic({'Teff':10125.0, 'log_g':4.128, 'center_018':1.4509e-5})

    @param self: an instance of the model class
    @type self: object
    @param dic: a dictionary containing the attributes to be set in the model, e.g.
    @type dic: dict
    """
    # avail = dir(self)
    items = dic.items()
    n_items = len(items)
    if n_items == 0:
      logger.error('model: set_by_dic: The input dictionary has no items inside.')

    for item in items:
      key = item[0]
      val = item[1]
      if if not hasattr(self, key): #key not in avail:
        logger.error('model: set_by_dic: Non-standard key="{0}" cannot be set to the class'.format(key))
      setattr(self, key, value)

  # Getter
  def get(self, attr):
    """
    General-purpose method to get the value of a canonical attribute of the object
    E.g.

    >>>val = a_model.get('age')

    @param attr: the name of the available attribute of the class
    @type attr: string
    @return: the value of the attribute
    @rtype: float
    """
    if not hasattr(self, attr):
      logger.error('model: get: The attribute "{0}" is undefined'.format(attr))

    return getattr(self, attr)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
