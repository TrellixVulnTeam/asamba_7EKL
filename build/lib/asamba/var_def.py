
"""
This module provides the track and tracks class objects and some basic functionalities. The "tracks"
is build based on the "track" object.
"""
from __future__ import unicode_literals
from builtins import object

import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class track(object):
  """
  Class object that stores the data for MESA tracks
  """
  def __init__(self, M_ini=-1.0, Z=-1.0, fov=-1.0, logD=-1.0):
    """
    Constructor that stores the mass, metalicity, overshoot and extra diffusive mixing per each 
    track. By default, the attributes are set to "-1.0" as a physically meaningless initial value,
    to facilitate more convenient capturing of wrong initializations. E.g. you can initialize a track
    as:

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

    self.filename = ''

  # ...................................
  def __exit__(self, type, value, traceback):
    pass

  # ...................................
  def set_M_ini(self, M_ini):
    self.M_ini = M_ini

  # ...................................
  def set_Z(self, Z):
    self.Z = Z

  # ...................................
  def set_fov(self, fov):
    self.fov = fov

  # ...................................
  def set_logD(self, logD):
    self.logD = logD

  # ...................................
  def set_filename(self, filename):
    self.filename = filename

  # ...................................
  def get_M_ini(self):
    return self.M_ini

  # ...................................
  def get_Z(self):
    return self.Z

  # ...................................
  def get_fov(self):
    return self.fov 

  # ...................................
  def get_logD(self):
    return self.logD 

  # ...................................
  def get_filename(self):
    return self.filename

  # ...................................
  def get_attr_as_dic(self):
    """
    Convert the attributes of the "track" object into a dictionary, with the attribute names as keys
    """
    dic = dict()
    for attr in dir(self):
      dic[attr] = getattr(self, attr)

    return dic

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class tracks(object):
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

    self.mass_search_pattern = ''

    self.hist_search_pattern = ''
    self.hist_extension = '.hist'

    # extra attributes
    self.n_dirs_M_ini = 0
    self.list_dirs_M_ini = []
    self.n_tracks = 0
    self.list_tracks = []

  # ...................................
  def __enter__(self):
    return self 

  # ...................................
  def __exit__(self, type, value, traceback):
    pass

  # ...................................
  # Setters
  def set_dir_repos(self, dir_repos):
    self.dir_repos = dir_repos

  # ...................................
  def set_mass_search_pattern(self, mass_search_pattern):
    self.mass_search_pattern = mass_search_pattern

  # ...................................
  def set_hist_search_pattern(self, hist_search_pattern):
    self.hist_search_pattern = hist_search_pattern

  # ...................................
  def set_hist_extension(self, hist_extension):
    self.hist_extension = hist_extension

  # ...................................
  def set_n_dirs_M_ini(self, n_dirs_M_ini):
    self.n_dirs_M_ini = n_dirs_M_ini

  # ...................................
  def set_list_dirs_M_ini(self, list_dirs_M_ini):
    self.list_dirs_M_ini = list_dirs_M_ini

  # ...................................
  def set_n_tracks(self, n_tracks):
    self.n_tracks = n_tracks

  # ...................................
  def set_list_tracks(self, list_tracks):
    self.list_tracks = list_tracks

  # ...................................
  def set_mass_directories(self):
    """
    Return the list of directories labelled with track initial masses residing in the repository path.
    E.g. the directories have names like "dir_repos/M01.234", "dir_repos/M56.789", and so on.
    @result: list of full paths to the mass directories
    @rtype: list of strings
    """
    dir_repos = self.get_dir_repos()
    mass_search_pattern = self.mass_search_pattern
    if mass_search_pattern == '':
      logger.error('set_mass_directories: first set the attribute "mass_search_pattern"')
      sys.exit(1)

    dirs   = sorted( glob.glob(dir_repos + mass_search_pattern) )
    n_dirs = len(dirs)
    if n_dirs == 0:
      logger.error('var_def: get_mass_directories: Found no mass directory in {0}'.format(dir_repos))
      sys.exit(1)

    self.set_n_dirs_M_ini(n_dirs)
    self.set_list_dirs_M_ini(dirs)
    logger.info('set_mass_directories: Found "{0}" directories with "{1}"'.format(
           n_dirs, mass_search_pattern))

  # ...................................
  def set_track_parameters(self):
    """
    Glob and find all available tracks that are organized inside the repository (hence dir_repos).
    The tracks are organized based on their initial mass, and lie inside the "hist" subdirectory, e.g.
    "dir_repos/M12.345/hist/M12.345-ov0.010-Z0.018-logD01.23.hist"
    Then, the whole track parameters will be stored into the "tracks" object
    
    @param self: an instance of the var_def.tracks() object
    @type self: class object
    """
    if self.n_dirs_M_ini == 0:
      logger.error('set_track_parameters: first call set_mass_directories()')
      sys.exit(1)

    list_dirs_M_ini = self.get_list_dirs_M_ini()

    list_track_paths = []

    hist_search_pattern = self.get_hist_search_pattern()
    if hist_search_pattern == '':
      logger.error('set_track_parameters: first call set_hist_search_pattern()')
      sys.exit(1)

    # Collect all available hist files
    for dr in list_dirs_M_ini:
      hist_search = dr + hist_search_pattern 
      hists   = glob.glob(hist_search)
      n_hists = len(hists)
      if n_hists == 0:
        logger.error('var_def: no history files found in the path: "{0}"'.format(hist_search))
        sys.exit(1)
      list_track_paths += hists[:]

    # Extract parameters from history file paths
    # Store info the class objects
    list_tracks  = []
    n_tracks = len(list_track_paths)
    hist_extension = self.hist_extension

    for i, trck in enumerate(list_track_paths):
      ind_slash = trck.rfind('/')
      ind_point = trck.rfind(hist_extension)
      corename  = trck[ind_slash+1 : ind_point]
      params    = corename.split('-')
      n_params  = len(params)
      if n_params != 4:
        logger.error('var_def: the number of retrieved parameters is different from expected')
        sys.exit(1)

      M_ini     = float(params[0][1:])
      fov       = float(params[1][2:])
      Z         = float(params[2][1:])
      logD      = float(params[3][4:])

      one_track = track(M_ini=M_ini, Z=Z, fov=fov, logD=logD)
      one_track.set_filename(trck)
      list_tracks.append(one_track)

    # Store the data into the "tracks" object
    self.set_n_tracks(n_tracks)
    self.set_list_tracks(list_tracks)
    logger.info('set_track_parameters: Setting track numbers and list done.')

  # ...................................
  # Getters
  def get_dir_repos(self):
    return self.dir_repos

  # ...................................
  def get_mass_search_pattern(self):
    return self.mass_search_pattern

  # ...................................
  def get_hist_search_pattern(self):
    return self.hist_search_pattern

  # ...................................
  def get_hist_extension(self):
    return self.hist_extension

  # ...................................
  def get_n_dirs_M_ini(self):
    return self.n_dirs_M_ini

  # ...................................
  def get_list_dirs_M_ini(self):
    return self.list_dirs_M_ini

  # ...................................
  def get_n_tracks(self):
    return self.n_tracks

  # ...................................
  def get_list_tracks(self):
    return self.list_tracks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class model(object):
  """
  The class that encapsulates the properties of each of MESA output model files which serve as inputs
  to GYRE.
  """
  def __init__(self):
    """
    constructor of the class
    """
    self.filename          = ''
    # self.track             = track(-1.0, -1.0, -1.0, -1.0)

    self.M_ini             = 0.
    self.fov               = 0. 
    self.Z                 = 0. 
    self.logD              = 0. 
    self.Xc                = 0. 
    self.model_number      = 0

    self.star_mass         = 0.
    self.radius            = 0.
    self.log_Teff          = 0.
    self.log_g             = 0.
    self.log_L             = 0.
    self.log_Ledd          = 0.
    self.log_abs_mdot      = 0.
    self.mass_conv_core    = 0.

    self.star_age          = 0.
    self.dynamic_timescale = 0.
    self.kh_timescale      = 0.
    self.nuc_timescale     = 0.

    self.log_center_T      = 0. 
    self.log_center_Rho    = 0. 
    self.log_center_P      = 0. 
 
    self.center_h1         = 0.
    self.center_h2         = 0.
    self.center_he3        = 0.
    self.center_he4        = 0.
    self.center_c12        = 0.
    self.center_c13        = 0.
    self.center_n14        = 0.
    self.center_n15        = 0.
    self.center_o16        = 0.
    self.center_o18        = 0.
    self.center_ne20       = 0.
    self.center_ne22       = 0.
    self.center_mg24       = 0.

    self.surface_h1        = 0.
    self.surface_h2        = 0.
    self.surface_he3       = 0.
    self.surface_he4       = 0.
    self.surface_c12       = 0.
    self.surface_c13       = 0.
    self.surface_n14       = 0.
    self.surface_n15       = 0.
    self.surface_o16       = 0.
    self.surface_o18       = 0.
    self.surface_ne20      = 0.
    self.surface_ne22      = 0.
    self.surface_mg24      = 0.

    self.delta_nu          = 0.
    self.nu_max            = 0.
    self.acoustic_cutoff   = 0.
    self.delta_Pg          = 0.

    self.Mbol              = 0.
    self.bcv               = 0.
    self.U_B               = 0.
    self.B_V               = 0.
    self.V_R               = 0.
    self.V_I               = 0.
    self.V_K               = 0.
    self.R_I               = 0.
    self.I_K               = 0.
    self.J_H               = 0.
    self.H_K               = 0.
    self.K_L               = 0.
    self.J_K               = 0.
    self.J_L               = 0.
    self.J_Lp              = 0.
    self.K_M               = 0.

  # ...................................
  def __enter__(self):
    return self 

  # ...................................
  def __exit__(self, type, value, traceback):
    pass

  # ...................................
  # Setters
  # ...................................
  def set_filename(self, filename):
    self.filename = filename

  # def set_track(self, track):
  #   self.track = track 
    
  # setter (by dictionary) for the rest of the class attribute
  # ...................................
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
    items = list(dic.items())
    n_items = len(items)
    if n_items == 0:
      logger.error('model: set_by_dic: The input dictionary has no items inside.')
      sys.exit(1)

    for item in items:
      key = item[0]
      val = item[1]
      if not hasattr(self, key): #key not in avail:
        logger.error('model: set_by_dic: Non-standard key="{0}" cannot be set to the class'.format(key))
        sys.exit(1)
      setattr(self, key, value)

  # ...................................
  def set(self, attr, val):
    """
    Set the value of the specific attribute "attr" of the model object
    """
    if not hasattr(self, attr):
      logger.error('model: set: The attribute "{0}" is undefined'.format(attr))
      sys.exit(1)
    setattr(self, attr, val)
    
  # ...................................
  # Getter
  # ...................................
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
      sys.exit(1)

    return getattr(self, attr)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class models(object):
  """
  An agglomeration (container) of the objects from the "model" class
  """
  def __init__(self, dir_repos):
    """
    Constructor of the class. It can be instantiated by specifying the full path to the repository. 
    E.g.

    >>>many_models = var_def.models('/home/user/projects/mygrid')

    """
    if dir_repos[-1] != '/': dir_repos += '/'
    self.dir_repos = dir_repos

    self.model_search_pattern = ''
    self.model_extension = '.gyre'

    self.n_models = 0
    self.list_filenames = []

    self.list_models = []

  # ...................................
  def __enter__(self):
    return self 

  # ...................................
  def __exit__(self, type, value, traceback):
    pass

  # ...................................
  # Setters
  # ...................................
  def set_model_search_pattern(self, model_search_pattern):
    self.model_search_pattern = model_search_pattern

  # ...................................
  def set_model_extension(self, model_extension):
    self.model_extension = model_extension

  # ...................................
  def set_n_models(self, n_models):
    self.n_models = n_models

  # ...................................
  def set_list_filenames(self, list_filenames):
    self.list_filenames = list_filenames

  # ...................................
  def set_list_models(self, list_models):
    self.list_models = list_models

  # ...................................
  def find_list_filenames(self):
    """
    Find present models on the disk that match the search pattern.

    @param dir_repos: the full path to the repository where the files sit, e.g. '/home/user/mygrid'
    @type dir_repos: string
    @param model_search_pattern: the search pattern for globbing the available model files. e.g.
          'M*/gyre_in/*'
    @type model_search_pattern: string
    """
    dir_repos = self.dir_repos
    model_search_pattern = self.model_search_pattern

    if not os.path.exists(dir_repos):
      logger.error('find_list_filenames: "{0}" does not exist'.format(dir_repos))
      sys.exit(1)

    if model_search_pattern == '':
      logger.error('find_list_filenames: attribute "model_search_pattern" not set yet.')
      sys.exit(1)

    model_search   = dir_repos + model_search_pattern
    list_filenames = glob.glob(model_search)
    n_files = len(list_filenames)
    if n_files == 0:
      logger.error('find_list_filenames: found no model files in "{0}"'.format(model_search))
      sys.exit(1)

    self.set_n_models(n_files)
    self.set_list_filenames(list_filenames)
    logger.info('find_list_filenames: found "{0}" model files'.format(n_files))

  # ...................................
  def sort_list_filenames(self):
    # filenames  = self.get_list_filenames()
    # n_files    = len(files)
    if self.get_n_models() == 0:
      logger.error('sort_list_filenames: list of filenames is empty. Call find_list_filenames()')
      sys.exit(1)
    self.list_filenames.sort()

  # ...................................
  # Getters
  # ...................................
  def get_model_search_pattern(self):
    return self.model_search_pattern

  # ...................................
  def get_model_extension(self):
    return self.model_extension

  # ...................................
  def get_list_filenames(self):
    return self.list_filenames

  # ...................................
  def get_list_models(self):
    return self.list_models

  # ...................................
  def get_n_models(self):
    return self.n_models

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class modes(object):
  """
  This is a light-weight container for the GYRE outputs. All attriutes corresponding to the summary
  files or the eigenfunction files are available. The fields are initiated to "None" to keep the default
  volume of the objects minimal.
  """

  def __init__(self):
    """
    Constructor of the class. The attributes are based on the GYRE v.4.4, and the list of attributes 
    are availble here: <https://bitbucket.org/rhdtownsend/gyre/wiki/Output%20Files%20(4.4)>
    The data types after allocation is either integer, float, string
    """
    # Attributes of the summary files (eigenfrequency list)
    self.l            = None
    self.l_0          = None
    self.m            = None
    self.n_pg         = None
    self.n_p          = None
    self.n_g          = None
    self.omega        = None
    self.omega_int    = None
    self.freq         = None
    self.freq_units   = ''
    self.f_T          = None
    self.f_g          = None
    self.psi_T        = None
    self.psi_g        = None
    self.beta         = None
    self.E            = None
    self.E_norm       = None
    self.W            = None
    self.M_star       = None
    self.R_star       = None
    self.L_star       = None
    self.n_poly       = None
    self.W_eps        = None

    # Attributes of the mode files (eigenfunction files)
    self.n            = None
    self.x            = None
    self.V            = None
    self.As           = None
    self.U            = None
    self.c_1          = None
    self.Gamma_1      = None
    self.nabla_ad     = None
    self.delta        = None
    self.Omega_rot    = None
    self.xi_r         = None
    self.xi_h         = None
    self.Yt_1         = None
    self.Yt_2         = None
    self.eul_phi      = None
    self.deul_phi     = None
    self.eul_p        = None
    self.eul_rho      = None
    self.eul_T        = None
    self.lag_S        = None
    self.lag_L        = None
    self.lag_p        = None
    self.lag_rho      = None
    self.lag_T        = None
    self.dE_dx        = None
    self.dW_eps_dx    = None
    self.dW_dx        = None
    self.prop_type    = None
    self.K            = None
    self.M_r          = None
    self.p            = None
    self.rho          = None
    self.T            = None
    self.F_j          = None
    self.div_F_j      = None

    # my customized attributes
    self.freq_rot     = None
    self.freq_crit    = None
    self.eta_rot      = None
    self.omega_rot    = None

    # Other attributes
    self.label        = None

  # ...................................
  def __enter__(self):
    return self 

  # ...................................
  def __exit__(self, type, value, traceback):
    pass

  # ...................................
  # Setters
  # ...................................
  def set(self, attr, val):
    """
    Set the value of an attribute
    """
    if not hasattr(self, attr):
      logger.error('modes.set(): Attribute "{0}" not available'.format(attr))
      sys.exit(1)
    setattr(self, attr, val)

  # ...................................
  def set_by_dic(self, dic):
    """
    Set the attributes of the object through the available items (key, values) in the passed 
    dictionary. This is to minimize the number of necessary calls to the "set()" method.

    @param self: An instance of the modes class
    @type self: object
    @param dic: A dictionary containing the contents of the GYRE output files
    @type dic: dictionary
    """
    items = list(dic.items())
    n_items = len(items)
    if n_items == 0:
      logger.error('modes: set_by_dic: The input dictionary has no items inside.')
      sys.exit(1)

    for item in items:
      key = item[0]
      val = item[1]
      self.set(key, val)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
