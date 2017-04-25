
"""
This module provides interpolation between frequencies over a give range of input 
stellar parameters. With this tool, it is no longer needed to compute too 
highly-resolved grids around the best asteroseismic models. Instead, the resolved 
models are prepared by the interpolation in between the grid points from the coarse model.

The interpolation class is a derived/subclass of the sampler.sampling class, because we require
several of the methods defined there in this module. This is to ensure we minimize/suppress 
the redundancy, and make the best runtime use of the parameters that are set therein.
"""

import sys, os, glob
import logging
import numpy as np
from scipy.interpolate import griddata

from grid import utils, db_def, db_lib, query, sampler

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    ###  ### ######  ###     ###   #####
    #    #    #    #  #     #  #       #   #     #
    #     #   #    #  #     #  #       #  #
    #    #    #    #  ######   #       #  #
    #####     #    #  #     #  #       #  #
    #         #    #  #     #  #    #  #   #     #
    #          ####   ######  ####### ###   #####


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class interpolation(sampler.sampling): # inheriting ...
  """
  The base class for internal interpolation means, which extends upon the functionalities in the 
  sampler.sampling class.

  If interp_... is True, then that parameter will be interpolated
  from interp_..._from to interp_..._to, in interp_..._steps number of 
  meshpoints, including the last point (i.e. interp_..._to).
  """
  def __init__(self):

    super(interpolation, self).__init__()

    # self.dbname = ''

    #.............................
    # The Sampled data         
    #.............................
    # # Matrix of features, similar to sampling.learning_x
    # self.original_x = []
    # # Matrix of frequencies, similar to sampling.learning_y
    # self.original_y = []
    # # Matrix of radial orders, similar to sampling.learning_radial_orders
    # self.original_n_pg = []
    # # Matrix of mode types, similar to sampling.learning_mode_types
    # self.original_mode_types = []

    #.............................
    # Anchor (or the best) model. One GYRE ouput model
    # already exists for the anchor model. 
    #.............................
    # Names of the parameters/features
    # similar to sampling.feature_names
    self.anchor_param_names = []
    # Values for the parameters of anchor model
    # similar to neural_net.marginal_features
    self.anchor_param_values = []
    # Frequencies corresponding to the anchor parameters,
    # coming from neural_net.MAP_frequencies
    self.anchor_frequencies = []
    # Radial orders corresponding to each anchor frequency
    self.anchor_radial_orders = []
    # Mode identification corresponding to each anchor
    # frequency, as defined in grid.sql schema file
    self.anchor_mode_types = []

    #.............................
    # How to collect inputs?
    #.............................
    # Query around the anchor model
    self.inputs_around_anchor = False
    # if True, then, specify n number of points to 
    # each side of the anchor model, thus ending 
    # "almost" with 2n+1 points, but not necessarily 
    self.inputs_around_anchor_M_ini_n = 0
    self.inputs_around_anchor_fov_n   = 0
    self.inputs_around_anchor_Z_n     = 0
    self.inputs_around_anchor_logD_n  = 0
    self.inputs_around_anchor_Xc_n    = 0
    self.inputs_around_anchor_eta_n   = 0

    # Query in a range of models
    self.inputs_by_range = False

    #.............................
    # Input features and frequencies
    # from the grid (actual GYRE outputs)
    # m: number of rows
    # n: number of features
    # K: number of frequencies per row
    #.............................
    # The ndarray of input features; shape: (m, n)
    self.input_features     = []
    # The ndarray of input frequencies; shape: (m, K)
    self.input_frequencies  = []

    #.............................
    # Specifications for interpolation
    #.............................
    # Parameter ranges, and stepsizes
    self.interp_M_ini       = False
    self.interp_M_ini_from  = 0
    self.interp_M_ini_to    = 0
    self.interp_M_ini_steps = 0

    self.interp_fov         = False
    self.interp_fov_from    = 0
    self.interp_fov_to      = 0
    self.interp_fov_steps   = 0

    self.interp_Z           = False
    self.interp_Z_from      = 0
    self.interp_Z_to        = 0
    self.interp_Z_steps     = 0

    self.interp_logD        = False
    self.interp_logD_from   = 0
    self.interp_logD_to     = 0
    self.interp_logD_steps  = 0

    self.interp_Xc          = False
    self.interp_Xc_from     = 0
    self.interp_Xc_to       = 0
    self.interp_Xc_steps    = 0

    self.interp_eta         = False
    self.interp_eta_from    = 0
    self.interp_eta_to      = 0
    self.interp_eta_steps   = 0

    #.............................
    # Bookkeeping of the process
    #.............................
    # The status of the preparation
    self.interp_prepare_OK  = False
    # Effective parameters used for interpolation
    self.interp_param_names = ['']
    # Number of multi-D dimensions of the interpolant
    self.interp_n_dim       = 0
    # List of slice objects (Python built-in) for np.mgrid
    self.interp_slices      = []
    # List of 1D ndarrays for all interpolation dimensions
    self.interp_1d_points   = []
    # Count the number of prepared points: 
    # n=Prod(n_k), n_k=len(points_k), for the k-th dimension
    self.interp_n_points    = 0
    # The status of calling numpy.mgrid, and building the meshgrid
    self.interp_meshgrid_OK = False
    # The resulting (interp_n_dim) tuple of meshgrids, all with
    # identical shape
    self.interp_meshgrid    = []
    # The shape of the resulting meshgrid
    self.interp_meshgrid_shape = tuple()


  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    if not hasattr(self, attr):
      logger.error('interpolation: set: Attribute "{0}" is unavailable.')
      sys.exit(1)

    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('interpolation: get: Attribute "{0}" is unavailable.')
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def do_interpolate(self):
    """
    This routine carries out the interpolation of frequencies over non-uniformly 
    gridded background layout of data points (attributes like M_ini, Z, etc).
    """
    _do_interpolate(self)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    #####    ###  ###  ###    ###    ###########  ######### 
    #    #   #    #    #    #    #     # #    #    #    #  #       #
    #     #  #     #   #    #    #     # #         #       #
    #    #   #    #    #     #  #     #   #        #       ####
    #####    #####     #     #  #     #####        #       #
    #        #  #      #      #      #     #       #       #       #
    #        #    #   ###    ###    ###   ###      #       #########


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _do_interpolate(self):
  """
  Refer to the documentation of the do_interpolate() method for detailed information.
  """
  _collect_inputs(self)

  _check_inputs(self)

  _prepare(self)
  if not self.interp_prepare_OK:
    return False

  _build_meshgrid(self)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs_around_anchor(self):
  """
  Query the database for fixed points around the anchor point.
  """
  # Get the individual anchor parameter values
  anc_names = self.get('anchor_param_names')
  anc_vals  = self.get('anchor_param_values')
  for k, name in enumerate(anc_names):
    val         = anc_vals[k]
    if name == 'M_ini':
      anc_M_ini = val
    elif name == 'fov':
      anc_fov   = val
    elif name == 'Z':
      anc_Z     = val
    elif name == 'logD':
      anc_logD  = val
    elif name == 'Xc':
      anc_Xc    = val
    elif name == 'eta':
      anc_eta   = val
    else:
      logger.error('_collect_inputs_around_anchor: Anchor name "{0}" unrecognized!'.format(name))
      sys.exit(1)

  M_ini_n = self.get('inputs_around_anchor_M_ini_n')
  fov_n   = self.get('inputs_around_anchor_fov_n')
  Z_n     = self.get('inputs_around_anchor_Z_n')
  logD_n  = self.get('inputs_around_anchor_logD_n')
  Xc_n    = self.get('inputs_around_anchor_Xc_n')
  eta_n   = self.get('inputs_around_anchor_eta_n')
  if np.sum(np.array([M_ini_n, fov_n, Z_n, logD_n, Xc_n, eta_n])) == 0:
    logger.error('_collect_inputs_around_anchor: Set inputs_around_anchor_..._n > 0')
    sys.exit(1)

  names   = self.get('anchor_param_names')
  vals    = self.get('anchor_param_values')
  freqs   = self.get('anchor_frequencies')
  ords    = self.get('anchor_radial_orders')
  types   = self.get('anchor_mode_types')

  # Get the entire grid attributes of the "tracks" table (>12,000 records)
  dbname  = self.get('dbname')
  q_tracks= query.without_constraint(dbname=dbname, table='tracks', 
                                     returned_columns=['id', 'M_ini', 'fov', 'Z', 'logD'])

  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(q_tracks, None)
    tup_tracks = the_db.fetch_all()
    dtype_     = [('id', int), ('M_ini', 'f4'), ('fov', 'f4'), ('Z', 'f4'), ('logD', 'f4')]
    rec_       = utils.list_to_recarray(list_input=tup_tracks, dtype=dtype_)
    rec_tracks = np.sort(rec_)[:]
    n_tracks   = len(rec_tracks)

    trk_M_ini  = rec_tracks['M_ini']
    trk_fov    = rec_tracks['fov']
    trk_Z      = rec_tracks['Z']
    trk_logD   = rec_tracks['logD']

    uniq_M_ini = np.unique(trk_M_ini)
    uniq_fov   = np.unique(trk_fov)
    uniq_Z     = np.unique(trk_Z)
    uniq_logD  = np.unique(trk_logD)

    len_M_ini  = len(uniq_M_ini)
    len_fov    = len(uniq_fov)
    len_Z      = len(uniq_Z)
    len_logD   = len(uniq_logD)

  # Find neighbors in M_ini by manipulating the indixes
  def _get_ind(arr, target, n):
    ind     = np.argmin(np.abs(arr - target))
    i_from  = ind - n if ind - n >= 0 else 0
    i_to    = ind + n + 1 if ind + n + 1 <= len(arr) else len(arr)
    return (i_from, i_to)

  M_ini_from, M_ini_to = _get_ind(uniq_M_ini, anc_M_ini, M_ini_n)
  neighb_M_ini         = uniq_M_ini[M_ini_from : M_ini_to]
  M_ini_range          = [np.min(neighb_M_ini), np.max(neighb_M_ini)]

  # Find neighbors in fov by manipulating the indixes 
  fov_from, fov_to     = _get_ind(uniq_fov, anc_fov, fov_n)
  neighb_fov           = uniq_fov[fov_from : fov_to]
  fov_range            = [np.min(neighb_fov), np.max(neighb_fov)]

  # Find neighbors in Z, knowing that only 3 unique Z values are used in the grid
  Z_from, Z_to = _get_ind(uniq_Z, anc_Z, Z_n)
  neighb_Z     = uniq_Z[Z_from : Z_to]
  Z_range      = [np.min(neighb_Z), np.max(neighb_Z)]

  # For logD "it's complicated" ...
  # To simplify this, we set a scanning range for logD between 0 and the max(logD) value for the track
  # which has the highest M_ini
  max_M_ini    = np.max(neighb_M_ini)

  # Query the database to get all unique (M_ini, logD) values from tracks table.
  q_tracks     = query.get_tracks_distinct_M_ini_logD()
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(q_tracks, None)
    tup_M_logD = the_db.fetch_all()
    dtype_     = [('M_ini', 'f4'), ('logD', 'f4')]
    rec_M_logD = utils.list_to_recarray(tup_M_logD, dtype=dtype_)
    ind_logD   = np.where(rec_M_logD['M_ini'] == max_M_ini)[0]
    logD_vals_ = rec_M_logD['logD'][ind_logD]
    max_logD_  = np.max(logD_vals_)
  logD_range   = [0, max_logD_]

  # Now, we have to constrain the Xc range
  # First, we find the id of the anchor track 
  anc_track_id = db_lib.get_track_id(dbname_or_dbobj=dbname, M_ini=anc_M_ini, fov=anc_fov, Z=anc_Z, logD=anc_logD)
  if isinstance(anc_track_id, int):
    logger.info('_collect_inputs_around_anchor: tracks.id for the anchor model is "{0}"'.format(anc_track_id))
  elif isinstance(anc_track_id, bool):
    logger.error('_collect_inputs_around_anchor: Failed to find the ahchor model tracks.id')
    sys.exit(1)

  # Second, find all Xcs from this track
  q_Xc         = query.with_constraints(dbname=dbname, table='models', returned_columns=['Xc'], 
                       constraints_keys=['id_track'], constraints_ranges=[[anc_track_id, anc_track_id]])
  # Get all Xcs now
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(q_Xc, None)
    tup_Xcs    = the_db.fetch_all()
    tup_Xcs    = [tup[0] for tup in tup_Xcs]
    dtype_     = [('Xc', 'f4')]
    arr_Xc     = np.array(tup_Xcs)

  Xc_from, Xc_to = _get_ind(arr_Xc, anc_Xc, Xc_n)    
  neighb_Xc    = arr_Xc[Xc_from : Xc_to]
  Xc_range     = [neighb_Xc.min(), neighb_Xc.max()]

  # Get the query string for retrieving models.id based on the ranges found above
  q_models_id  = query.get_models_id_from_M_ini_fov_Z_logD_Xc(M_ini_range=M_ini_range, 
                              fov_range=fov_range, Z_range=Z_range, 
                              logD_range=logD_range, Xc_range=Xc_range)

  # Now, get the models.id
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(q_models_id, None)
    tup_ids    = the_db.fetch_all()
    models_ids = np.array([ tup[0] for tup in tup_ids ], dtype=int)
    n_models   = len(models_ids)
    logger.info('_collect_inputs_around_anchor: Found "{0}" models\n'.format(n_models))

  # Get the proper rotation rates
  dic_rot      = db_lib.get_dic_look_up_rotation_rates_id(self.dbname)
  ids_etas     = dic_rot.values()
  eta_vals     = dic_rot.keys()
  n_etas       = len(ids_etas)
  ind_sort     = sorted(range(n_etas), key=lambda k: eta_vals[k])
  ids_etas     = [ids_etas[k] for k in ind_sort] 
  eta_vals     = [eta_vals[k] for k in ind_sort]
  eta_from, eta_to = _get_ind(eta_vals, anc_eta, eta_n)
  neighb_eta   = eta_vals[eta_from : eta_to]
  neighb_id_eta= ids_etas[eta_from : eta_to]
  n_etas       = len(neighb_id_eta)
  
  ############################
  # Using the methods from sampling class
  ############################
  search_func  = self.get('search_function')
  if search_func is self.trim_modes_freely:
    self.set('search_freely_for_frequencies', True)
  elif search_func is self.trim_modes_by_dP:
    self.get('search_strictly_for_dP', True)
  elif search_func is self.trim_modes_by_df:
    self.get('trim_modes_by_df', True)
  else:
    print search_func
    logger.error('_collect_inputs_around_anchor: Could not locate the proper search_function')
    sys.exit(1)

  print self.search_freely_for_frequencies, self.search_strictly_for_dP, self.search_strictly_for_df
  sys.exit()
  # First, retrieve the models attributes (M_ini, fov, Z, logD, Xc) by providing models.id
  features_    = self.get_M_ini_fov_Z_logD_Xc_from_models_id(models_ids)

  tup_extract  = self.extract_gyre_modes_from_id_model_id_rot(models_ids, neighb_id_eta, 
                                                          features_)
  



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs_by_range(self):
  pass

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs(self):
  """
  This routine collects the inputs from the database by quering it. There are two possibilities:
  - querying around the anchor model (check out self.anchor_param_values)
  - querying for a range of input parameters, e.g. M_ini: [2 - 5], etc

  """
  if self.dbname == '':
    logger.error('_collect_inputs: You must specify the dbname.')
    sys.exit(1)
    
  flags  = np.array([self.inputs_around_anchor, self.inputs_by_range])
  n_True = np.sum(flags * 1)
  if n_True != 1:
    logger.error('_collect_inputs: Set only inputs_around_anchor or inputs_by_range to True')
    sys.exit(1)

  # eta will be missing, if sampling.exclude_eta_column = False
  # Thus, we recover the zero-valued eta here.
  if 'eta' not in self.get('anchor_param_names'):
    logger.warning('_collect_inputs: "eta" not in "anchor_param_names". We add eta=0 manually\n')
    self.anchor_param_names.append('eta')
    if isinstance(self.anchor_param_values, list):
      self.anchor_param_values.append(0.0)
    elif isinstance(self.anchor_param_values, np.ndarray):
      _vals = np.array( [v for v in self.get('anchor_param_values')] + [0.0] )
      self.set('anchor_param_values', _vals[:])
    else:
      logger.error('_collect_inputs: The type of anchor_param_values not supported yet.')
      sys.exit(1)

  # Choose one of the two possilbe methods to collect the input from the grid
  if self.inputs_around_anchor:
    _collect_inputs_around_anchor(self)

  if self.inputs_by_range:
    _collect_inputs_by_range(self)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _check_inputs(self):
  """
  For interpolation, you deinitely need an input, which must be compatible with the number of interpolation
  parameters, i.e. self.interp_n_dim. This routine provides two ndarrays, one for the x and one for y.
  The input x is a multi-dimensional ndarray of shape say (n, D), and the y is an ndarray of shape (n, ).
  """
  if isinstance(self.input_features, list):
    self.input_features = np.array(self.input_features)
  if isinstance(self.input_frequencies, list):
    self.input_frequencies = np.array(self.input_frequencies)

  shape_x = self.input_features.shape 
  shape_y = self.input_frequencies.shape
  # the number of rows of the two must match
  if shape_x[0] != shape_y[0]:
    logger.error('_check_inputs: The input features and frequencies have different number of row')
    sys.exit(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _build_meshgrid(self):
  """
  Build the underlying meshgrid on-top-of-which the interpolation will be carried out. The size of
  this each of the meshgrids might become huge, specifically when requiring too many interpolation 
  points along each of the parameter dimensions. Thus, care must be practiced here to ensure all 
  needed intermediate matirxes fit properly into the memory of the computing hardware/node.
  """
  if not self.interp_prepare_OK:
    return False

  slices = self.get('interp_slices')

  try:
    msh  = np.mgrid[[the_slice for the_slice in slices]] 
    self.set('interp_meshgrid_OK', True)
    self.set('interp_meshgrid', msh)
    self.set('interp_meshgrid_shape', msh.shape)
  except:
    self.set('interp_meshgrid_OK', False)
    self.set('interp_meshgrid', [])
    self.set('interp_meshgrid_shape', (0, ))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _prepare(self):
  """
  Prepare the variables needed for the multi-D interpolation  
  """
  if self.interp_Xc:    names.append('Xc')
  if self.interp_eta:   names.append('eta')

  names  = []
  slices = []
  points = []
  n_pts  = 1
  if self.interp_M_ini:
    names.append('M_ini')
    slices.append(slice(self.interp_M_ini_from, self.interp_M_ini_to, complex(0, self.interp_M_ini_steps)))
    points.append(np.linspace(self.interp_M_ini_from, self.interp_M_ini_to, self.interp_M_ini_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_fov:
    names.append('fov')
    slices.append(slice(self.interp_fov_from, self.interp_fov_to, complex(0, self.interp_fov_steps)))
    points.append(np.linspace(self.interp_fov_from, self.interp_fov_to, self.interp_fov_steps))
    n_pts *= len(points[-1])

  if self.interp_Z:
    names.append('Z')
    slices.append(slice(self.interp_Z_from, self.interp_Z_to, complex(0, self.interp_Z_steps)))
    points.append(np.linspace(self.interp_Z_from, self. interp_Z_to, self.interp_Z_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_logD:
    names.append('logD')
    slices.append(slice(self.interp_logD_from, self.interp_logD_to, complex(0, self.interp_logD_steps)))
    points.append(np.linspace(self.interp_logD_from, self.interp_logD_to, self.interp_logD_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_Xc:
    names.append('Xc')
    slices.append(slice(self.interp_Xc_from, self.interp_Xc_to, complex(0, self.interp_Xc_steps)))
    points.append(np.linspace(self.interp_Xc_from, self.interp_Xc_to, self.interp_Xc_steps, endpoint=True))
    n_pts *= len(points[-1])

  if self.interp_eta:
    names.append('eta')
    slices.append(slice(self.interp_eta_from, self.interp_eta_to, complex(0, self.interp_eta_steps)))
    points.append(np.linspace(self.interp_eta_from, self.interp_eta_to, self.interp_eta_steps, endpoint=True))
    n_pts *= len(points[-1])

  n = len(names)
  if n == 0:
    logger.error('_prepare: You must specify at least one parameter for interpolation')
    self.set('interp_prepare_OK', False)
  else:
    self.set('interp_prepare_OK', True)

  self.set('interp_param_names', names)
  self.set('interp_n_dim', n)
  self.set('interp_slices', slices)
  self.set('interp_1d_points', points)
  self.set('interp_n_points', n_pts)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
