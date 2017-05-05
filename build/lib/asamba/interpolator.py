
"""
This module provides interpolation between frequencies over a give range of input 
stellar parameters. With this tool, it is no longer needed to compute too 
highly-resolved grids around the best asteroseismic models. Instead, the resolved 
models are prepared by the interpolation in between the grid points from the coarse model.

The interpolation class is a derived/subclass of the sampler.sampling class, because we require
several of the methods defined there in this module. This is to ensure we minimize/suppress 
the redundancy, and make the best runtime use of the parameters that are set therein.
"""
from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
import sys, os, glob
import time
import logging
import numpy as np
from scipy.interpolate import griddata

from asamba import utils, db_def, db_lib, query, sampler

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger  = logging.getLogger(__name__)
is_py3x = sys.version_info[0] >= 3 # to handle unicode encoding for Python v2.7

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
    self.inputs_by_range    = False

    # Ranges to search/query the database for the inputs
    self.interp_inputs_OK   = False
    self.interp_range_M_ini = []
    self.interp_range_fov   = []
    self.interp_range_Z     = []
    self.interp_range_logD  = []
    self.interp_range_Xc    = []
    self.interp_range_eta   = []
    self.interp_eta_ids     = []

    # Check flag for the input
    self.interp_check_inputs_OK = False

    #.............................
    # Input features and frequencies
    # from the database (actual GYRE outputs)
    # m: number of rows
    # n: number of features
    # K: number of frequencies per row
    #.............................
    # The ndarray of input features; shape: (m, n)
    self.input_features     = []
    # The same matrix as a record array
    self.input_features_rec = []
    # The ndarray of input frequencies; shape: (m, K)
    self.input_frequencies  = []
    # The same matrix as a record array
    self.input_frequencies_rec = []

    #.............................
    # Specifications for interpolation
    #.............................
    # Parameter ranges, stepsizes, and ndarrays
    # *_array are ndarray of unique valeus, and 
    # interp_grid_* are n-dimensional matrixes
    self.interp_M_ini       = False
    self.interp_M_ini_from  = 0
    self.interp_M_ini_to    = 0
    self.interp_M_ini_steps = 0
    self.interp_M_ini_array = []
    self.interp_grid_M_ini  = []

    self.interp_fov         = False
    self.interp_fov_from    = 0
    self.interp_fov_to      = 0
    self.interp_fov_steps   = 0
    self.interp_fov_array   = []
    self.interp_grid_fov    = []

    self.interp_Z           = False
    self.interp_Z_from      = 0
    self.interp_Z_to        = 0
    self.interp_Z_steps     = 0
    self.interp_Z_array     = []
    self.interp_grid_Z      = []

    self.interp_logD        = False
    self.interp_logD_from   = 0
    self.interp_logD_to     = 0
    self.interp_logD_steps  = 0
    self.interp_logD_array  = []
    self.interp_grid_logD   = []

    self.interp_Xc          = False
    self.interp_Xc_from     = 0
    self.interp_Xc_to       = 0
    self.interp_Xc_steps    = 0
    self.interp_Xc_array    = []
    self.interp_grid_Xc     = []

    self.interp_eta         = False
    self.interp_eta_from    = 0
    self.interp_eta_to      = 0
    self.interp_eta_steps   = 0
    self.interp_eta_array   = []
    self.interp_grid_eta    = []

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
    # The flag on checking the meshgrid dimensionalities
    self.interp_check_meshgrid_OK = False
    # The resulting (interp_n_dim) tuple of meshgrids, all with
    # identical shape
    self.interp_meshgrid    = tuple()


  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    try:
      super(interpolation, self).set(attr, val)
    except:
      if not hasattr(self, attr):
        logger.error('interpolation: set: Attribute "{0}" is unavailable.'.format(attr))
        sys.exit(1)
      setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('interpolation: get: Attribute "{0}" is unavailable.'.format(attr))
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

  #.........................
  def collect_inputs(self):
    _collect_inputs(self)

  #.........................
  def check_inputs(self):
    _check_inputs(self)

  #.........................
  def prepare(self):
    _prepare(self)

  #.........................
  def build_meshgrid(self):
    _build_meshgrid(self)

  #.........................
  def check_meshgrid(self):
    _check_meshgrid(self)

  #.........................
  def call_griddata(self):
    _call_griddata(self)

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

  _build_meshgrid(self)

  # _check_meshgrid(self)

  _call_griddata(self)

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
    i_to    = ind + n + 1 if ind + n + 1 < len(arr) else len(arr)
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

  # Then, find the appropriate rotation rates
  dic_rot      = db_lib.get_dic_look_up_rotation_rates_id(self.dbname)
  ids_etas     = list(dic_rot.values())
  eta_vals     = list(dic_rot.keys())
  n_etas       = len(ids_etas)
  ind_sort     = sorted(list(range(n_etas)), key=lambda k: eta_vals[k])
  ids_etas     = [ids_etas[k] for k in ind_sort] 
  eta_vals     = [eta_vals[k] for k in ind_sort]
  eta_from, eta_to = _get_ind(eta_vals, anc_eta, eta_n)
  neighb_eta   = eta_range = eta_vals[eta_from : eta_to]
  neighb_id_eta= ids_etas[eta_from : eta_to]
  n_etas       = len(neighb_id_eta)
  
  # Set the ranges as the self attributes
  self.set('interp_range_M_ini', M_ini_range)
  self.set('interp_range_fov', fov_range)
  self.set('interp_range_Z', Z_range)
  self.set('interp_range_logD', logD_range)
  self.set('interp_range_Xc', Xc_range)
  if not self.exclude_eta_column:
    self.set('interp_range_eta', eta_range)
    self.set('interp_eta_ids', neighb_id_eta)
  
  # Now, the ranges are set internally, and ready to collect the inputs by range, as follows:
  _collect_inputs_by_range(self)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs_by_range(self):
  """
  This function queries the grid in order to provide as much interpolation nodes as possible. For that,
  it looks at the observed frequencies, and cherry picks those models (M_ini, fov, ...) from the grid 
  where the observed frequencies are reproduced (based on e.g. period spacing, frequency spacing, etc.).
  As a return, the "input_features", and "input_frequencies" attributes of the class will be updated.
  """
  # Get the query string for retrieving models.id based on the ranges
  # There ranges are either set by the user, or are found after a call to collect_inputs_around_anchor()
  M_ini_range  = self.get('interp_range_M_ini')
  fov_range    = self.get('interp_range_fov')
  Z_range      = self.get('interp_range_Z')
  logD_range   = self.get('interp_range_logD')
  Xc_range     = self.get('interp_range_Xc')
  eta_range    = self.get('interp_range_eta')
  eta_ids      = self.get('interp_eta_ids')

  # Check if the _range lists are OK
  try:
    assert len(M_ini_range) == 2
    assert len(fov_range) == 2
    assert len(Z_range) == 2
    assert len(logD_range) == 2
    assert len(Xc_range) == 2
    if not self.exclude_eta_column:
      assert len(eta_range) == 2
      assert len(eta_ids) > 0
  except AssertionError:
    logger.error('_collect_inputs_by_range: The input ranges do not have the proper length. Check them!')
    sys.exit(1)
    
  q_models_id  = query.get_models_id_from_M_ini_fov_Z_logD_Xc(M_ini_range=M_ini_range, 
                              fov_range=fov_range, Z_range=Z_range, 
                              logD_range=logD_range, Xc_range=Xc_range)

  # Now, get the models.id
  with db_def.grid_db(dbname=self.dbname) as the_db:
    the_db.execute_one(q_models_id, None)
    tup_ids    = the_db.fetch_all()
    models_ids = np.array([ tup[0] for tup in tup_ids ], dtype=int)
    n_models   = len(models_ids)
    logger.info('_collect_inputs_by_range: Found "{0}" models\n'.format(n_models))

  ############################
  # Using the methods from sampling class
  ############################
  # First, retrieve the models attributes (M_ini, fov, Z, logD, Xc) by providing models.id
  features_    = self.get_M_ini_fov_Z_logD_Xc_from_models_id(models_ids)
  n_features   = len(features_)
  try:
    assert n_features == n_models
  except AssertionError:
    logger.error('_collect_inputs_by_range: Inconsistent number of feature rows retrieved!')
    sys.exit(1)

  # Figure out whether or not to include the eta column
  if self.exclude_eta_column:
    models_ids_= models_ids
    dic_rot    = db_lib.get_dic_look_up_rotation_rates_id(self.dbname)
    ids_etas   = list(dic_rot.values())
    rot_ids_   = [min(ids_etas)] * n_features
    rows_names = ['M_ini', 'fov', 'Z', 'logD', 'Xc']
    stiched    = features_[:]
  else:
    models_ids_= models_ids * n_etas                     # size: n_models * n_etas
    rot_ids_   = ids_etas[eta_from : eta_to] * n_models  # size: n_models * n_etas
    eta_vals_  = eta_vals[eta_from : eta_to] * n_models  # size: n_models * n_etas
    rows_names = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'eta']
    stiched    = [features_[k] + eta_vals_[k] for k in range(n_models * n_etas)]

  # Then, extract the GYRE mode lists from their id
  tup_extract  = self.extract_gyre_modes_from_id_model_id_rot(models_ids_, rot_ids_, stiched)
  rows_keep    = tup_extract[0]
  model_keep   = tup_extract[1]
  rot_keep     = tup_extract[2]
  rec_keep     = tup_extract[3]
  
  # convert surviving rows to recarray to fetch some information
  rows_type    = [(name, 'f4') for name in rows_names]
  rec_rows     = utils.list_to_recarray(rows_keep, rows_type)

  # Reset the feature ranges, now that the outliying models are filtered out
  self.set('interp_range_M_ini', [rec_rows['M_ini'].min(), rec_rows['M_ini'].max()])
  self.set('interp_range_fov', [rec_rows['fov'].min(), rec_rows['fov'].max()])
  self.set('interp_range_Z', [rec_rows['Z'].min(), rec_rows['Z'].max()])
  self.set('interp_range_logD', [rec_rows['logD'].min(), rec_rows['logD'].max()])
  self.set('interp_range_Xc', [rec_rows['Xc'].min(), rec_rows['Xc'].max()])
  if not self.exclude_eta_column:
    self.set('interp_range_eta', [rec_rows['eta'].min(), rec_rows['eta'].max()])
    self.set('interp_eta_ids', neighb_id_eta)

  logger.info('\n _collect_inputs_by_range: min to max for features around the anchor model:')
  logger.info('M_ini: {0:.3f} to {1:.3f}'.format(min(rec_rows['M_ini']), max(rec_rows['M_ini'])))
  logger.info('fov  : {0:.3f} to {1:.3f}'.format(min(rec_rows['fov']), max(rec_rows['fov'])))
  logger.info('Z    : {0:.3f} to {1:.3f}'.format(min(rec_rows['Z']), max(rec_rows['Z'])))
  logger.info('logD : {0:.3f} to {1:.3f}'.format(min(rec_rows['logD']), max(rec_rows['logD'])))
  logger.info('Xc   : {0:.3f} to {1:.3f}'.format(min(rec_rows['Xc']), max(rec_rows['Xc'])))
  if not self.exclude_eta_column:
    logger.info(' - eta  : {0:.3f} to {1:.3f}'.format(min(rec_rows['eta']), max(rec_rows['eta'])))
  print() 

  # assign the returned feature rows and frequencies to the self
  rows_keep    = utils.list_to_ndarray(rows_keep)
  self.set('input_features', rows_keep)
  rec_         = utils.list_to_recarray(rows_keep, dtype=[(name, 'f4') for name in rows_names])
  self.set('input_features_rec', rec_)

  freq_keep    = utils.list_to_ndarray([rec_['freq'] for rec_ in rec_keep])
  self.set('input_frequencies', freq_keep)
  n_freq       = self.input_frequencies.shape[1]
  f_names      = ['f{0}'.format(k) for k in range(n_freq)]
  dtype_       = [(f_name, 'f4') for f_name in f_names]
  rec_         = utils.list_to_recarray([rec_['freq'] for rec_ in rec_keep], dtype=dtype_)
  self.set('input_frequencies_rec', rec_)

  logger.info('_collect_inputs_by_range: inputs successfully collected \n')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _collect_inputs(self):
  """
  This routine collects the inputs from the database by quering it. There are two possibilities:
  - querying around the anchor model (check out self.anchor_param_values)
  - querying for a range of input parameters, e.g. M_ini: [2 - 5], etc
  The choice on the plan on how to do the input collection is made by the user through setting one 
  of the following attributes to "True": self.inputs_around_anchor, self.inputs_by_range
  """
  if self.interp_inputs_OK: return 

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

  # Choose one of the two possilbe methods to collect the input from the database
  if self.inputs_around_anchor:
    _collect_inputs_around_anchor(self) # and _collect_inputs_by_range() will be called immediately
  elif self.inputs_by_range:
    _collect_inputs_by_range(self)      # just call _collect_inputs_by_range()
  else:
    logger.error('_collect_inputs: Ambiguous collection plan!')
    sys.exit(1)

  self.set('interp_inputs_OK', True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _check_inputs(self):
  """
  For interpolation, you deinitely need an input, which must be compatible with the number of interpolation
  parameters, i.e. self.interp_n_dim. This routine provides two ndarrays, one for the x and one for y.
  The input x is a multi-dimensional ndarray of shape say (n, D), and the y is an ndarray of shape (n, ).
  
  In case of an inconsistency between the ranges for interpolation, and what is found from querying the database, 
  a warning is raised, and the interpolation attribute is forcefully set.
  """
  if self.interp_check_inputs_OK: return 

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

  # Ensure the interpolation range asked by the user complies with the ranges of input features.
  # In case of an inconsistency, a warning is raised, and the interpolation attribute is forcefully set.
  if self.interp_M_ini:
    if self.interp_M_ini_from < self.interp_range_M_ini[0]:
      self.set('interp_M_ini_from', self.interp_range_M_ini[0])
      logger.warning('_check_inputs: Forcing interp_M_ini_from to {0:.3f}'.format(self.interp_M_ini_from))
    if self.interp_M_ini_to > self.interp_range_M_ini[1]:
      self.set('interp_M_ini_to', self.interp_range_M_ini[1])
      logger.warning('_check_inputs: Forcing interp_M_ini_to to {0:.3f}'.format(self.interp_M_ini_to))
    if self.interp_M_ini_steps < 1:
      logger.error('_check_inputs: interp_M_ini_steps must be > 0')
      sys.exit(1)

  if self.interp_fov:
    if self.interp_fov_from < self.interp_range_fov[0]:
      self.set('interp_fov_from', self.interp_range_fov[0])
      logger.warning('_check_inputs: Forcing interp_fov_from to {0:.3f}'.format(self.interp_fov_from))
    if self.interp_fov_to > self.interp_range_fov[1]:
      self.set('interp_fov_to', self.interp_range_fov[1])
      logger.warning('_check_inputs: Forcing interp_fov_to to {0:.3f}'.format(self.interp_fov_to))
    if self.interp_fov_steps < 1:
      logger.error('_check_inputs: interp_fov_steps must be > 0')
      sys.exit(1)

  if self.interp_Z:
    if self.interp_Z_from < self.interp_range_Z[0]:
      self.set('interp_Z_from', self.interp_range_Z[0])
      logger.warning('_check_inputs: Forcing interp_Z_from to {0:.3f}'.format(self.interp_Z_from))
    if self.interp_Z_to > self.interp_range_Z[1]:
      self.set('interp_Z_to', self.interp_range_Z[1])
      logger.warning('_check_inputs: Forcing interp_Z_to to {0:.3f}'.format(self.interp_Z_to))
    if self.interp_Z_steps < 1:
      logger.error('_check_inputs: interp_Z_steps must be > 0')
      sys.exit(1)

  if self.interp_logD:
    if self.interp_logD_from < self.interp_range_logD[0]:
      self.set('interp_logD_from', self.interp_range_logD[0])
      logger.warning('_check_inputs: Forcing interp_logD_from to {0:.3f}'.format(self.interp_logD_from))
    if self.interp_logD_to > self.interp_range_logD[1]:
      self.set('interp_logD_to', self.interp_range_logD[1])
      logger.warning('_check_inputs: Forcing interp_logD_to to {0:.3f}'.format(self.interp_logD_to))
    if self.interp_logD_steps < 1:
      logger.error('_check_inputs: interp_logD_steps must be > 0')
      sys.exit(1)

  if self.interp_Xc:
    if self.interp_Xc_from < self.interp_range_Xc[0]:
      self.set('interp_Xc_from', self.interp_range_Xc[0])
      logger.warning('_check_inputs: Forcing interp_Xc_from to {0:.3f}'.format(self.interp_Xc_from))
    if self.interp_Xc_to > self.interp_range_Xc[1]:
      self.set('interp_Xc_to', self.interp_range_Xc[1])
      logger.warning('_check_inputs: Forcing interp_Xc_to to {0:.3f}'.format(self.interp_Xc_to))
    if self.interp_Xc_steps < 1:
      logger.error('_check_inputs: interp_Xc_steps must be > 0')
      sys.exit(1)
  
  if self.exclude_eta_column and self.interp_eta:
    logger.error('_check_inputs: Inconsistecy found: set exclude_eta_column opposite to interp_eta')
    sys.exit(1)

  if self.interp_eta:
    if self.interp_eta_from < self.interp_range_eta[0]:
      self.set('interp_eta_from', self.interp_range_eta[0])
      logger.warning('_check_inputs: Forcing interp_eta_from to {0:.3f}'.format(self.interp_eta_from))
    if self.interp_eta_to > self.interp_range_eta[1]:
      self.set(['interp_eta_to'], self.interp_range_eta[1])
      logger.warning('_check_inputs: Forcing interp_eta_to to {0:.3f}'.format(self.interp_eta_to))
    if self.interp_eta_steps < 1:
      logger.error('_check_inputs: interp_eta_steps must be > 0')
      sys.exit(1)

  self.set('interp_check_inputs_OK', True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _prepare(self):
  """
  Prepare the variables needed for the multi-D interpolation  
  """
  if self.interp_prepare_OK: return 

  if not self.interp_inputs_OK:
    logger.error('_prepare: You must first collect the inputs')
    sys.exit(1)

  # Get interpolation feature matrix 
  X      = self.get('input_features')

  # Derived lists relevant for the interpolation
  names  = []
  slices = []
  points = []
  n_pts  = 1

  # M_ini
  if self.interp_M_ini:
    slices.append(slice(self.interp_M_ini_from, self.interp_M_ini_to, complex(0, self.interp_M_ini_steps)))
    _a   = np.linspace(self.interp_M_ini_from, self.interp_M_ini_to, self.interp_M_ini_steps, endpoint=True)
  else:
    _a   = np.unique(X['M_ini'])
  names.append('M_ini')
  self.set('interp_M_ini_array', _a)
  points.append(_a)
  n_pts  *= len(_a)

  # fov
  if self.interp_fov:
    slices.append(slice(self.interp_fov_from, self.interp_fov_to, complex(0, self.interp_fov_steps)))
    _b   = np.linspace(self.interp_fov_from, self.interp_fov_to, self.interp_fov_steps, endpoint=True)
  else:
    _b   = np.unique(X['fov'])   
  names.append('fov')
  self.set('interp_fov_array', _b)
  points.append(_b)
  n_pts  *= len(_b)

  # Z
  if self.interp_Z:
    slices.append(slice(self.interp_Z_from, self.interp_Z_to, complex(0, self.interp_Z_steps)))
    _c   = np.linspace(self.interp_Z_from, self. interp_Z_to, self.interp_Z_steps, endpoint=True)
  else:
    _c   = np.unique(X['Z'])
  names.append('Z')
  self.set('interp_Z_array', _c)
  points.append(_c)
  n_pts *= len(_c)

  # logD
  if self.interp_logD:
    slices.append(slice(self.interp_logD_from, self.interp_logD_to, complex(0, self.interp_logD_steps)))
    _d   = np.linspace(self.interp_logD_from, self.interp_logD_to, self.interp_logD_steps, endpoint=True)
  else:
    _d   = np.unique(X['logD'])
  names.append('logD')
  self.set('interp_logD_array', _d)
  points.append(_d)
  n_pts *= len(points[-1])

  # Xc
  if self.interp_Xc:
    slices.append(slice(self.interp_Xc_from, self.interp_Xc_to, complex(0, self.interp_Xc_steps)))
    _e   = np.linspace(self.interp_Xc_from, self.interp_Xc_to, self.interp_Xc_steps, endpoint=True)
  else:
    _e   = np.unique(X['Xc'])
  names.append('Xc')
  self.set('interp_Xc_array', _e)
  points.append(_e)
  n_pts *= len(_e)

  # eta
  if not self.exclude_eta_column and self.interp_eta:
    slices.append(slice(self.interp_eta_from, self.interp_eta_to, complex(0, self.interp_eta_steps)))
    _f   = np.linspace(self.interp_eta_from, self.interp_eta_to, self.interp_eta_steps, endpoint=True)
    names.append('eta')
    self.set('interp_eta_array', _f)
    points.append(_f)
    n_pts *= len(_f)
  else:
    pass

  ndim = len(names)
  if ndim == 0:
    self.set('interp_prepare_OK', False)
    logger.warning('_prepare: You must specify at least one parameter for interpolation')
  else:
    self.set('interp_prepare_OK', True)

  self.set('interp_param_names', names)
  self.set('interp_n_dim', ndim)
  self.set('interp_slices', slices)
  self.set('interp_1d_points', points)
  self.set('interp_n_points', n_pts)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _build_meshgrid(self):
  """
  Build the underlying meshgrid on-top-of-which the interpolation will be carried out. The size of
  this each of the meshgrids might become huge, specifically when requiring too many interpolation 
  points along each of the parameter dimensions. Thus, care must be practiced here to ensure all 
  needed intermediate matirxes fit properly into the memory of the computing hardware/node.
  """
  if self.interp_meshgrid_OK: return 

  if not self.interp_prepare_OK:
    return False

  # slices    = self.get('interp_slices')
  arr_M_ini = self.get('interp_M_ini_array')
  arr_fov   = self.get('interp_fov_array')
  arr_Z     = self.get('interp_Z_array')
  arr_logD  = self.get('interp_logD_array')
  arr_Xc    = self.get('interp_Xc_array')
  arr_eta   = self.get('interp_eta_array')

  print(arr_M_ini)
  print(arr_fov)
  print(arr_Z)
  print(arr_logD)
  print(arr_Xc)
  print(arr_eta)
  print(self.interp_n_dim)
  print(self.interp_n_points)

  try:
    if self.exclude_eta_column:
      grid_M_ini, grid_fov, grid_Z, grid_logD, grid_Xc = np.meshgrid(
          arr_M_ini, arr_fov, arr_Z, arr_logD, arr_Xc, indexing='ij')
      msh   = (grid_M_ini, grid_fov, grid_Z, grid_logD, grid_Xc)
    else:
      grid_M_ini, grid_fov, grid_Z, grid_logD, grid_Xc, grid_eta = np.meshgrid(
          arr_M_ini, arr_fov, arr_Z, arr_logD, arr_Xc, arr_eta, indexing='ij')
      msh   = (grid_M_ini, grid_fov, grid_Z, grid_logD, grid_Xc, grid_eta)
      self.set('interp_grid_eta', grid_eta)

    # msh  = np.mgrid[[the_slice for the_slice in slices]] 
    self.set('interp_meshgrid_OK', True)

    self.set('interp_meshgrid', msh)

    self.set('interp_grid_M_ini', grid_M_ini)
    self.set('interp_grid_fov', grid_fov)
    self.set('interp_grid_Z', grid_Z)
    self.set('interp_grid_logD', grid_logD)
    self.set('interp_grid_Xc', grid_Xc)

    logger.info('_build_meshgrid: succeeded\n')
  except:
    self.set('interp_meshgrid_OK', False)
    logger.warning('_build_meshgrid: failed\n')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _check_meshgrid(self):
  """
  This routine checks the dimensionalities of all inputs to the scipy.interpolate.griddata() routine,
  before calling the routine, to ensure we comply with the expected array/matrix shapes. The variable
  naming conventions inside this routine follows closely the documentation of the griddata function.
  For further details see: 
  <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata">griddata</a>
  """
  if self.interp_check_meshgrid_OK: return 

  points   = self.get('input_features')
  freqs    = self.get('input_frequencies')
  meshgrid = self.get('interp_meshgrid')

  n, D     = points.shape
  n_       = freqs[:,0].shape
  M, D_    = meshgrid.shape

  try:
    assert n == n_
  except AssertionError:
    logger.error('_check_meshgrid: Incompatible number of input rows')
    sys.exit(1)

  try: 
    assert D == D_ 
  except AssertionError:
    logger.error('_check_meshgrid: Incompatible number of dimensions')
    sys.exit(1)

  print(n, D)
  print(M, D_)

  self.set('interp_check_meshgrid_OK', True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _call_griddata(self):
  """
  This function is a wrapper around the scipy.interpolate.griddata(), to have maximum control over the
  inputs/outputs of that function, and provide as much flexibility as we desire in using that routine
  for interpolations along any desired dimention of our problem.
  """
  if not self.interp_meshgrid_OK:
    self.error('_call_griddata: You must create the meshgrid first')
    sys.exit(1)
  meshgrid = self.get('interp_meshgrid')

  points = self.get('input_features')
  freqs  = self.get('input_frequencies')
  m, K   = freqs.shape

  for k in range(K):
    t0 = time.time()
    print('loop: ', k)
    freq_k = freqs[:, k]
    a = griddata(points, freq_k, meshgrid, method='linear')

    print(a.shape)
    print('took {0:.4f} sec'.format(time.time() - t0))
    break

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
