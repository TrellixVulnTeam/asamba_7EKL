
"""
This module prepares training/validatin/test datasets to train/validate/test an 
artificial neural network. This is achieved through the "sampling" class, which 
handles the task of collecting the models properly from the database.

This module inherits from the "star" module, in order to sample the model frequencies
based on the observed frequencies. On the flip side, it serves as superclass for the 
interpolator.interpolation() class, who inherits/needs several of the functionlaities 
offered in here. 
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from builtins import range
import sys, os, glob
import logging
import time
import itertools
import numpy as np 

from asamba import utils, db_def, db_lib, query, star

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
# S A M P L I N G   C L A S S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class sampling(star.star):
  """
  This class carries out sampling of the learning sets from the database. This class inherits the
  "star.star()" object to represent a star
  """

  global_ = 0

  def __init__(self):
    super(sampling, self).__init__()

    #.............................
    # The basic search constraints
    #.............................
    # The database to retrieve samples from
    self.dbname = ''
    # Sampling function name
    self.sampling_func = None
    # shuffle the sample
    self.sampling_shuffle = True
    # Maximum sample size to slice from all possible combinations
    self.max_sample_size = -1
    # The range in log_Teff to constrain 
    self.range_log_Teff = [-1, -1]
    # The range in log_g to constrain
    self.range_log_g = [-1, -1]
    # The range in rotation rate (percentage)
    self.range_eta = [-1, -1]

    #.............................
    # Return of the basic search
    #.............................
    # The models.id for the sample
    self.ids_models = []
    # The rotation_rates.id for the sample
    self.ids_rot = []
    # log_Teff of models
    # self.log_Teff_models = []
    # log_g of models
    # self.log_g_models = []

    #.............................
    # The resulting sample of attributes
    #.............................
    # Status of the learning dataset
    self.learning_done = False
    # Exclude eta = 0 from features (avoid singular matrix)
    self.exclude_eta_column = False
    # Explicit the number of features
    self.num_features = 0
    # Names of learning features in the order queried from database
    self.feature_names = ['']
    # model ids of the learning set
    self.learning_ids_models = []
    # rotation ids of the learning set
    self.learning_ids_rot = []
    # Resulting sample of features (type numpy.recarray)
    self.learning_x = None
    # Corresponding 2D frequency matrix for all features (type numpy.ndarray)
    self.learning_y = None
    # The sample size 
    self.sample_size = 0
    # log_Teff for the learning set
    self.learning_log_Teff = []
    # log_g for the learning set
    self.learning_log_g = []
    # Corresponding mode radial orders
    self.learning_radial_orders = []
    # Corresponding mode types (defined in grid.sql)
    self.learning_mode_types = []

    #.............................
    # Search constraints for modes
    #.............................
    # Modes id_types (as defined in grid.sql) to fetch frequencies from, e.g. [0, 6]
    # for radial (0) and quadrupole zonal (6) modes
    self.modes_id_types = []
    # Modes lower and upper frequency scan range
    self.modes_freq_range = []

    #.............................
    # Frequency search plans
    #.............................
    # Search function/callable used for sampling
    self.search_function        = None
    # Liberal search without any restriction
    self.search_for_closest_frequencies = False
    # Strict search for period spacings
    self.search_strictly_for_dP = False
    # Strict search for frequency spacings
    self.search_strictly_for_df = False
    # Match from closest smallest frequency
    # and proceed to higher frequencies
    self.match_lowest_frequency = True
    # What fraction of frequency difference around
    # the lowest and highest modes to keep the mode?
    # This cannot exceed 0.5
    self.trim_delta_freq_factor = 0.25

    #.............................
    # Sizes of different learning sets
    # Default: -1, means not set yet
    #.............................
    # Training, cross-validation and test samples
    self.training_percentage = -1
    self.training_size = -1
    self.training_x = -1
    self.training_y = -1
    self.training_log_Teff = []
    self.training_log_g = []
    self.training_set_done = False

    self.cross_valid_percentage = -1
    self.cross_valid_size = -1
    self.cross_valid_x = -1
    self.cross_valid_y = -1
    self.cross_valid_log_Teff = []
    self.cross_valid_log_g = []
    self.cross_valid_set_done = False

    self.test_percentage = -1
    self.test_size = -1
    self.test_x = -1
    self.test_y = -1
    self.test_log_Teff = []
    self.test_log_g = []
    self.test_set_done = False

    # #.............................
    # # Inheriting from the star module
    # #.............................
    # self.star = star.star()


  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    """
    Set a sampling attribute, e.g.
    >>>MySample = sampler.sampling()
    >>>MySample.set('range_log_Teff', [4.12, 4.27])

    @param attr: The name of the attribute to set
    @type attr: str
    @param val: The corresponding data (type and value) for the attribute. 
           Note that the users is mainly responsible for the sanity of the input values, 
           though we internally check for some basic compatibility. The val can take any
           datatype
    @type val: int, float, bool, list, etc.
    """
    super(sampling, self).set(attr, val)

    if not hasattr(self, attr):
      logger.error('sampling: set: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)

    # Some attributes require extra care/check
    if attr == 'dbname' and not is_py3x:
      val = val.encode('ascii')          # to handle unicode encoding for Python v2.7
    if attr == 'range_log_g' and val:
      if not isinstance(val, list) or len(val) != 2:
        logger.error('sampling: set: range_log_g: Range list must have only two elements')
        sys.exit(1)
      if self.get(attr)[0] == self.get(attr)[1] == -1:
        logger.error('sampling: set: You must specify the attribute: "{0}"'.format(attr))
        sys.exit(1)
    elif attr == 'range_log_Teff' and val:
      if not isinstance(val, list) or len(val) != 2:
        logger.error('sampling: set: range_log_Teff: Range list must have only two elements')
        sys.exit(1)
      if self.get(attr)[0] == self.get(attr)[1] == -1:
        logger.error('sampling: set: You must specify the attribute: "{0}"'.format(attr))
        sys.exit(1)
    elif attr == 'range_eta' and val:
      if not isinstance(val, list) or len(val) != 2:
        logger.error('sampling: set: range_eta: Range list must have only two elements')
        sys.exit(1)
      if self.get(attr)[0] == self.get(attr)[1] == -1:
        logger.error('sampling: set: You must specify the attribute: "{0}"'.format(attr))
        sys.exit(1)
    elif attr == 'modes_freq_range' and val:
      if not isinstance(val, list) or len(val) != 2:
        logger.error('sampling: set: modes_freq_range: Range list must have only two elements')
        sys.exit(1)
    elif attr == 'modes_id_types' and val:
      if not isinstance(val, list):
        logger.error('sampling: set: modes_id_types: Input must be a list of integers from grid.sql')
        sys.exit(1)
    
    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    """
    General-purpose method to get the value of a canonical attribute of the object
    E.g.

    >>>MySample = MyProblem.get('learning_x')

    @param attr: the name of the available attribute of the class
    @type attr: string
    @return: the value of the attribute
    @rtype: float
    """
    super(sampling, self).get(attr)

    if not hasattr(self, attr):
      logger.error('sampling: get: The attribute "{0}" is undefined'.format(attr))
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def build_learning_set(self):
    """
    This routine prepares a learning (training + cross-validation + test) set from the "tracks", "models",
    and "rotation_rates" table from the database "dbname". The sampling method of the data (constrained or
    unconstrained) is specified by passing the function name as "sampling_func", with the function arguments
    "sampling_args".

    The result from this function can be used to randomly build training, cross-validation, and/or test
    sets by random slicing.

    @param self: An instance of the sampling class
    @type self: obj
    @return: None. However, the "self.sample" attribute is set to a numpy record array whose columns are
          the following:
          - M_ini: initial mass of the model
          - fov: overshoot free parameter
          - Z: metallicity
          - logD: logarithm of extra diffusive mixing
          - Xc: central hydrogen mass fraction
          - eta: percentage rotation rate w.r.t. to the break up
    @rtype: None
    """
    _build_learning_sets(self)
    self.set('learning_done', True)

  ##########################
  def learning_log_Teff_log_g(self):
    """
    Fill up the "models_log_Teff" and "models_log_g" attributes of the class with the corresponding values retrieved
    from the models_ids. The resulting arrays are significantly important when dealing with priors for the Bayesian
    learning (see, e.g. artificial_neural_network.set_priors() method).
    """
    _learning_log_Teff_log_g(self)

  ##########################
  def split_learning_sets(self):
    """
    Split the learning set (prepared by calling build_learning_sets) into a training set, cross-validation
    set, and a test set. To do such, the following three attributes of the "sampling" class is used (so, they
    must have been already set to their non-default value):
      - training_percentage: (default -1); valid range: 0 to 100
      - cross_valid_percentage: (default -1); valid range: 0 to 100
      - test_percentage: (default -1); valid range: 0 to 100
    As a result of applying this method, the following variables are set
      - training_size = -1
      - training_x = -1
      - training_y = -1

      - cross_valid_size = -1
      - cross_valid_x = -1
      - cross_valid_y = -1

      - test_size = -1
      - test_x = -1
      - test_y = -1

    Note: once the training/cross-validation/test sets (i.e. *_x and *_y) are prepared, they are randomly
          shuffled internally. So, one shall never reshuffle them, else the ordering of different arrays 
          become inconsistent.

    @param self: An instance of the sampling class
    @type self: obj
    @return: the above nine parameters will be set
    @rtype: None
    """
    _split_learning_sets(self)
    self.set('training_set_done', True)
    self.set('cross_valid_set_done', True)
    self.set('test_set_done', True)

  ##########################
  def get_M_ini_fov_Z_logD_Xc_from_models_id(self, list_ids_models):
    """
    This routine queries the models table in the database, and returns tuples of the global attributes
    (M_ini, fov, Z, logD, Xc) that match the models.id passed by as the "list_ids_models" argument.
    This routine can lie in the heart of many applications which require retrieving of the global attributes
    by only providing/knowing the corresponding model id.

    Note: Even if one model id is passed (repeated) several times in the following query, only the first
    occurance is effective. Therefore, the size of the returned results from the following query
    is a factor (len(set(ids_rot))) larger than the result of the query. Then, the problem of 1-to-1
    matching is resolved by setting up a look-up dictionary, internally.

    @param self: an instance of the sampler.sampling class
    @type self: object
    @param list_ids_models: the list of models.id (integers) giving the exact models.id as in the database
    @type list_ids_models: list of integers
    @return: tuples of the corresponding attributes/feature values (M_ini, fov, Z, logD, Xc) for the unique
          values in the input list_ids_models. The repeated ids will be reconstructed by walking through the
          input models.id and putting the excluded attributes back in their place.
    @rtype: list of tuples
    """
    return _get_M_ini_fov_Z_logD_Xc_from_models_id(self, list_ids_models)

  ##########################
  def extract_gyre_modes_from_id_model_id_rot(self, list_ids_models, list_ids_rot, list_rows):
    """
    This finds a GYRE outputs for a model given its id_model, and id_rot. Then, it slices the
    modes based on the observed list, to ensure that there is a "reasonable" match between the
    model and observed frequencies. It returns various useful info, only for those models that
    survive the frequency filtering.

    @param self: an instance of the sampler class
    @type self: object
    @param list_ids_models: is a list of the id_model for all input models
    @type list_ids_models: list
    @param list_ids_rot: is a list of the id_rot for all input models
    @type list_ids_rot: list
    @param list_rows: a list of tuples, where each tuple is e.g. (M_ini, fov, Z, logD, Xc) for
             the input models. 
    @tupe list_rows: list
    @return: The following items are packed into the returned data structure:
           - list of (M_ini, fov, Z, logD) attribute tuples which fulfil the trimming condition. 
           - list of id_model which fulfil the trimming condition. This is basically
             a subset of the input list.
           - List of id_rot which fulfil the trimming condition. This is basically
             a subset of the input list.
           - List of record arrays for the corresponding models which fulfil the frequency
             filtering criteria.
    @rtype: tuple
    """
    return _extract_gyre_modes_from_id_model_id_rot(self, list_ids_models, list_ids_rot, list_rows)

  ##########################
  # Trimming Functions
  # Note: The signatures of the following three functions must be identical, because they are
  #       tossed into self.search_function, and can be called from external (inherited) modules
  ##########################
  def trim_closest_modes(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor):
    """ Not developed yet """
    _trim_closest_modes(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor)

  ##########################
  def trim_modes_by_dP(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor):
    """
    As the name explains, this function receives a record array of GYRE modes summary, and trims/clips
    the modes based on their period spacing pattern

    @param modes: The observed modes, where each mode in the list is an instance of the "star.mode" class
    @type modes: list of star.mode
    @param rec_gyre: The numpy record array from GYRE frequency list coming from one GYRE output file
    @type rec_gyre: np.recarray
    @param dic_mode_types: Look up dictionary to match the modes identification (l, m) with the modes.id_type
          attribute in the database. This dictionary is fetched from db_lib.get_dic_look_up_mode_types_id(). 
          However, we pass it as an argument instead of fetching it internally to speed up this function.
    @type dic_mode_types: dict
    @param trim_delta_freq_factor: This is the fraction of the frequency difference (delta_f) between the 
          first and last observed frequencies. Default:0.25. This delta is used to select models which have
          frequencies (f) in the range [f-df*factor , f+df*factor], where df is the frequency difference between
          two consecutive modes for the lowest and highest observed modes. If this factor is set to greater than
          0.5, the theoretical modes for two neighboring modes will overlap, and that will mess up the analysis.
          In that case, we raise an exception and terminate the program.
    @return: False if, for one among many reasons, it is not possible to trim the GYRE list based on the 
          observed modes. If it succeeds, the input GYRE list will be trimmed to match the size of the input
          modes, and then it will be returned.
    @rtype: np.recarray or bool
    """
    _trim_modes_by_dP(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor)

  ##########################
  def trim_modes_by_df(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor):
    """ Not developed yet """
    _trim_modes_by_df(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor)

  ##########################
  def trim_modes(self, rec_gyre, dic_mode_types):
    """
    Plan a strategy to trim the GYRE frequency list, and adapt it to the observed list based on the 
    requests of the user, i.e. based on the following attributes of the sampling object: 
    - search_for_closest_frequencies (Default = False)
    - search_strictly_for_dP (Default = False)
    - search_strictly_for_df (Default = False)
    - match_lowest_frequency (Default = True)
    - match_lowest_frequency (Default = 3.0)

    Note: The first three booleans specify the search method, and they are all False by defult. We check
    internally that only one of the flags is set to True, and the rest being False!

    Note: The return value from this routine is identical to the return from the following three functions:
    - _trim_closest_modes()
    - _trim_modes_by_dP()
    - _trim_modes_by_df()

    @param self: an instance of the "sampler.sampling" class
    @type self: object
    @param rec_gyre: the GYRE output list of frequencies as fetched from the database. The following
             columns are available here:
             - id_model: int32
             - id_rot: int16
             - n: int16
             - id_type: int16
             - freq: float32
    @type rec_gyre: np.recarray
    @param dic_mode_types: Look up dictionary to match the modes identification (l, m) with the modes.id_type
          attribute in the database. This dictionary is fetched from db_lib.get_dic_look_up_mode_types_id(). 
          However, we pass it as an argument instead of fetching it internally to speed up this function.
    @type dic_mode_types: dict
    @return: False, if for any reason no match is found between the observed and the modeled frequency lists.
             If successful, a matching slice of the input GYRE frequency list is returned.
    @rtype: np.recarray or bool
    """
    _trim_modes(self, rec_gyre, dic_mode_types)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    #####    ###  ###  ###    ###    ###########  ######### 
    #    #   #    #    #    #    #     # #    #    #    #  #       #
    #     #  #     #   #    #    #     # #         #       #
    #    #   #    #    #     #  #     #   #        #       ####
    #####    #####     #     #  #     #####        #       #
    #        #  #      #      #      #     #       #       #       #
    #        #    #   ###    ###    ###   ###      #       #########


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _split_learning_sets(self):
  """
  Refer to the documentation of the public method split_learning_set().  
  """
  p_train     = float(self.training_percentage)
  p_cv        = float(self.cross_valid_percentage)
  p_test      = float(self.test_percentage)
  percentages = [p_train, p_cv, p_test]
  if any(p < -1e-15 for p in percentages):
    logger.error('_split_learning_sets: Change the default (-1) for self.*_percentage=-1')
    sys.exit(1)

  if not all(0 <= p <= 1 for p in percentages):
    logger.error('_split_learning_sets: All three self.*_percentage must be set between 0 and 100')
    sys.exit(1)

  if np.abs(1-(p_train + p_cv + p_test)) > 1e-5:
    logger.error('_split_learning_sets: The three self.*_percentage do not all up to 1.0 (+/- 1e-5)')
    sys.exit(1)

  n_learn     = self.sample_size
  n_train     = int(n_learn * p_train)
  n_cv        = int(n_learn * p_cv)
  n_test      = int(n_learn * p_test)

  # due to round-off, the sum of splitted sets may not add up to sample_size, then ...
  if n_learn != n_train + n_cv + n_test:
    n_train   = n_learn - (n_cv + n_test)
  self.set('training_size', n_train)
  self.set('cross_valid_size', n_cv)
  self.set('test_size', n_test)

  # Make randomly shuffled indixes for slicing
  ind_learn   = np.arange(n_learn)
  np.random.shuffle(ind_learn)

  # reset various arrays according to the new indixes
  self.set('ids_models', self.ids_models[ind_learn])
  self.set('ids_rot', self.ids_rot[ind_learn])

  ind_train   = ind_learn[ : n_train]
  ind_cv      = ind_learn[n_train : n_train + n_cv]
  ind_test    = ind_learn[n_train + n_cv :]

  # slice the learning set into training/cross-validation/test sets
  learn_x     = np.empty_like(self.learning_x)
  learn_y     = np.empty_like(self.learning_y)
  learn_x[:]  = self.learning_x
  learn_y[:]  = self.learning_y

  self.set('training_x', learn_x[ind_train])
  self.set('training_y', learn_y[ind_train])

  self.set('cross_valid_x', learn_x[ind_cv])
  self.set('cross_valid_y', learn_y[ind_cv])

  self.set('test_x', learn_x[ind_test])
  self.set('test_y', learn_y[ind_test])

  # In some casas, the log_Teff and log_g values for the learning set is available too, which we fix now
  self.set('training_log_Teff', self.learning_log_Teff[ind_train])
  self.set('cross_valid_log_Teff', self.learning_log_Teff[ind_cv])
  self.set('test_log_Teff', self.learning_log_Teff[ind_test])

  self.set('training_log_g', self.learning_log_g[ind_train])
  self.set('cross_valid_log_g', self.learning_log_g[ind_cv])
  self.set('test_log_g', self.learning_log_g[ind_test])

  logger.info('_split_learning_sets" Training, Cross-Validation and Test sets are prepared')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _build_learning_sets(self):
  """
  Refer to the documentation of the public method build_learning_set().
  """
  # Sanity checks ...
  if not self.dbname:
    logger.error('_build_learning_sets: specify "dbname" attribute of the class')
    sys.exit(1)

  if self.sampling_func is None:
    logger.error('_build_learning_sets: specify "sampling_func" attribute of the class')
    sys.exit(1)

  # if self.star.num_modes == 0:
  if self.get('num_modes') == 0:
    logger.error('_build_learning_sets: The "modes" attribute of the "star" object of "sampling" not set yet!')
    sys.exit(1)

  # Get the list of tuples for the (id_model, id_rot) to fetch model attributes
  # tups_ids   = sampling_func(*sampler_args)  
  if self.sampling_func is constrained_pick_models_and_rotation_ids:

    if not self.range_log_Teff or not self.range_log_g or not self.range_eta:
      logger.error('_build_learning_sets: specify "ranges" properly')
      sys.exit(1)

    tups_ids = constrained_pick_models_and_rotation_ids(self) 
    logger.info('_build_learning_sets: constrained_pick_models_and_rotation_ids() succeeded')

  elif self.sampling_func is randomly_pick_models_and_rotation_ids:
    tups_ids = randomly_pick_models_and_rotation_ids(self)
    logger.info('_build_learning_sets: randomly_pick_models_and_rotation_ids succeeded')

  else:
    logger.error('_build_learning_sets: Wrong sampling function specified in the class')
    sys.exit(1)

  # Split the model ids from the eta ids
  n_tups     = len(tups_ids)
  if n_tups  == 0:
    logger.error('_build_learning_sets: The sampler returned empty list of ids.')
    sys.exit(1)
  # set the class attributes
  self.set('ids_models', np.array([tup[0] for tup in tups_ids]) )
  self.set('ids_rot', np.array([tup[1] for tup in tups_ids]) ) 
  self.set('sample_size', len(self.ids_models) )

  # convert the rotation ids to actual eta values through the look up dictionary
  dic_rot    = db_lib.get_dic_look_up_rotation_rates_id(self.dbname)

  # reverse the key/values of the dic, so that the id_rot be the key, and eta the values
  # also, the eta values are floats which are improper to compare. Instead, we convert
  # eta values to two-decimal point string representation, and do the conversion like that
  dic_rot_inv= {}
  for key, val in list(dic_rot.items()):
    str_eta  = '{0:.2f}'.format(key[0])
    dic_rot_inv[(val, )] = str_eta
  # create a 1-element tuple of eta values in f4 format to be stiched to a tuple of 
  # other attributes below
  eta_vals   = [ ( np.float32(dic_rot_inv[(id_rot,)]), ) for id_rot in self.ids_rot ]
  logger.info('_build_learning_sets: all eta values successfully collected')

  # Now, get the list of tuples for the features (M_ini, fov, ...) model per each
  features     = _get_M_ini_fov_Z_logD_Xc_from_models_id(self, self.ids_models)

  # whether or not include the eta column (must do for non-rot models, avoiding singular matrixes)
  if self.exclude_eta_column:
    stiched  = features[:]
  else:
    stiched  = [features[k] + eta_vals[k] for k in range(self.sample_size)]

  # Now, build the thoretical modes corresponding to each row in the sampled data
  # only accept those rows from the sample whose corresponding frequency row is useful
  # for our specific problem
  tup_extract= _extract_gyre_modes_from_id_model_id_rot(self, self.ids_models, 
                                                        self.ids_rot, stiched)
  rows_keep  = tup_extract[0]
  model_keep = tup_extract[1]
  rot_keep   = tup_extract[2]
  rec_keep   = tup_extract[3]

  # list of ndarrays to 2-D ndarrays
  mtrx_rows  = utils.list_to_ndarray(rows_keep) 
  stiched    = []           # destroy the list, and free up memory

  names      = ['M_ini', 'fov', 'Z', 'logD', 'Xc'] if self.exclude_eta_column else ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'eta']
  self.set('feature_names', names)
  self.set('num_features', len(self.feature_names))
  self.set('learning_x', mtrx_rows)
  self.set('sample_size', len(mtrx_rows))
  self.set('learning_ids_models', np.array( model_keep ))
  self.set('learning_ids_rot', np.array( rot_keep ))

  # and packing the frequencies (cycles per day) and friends
  rec_freq   = utils.list_to_ndarray([rec_['freq'] for rec_ in rec_keep]) 
  mtrx_n_pg  = utils.list_to_ndarray([rec_['n'] for rec_ in rec_keep]) 
  mtrx_types = utils.list_to_ndarray([rec_['id_type'] for rec_ in rec_keep]) 

  self.set('learning_y', rec_freq)
  self.set('learning_radial_orders', mtrx_n_pg)
  self.set('learning_mode_types', mtrx_types)

  # Include log_Teff and log_g for the learning set, at a cost of additional database query
  _learning_log_Teff_log_g(self)

  logger.info('_build_learning_sets: the attributes sampled successfully')

  return None

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _learning_log_Teff_log_g(self):
  """
  This routine iteratively finds the log_Teff and log_g for each of the models present in the learning
  set, based on their stored model ids. It can be used later to assign prior information based on Teff
  and log_g, if the user asks for it. So, this operation is not always needed.
  @param self: An instance of the sampling() class
  @type self: obj
  @return: The "learning_log_Teff" and "learning_log_g" attributes will be assigned
  @rtype: None
  """
  model_keep = self.get('learning_ids_models')
  n_keep     = len(model_keep)
  arr_log_Teff = np.empty(n_keep, dtype=np.float64)
  arr_log_g    = np.empty(n_keep, dtype=np.float64)

  with db_def.grid_db(dbname=self.dbname) as the_db:
    for i in range(n_keep):
      id_model = int(model_keep[i])
      q_models = query.get_log_Teff_log_g_from_models_id(id_model)
      the_db.execute_one(q_models, None)
      tup_res  = the_db.fetch_one()
      arr_log_Teff[i] = tup_res[0]
      arr_log_g[i]    = tup_res[1]

  # Add the filled-up arrays to the correct attributes of the sampling class
  self.set('learning_log_Teff', arr_log_Teff)
  logger.info('learning_log_Teff_log_g: Attribute learning_log_Teff is assigned')
  self.set('learning_log_g', arr_log_g)
  logger.info('learning_log_Teff_log_g: Attribute learning_log_g is assigned')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _get_M_ini_fov_Z_logD_Xc_from_models_id(self, list_ids_models):
  """
  For detailed documentation, please refer to get_M_ini_fov_Z_logD_Xc_from_models_id()
  """
  the_query  = query.get_M_ini_fov_Z_logD_Xc_from_models_id(list_ids_models)

  with db_def.grid_db(dbname=self.dbname) as the_db:
    the_db.execute_one(the_query, None)
    params   = the_db.fetch_all()
    n_par    = len(params)
    if n_par == 0:
      logger.error('_get_M_ini_fov_Z_logD_Xc_from_models_id: Found no matching model attributes')
      sys.exit(1)
    else:
      logger.info('_get_M_ini_fov_Z_logD_Xc_from_models_id: Fetched "{0}" unique models'.format(n_par))
    
    # local look-up dictionary
    dic_par  = {}
    for tup in params:
      key    = (tup[0], )   # i.e. models.id
      val    = tup[1:]      # i.e. (M_ini, fov, Z, logD, Xc)
      dic_par[key] = val
    # logger.info('_get_M_ini_fov_Z_logD_Xc_from_models_id: Look up dictionary for models is built')

  features   = [dic_par[(key, )] for key in list_ids_models]
  n_f        = len(features)

  logger.info('_get_M_ini_fov_Z_logD_Xc_from_models_id: Returned "{0}" model attributes\n'.format(n_f))

  return features
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _extract_gyre_modes_from_id_model_id_rot(self, list_ids_models, list_ids_rot, list_rows):
  """
  For detailed documentation, please refer to extract_gyre_modes_from_id_model_id_rot()
  """
  rows_keep  = []
  rec_keep   = []
  model_keep = []
  rot_keep   = []

  modes_dtype= [('id_model', np.int32), ('id_rot', np.int16), ('n', np.int16), 
                ('id_type', np.int16), ('freq', np.float32)]

  with db_def.grid_db(dbname=self.dbname) as the_db:
    
    # Get the mode_types look up dictionary
    dic_mode_types = db_lib.get_dic_look_up_mode_types_id(the_db)

    # Execute the prepared statement to speed up querying for self.sample_size times
    statement= 'prepared_statement_modes_from_fixed_id_model_id_rot'
    if the_db.has_prepared_statement(statement):
      the_db.execute_one('deallocate {0}'.format(statement), None)

    tup_query= query.modes_from_fixed_id_model_id_rot_prepared_statement(statement,
                     id_type=self.modes_id_types, freq_range=self.modes_freq_range)
    prepared_statement = tup_query[0]
    exec_statement     = tup_query[1]
    the_db.execute_one(prepared_statement, None)

    # Now, query the database iteratively for all sampling ids
    for k, row in enumerate(list_rows):
      id_model = int(list_ids_models[k]) # self.ids_models[k]
      id_rot   = int(list_ids_rot[k])    # self.ids_rot[k]

      # pack all query constraints into a tuple
      tup_exec = (id_model, id_rot) + tuple(self.modes_id_types) + tuple(self.modes_freq_range)

      the_db.execute_one(exec_statement, tup_exec)
      this     = the_db.fetch_all()

      try:
        rec_this = utils.list_to_recarray(this, modes_dtype)
      except terr:
        # print(sys.exc_info()[0])
        # print(type(this))
        sys.exit()

      # convert GYRE frequencies from "Hz" to "cd" for several benefits!
      rec_this['freq'] *= star.Hz_to_cd 

      # Trim off the GYRE list to match the observations
      rec_trim = _trim_modes(self, rec_this, dic_mode_types)

      # Decide whether or not to keep this (k-th) row based on the result of trimming
      if isinstance(rec_trim, bool) and rec_trim == False:
        continue            # skip this row
      else:
        rows_keep.append(row)
        model_keep.append(id_model)
        rot_keep.append(id_rot)
        rec_keep.append(rec_trim)

    logger.info('_extract_gyre_modes_from_id_model_id_rot: "{0}" models extracted\n'.format(len(rows_keep)))

    return rows_keep, model_keep, rows_keep, rec_keep

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _trim_modes(self, rec_gyre, dic_mode_types):
  """
  For the full documentation, please refer to the method trim_modes().
  """
  bool_arr = np.array([self.search_for_closest_frequencies, self.search_strictly_for_dP, 
                       self.search_strictly_for_df])
  n_True   = np.sum( bool_arr * 1 )
  if n_True != 1:
    logger.error('_trim_modes: Only one of the frequency search flags must be True, and the rest False.')
    sys.exit(1)

  if self.search_for_closest_frequencies:
    self.set('search_function', self.trim_closest_modes)
    return _trim_closest_modes()
  elif self.search_strictly_for_dP:
    self.set('search_function', self.trim_modes_by_dP)
    return _trim_modes_by_dP(self.get('modes'), rec_gyre, dic_mode_types, self.trim_delta_freq_factor)
  elif self.search_strictly_for_df:
    self.set('search_function', self.trim_modes_by_df)
    return _trim_modes_by_df()
  else:
    logger.error('_trim_modes: unexpected frequency search plan')
    sys.exit(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _trim_closest_modes(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor):
  """
  Choose matching frequencies with all liberty, without any restrictions/constraint.
  Not developed yet
  """
  n_modes = len(modes)
  n_rec   = len(rec_gyre)
  if n_rec < n_modes:
    # logger.warning('_trim_closest_modes: The number of observed modes is greater than the GYRE frequency list')
    return False

  freq_unit= modes[0].freq_unit
  if freq_unit != 'cd':
    logger.error('_trim_closest_modes: The observed frequencies must be in "per day (cd)" unit')
    sys.exit(1)

  # From observations, we have ...
  obs_freq = np.array([mode.freq for mode in modes]) # unit: per day
  n_freq   = len(obs_freq)
  d_freq   = obs_freq[1:] - obs_freq[:-1]
  d_freq_lo= d_freq[0]  
  d_freq_hi= d_freq[-1] 
  obs_l    = np.array([mode.l for mode in modes])
  obs_m    = np.array([mode.m for mode in modes])
  obs_n    = np.array([mode.n for mode in modes])

  obs_l_m  = [(obs_l[k], obs_m[k]) for k in range(n_freq)]

  # the_l    = list(set(obs_l))[0]
  # the_m    = list(set(obs_m))[0]
  # the_key  = (the_l, the_m)
  # the_id   = dic_mode_types[the_key]

  # From the GYRE output, we also have ...
  rec_types= rec_gyre['id_type']

  ind_l_m  = np.where(rec_types == the_id)[0]
  n_ind    = len(ind_l_m)
  if n_ind == 0:
    logger.error('_trim_closest_modes: No match between (l,m) of observed and model modes list')
    return False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _trim_modes_by_dP(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor):
  """
  For detailed documentation, please refer to the trim_modes_by_dP() method
  """
  n_modes = len(modes)
  n_rec   = len(rec_gyre)
  if n_rec < n_modes:
    # logger.warning('_trim_modes_by_dP: The number of observed modes is greater than the GYRE frequency list')
    return False

  freq_unit= modes[0].freq_unit
  if freq_unit != 'cd':
    logger.error('_trim_modes_by_dP: The observed frequencies must be in "per day (cd)" unit')
    sys.exit(1)

  # From observations, we have ...
  obs_freq = np.array([mode.freq for mode in modes]) # unit: per day
  d_freq   = obs_freq[1:] - obs_freq[:-1]
  d_freq_lo= d_freq[0]  
  d_freq_hi= d_freq[-1] 
  obs_l    = np.array([mode.l for mode in modes])
  obs_m    = np.array([mode.m for mode in modes])
  obs_n    = np.array([mode.n for mode in modes])

  the_l    = list(set(obs_l))[0]
  the_m    = list(set(obs_m))[0]
  the_key  = (the_l, the_m)
  the_id   = dic_mode_types[the_key]

  # From the GYRE output, we also have ...
  rec_types= rec_gyre['id_type']

  ind_l_m  = np.where(rec_types == the_id)[0]
  n_ind    = len(ind_l_m)
  if n_ind == 0:
    logger.error('_trim_modes_by_dP: No match between (l,m) of observed and model modes list')
    return False

  rec_gyre = rec_gyre[ind_l_m]
  rec_freq = rec_gyre['freq']
  n_rec    = len(rec_gyre)
  if n_rec < n_modes:
    logger.error('_trim_modes_by_dP: Frequency list is too short for this observations')
    return False
 
  factor    = trim_delta_freq_factor
  if factor > 0.5 :
    logger.error('_trim_modes_by_dP: We strongly recommend to lower trim_delta_freq_factor below 0.5. E.g. 0.25')
    sys.exit(1)
  freq_lo   = obs_freq[0] - d_freq_lo * factor
  freq_hi   = obs_freq[-1] + d_freq_hi * factor
  ind_trim  = np.where((rec_freq >= freq_lo) & (rec_freq <= freq_hi))[0]
  n_trim    = len(ind_trim)

  if n_trim != n_modes:
    # logger.warning('_trim_modes_by_dP: The trimmed array is smaller than the list of observed modes!')
    return False

  trimmed   = rec_gyre[ind_trim]
  
  return trimmed

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _trim_modes_by_df(modes, rec_gyre, dic_mode_types, trim_delta_freq_factor):
  """
  Choose matching frequencies based on regularities/spacings in frequency domain
  Not developed yet
  """
  logger.error('_trim_modes_by_df: Not developed yet')
  sys.exit(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  S A M P L I N G   T H E   I N P U T   M O D E L S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def constrained_pick_models_and_rotation_ids(self): 
               # (dbname, n, 
               # range_log_Teff=[3.5, 5], range_log_g=[0, 5], range_eta=[0, 51])
  """
  Return a combination of "models" id and "rotation_rate" id by applying constraints on log_Teff,
  log_g and rotation rates. For a totally random (unconstrained) 
  selection, you may call "randomly_pick_models_and_rotation_ids()", instead. 

  Notes:
  - the constraint ranges are inclusive. 
  - the results are fetched firectly from executing a SQL query
  - the combination of the models and rotation rates are shuffled
  
  Example of calling:
  >>>

  @param dbname: the name of the database
  @type dbname: str
  @param n: the *maximum* number of models to retrieve
  @type n: int
  @param range_log_Teff: the lower and upper range of log_Teff to scan the database. Default: [3.5, 5]
  @type range_log_Teff: list/tuple
  @param range_log_g: the lower and upper range of log_g to scan the database. Default: [0, 5]
  @type range_log_g: list/tuple
  @param range_eta: The range of rotation rates (in percentage w.r.t to critical, e.g. 15). 
         Default: [0, 50]
  @type range_eta: list/tuple
  @return: a shuffled list of 2-element tuples, with the first element being the model id, and the
         second element being the rotation_rate id.
  @rtype: list of tuples
  """
  dbname         = self.get('dbname')
  n              = self.get('max_sample_size')
  range_log_Teff = self.get('range_log_Teff')
  range_log_g    = self.get('range_log_g')
  range_eta      = self.get('range_eta')

  if not (len(range_log_Teff) == len(range_log_g) == len(range_eta) == 2):
    logger.error('constrained_pick_models_and_rotation_ids: Input "range" lists must have size = 2')
    sys.exit(1)

  # Get proper queries for each table
  q_models   = query.with_constraints(dbname=dbname, table='models',
                            returned_columns=['id'], 
                            constraints_keys=['log_Teff', 'log_g'], 
                            constraints_ranges=[range_log_Teff, range_log_g])

  q_rot      = query.with_constraints(dbname=dbname, table='rotation_rates',
                            returned_columns=['id'], 
                            constraints_keys=['eta'],
                            constraints_ranges=[range_eta])

  # Now, execute the queries, and fetch the data
  with db_def.grid_db(dbname=dbname) as the_db:
    # Execute the query for models
    the_db.execute_one(q_models, None)
    ids_models = [tup[0] for tup in the_db.fetch_all()]
    n_mod    = len(ids_models)
    if n_mod == 0:
      logger.error('constrained_pick_models_and_rotation_ids: Found no matching models.')
      sys.exit()

    # Execute the query for rotation rates
    the_db.execute_one(q_rot, None)
    ids_rot    = [tup[0] for tup in the_db.fetch_all()]
    n_rot      = len(ids_rot)
    if n_rot   == 0:
      logger.error('constrained_pick_models_and_rotation_ids: Found no matching rotation rates')
      sys.exit(1)

  combo      = [] 
  for id_rot in ids_rot:
    combo.extend( [(id_model, id_rot) for id_model in ids_models] )
  
  if self.sampling_shuffle:
    np.random.shuffle(combo)

  if n > 0:
    return combo[:n]
  else:
    return combo

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def randomly_pick_models_and_rotation_ids(self): 
  """
  Return a randomly-selected models together with their rotation rates from the database.
  This function fetches all model "id" number from the "models" table, in addition to all the "id"
  numbers from the "rotation_rates" table. Then, it iterates over them all, and creates all possible
  tupls with two elements: first element being the model id, and the second element being the rotaiton
  id. Then, this list is shuffled using the numpy.random.shuffle method, and only the subset of this
  whole list is returned, with the size specified by "n".

  @param dbname: The name of the database
  @type dbname: grid
  @param n: the size of the randomly-selected combinations of model id and rotation ids
  @type n: int
  @return: list of tuples where each tuple consists of two integers: 
     - the model id
     - the rotaiton id
  @rtype: list of tuples
  """
  dbname = self.get('dbname')
  n      = self.get('max_sample_size')

  if n < 1:
    logger.error('randomly_pick_models_and_rotation_ids: Specify n > 1')
    sys.exit(1)

  # Retrieve two look up dictionaries for the models table and the rotation table
  t1         = time.time()
  dic_models = db_lib.get_dic_look_up_models_id(dbname_or_dbobj=dbname)
  dic_rot    = db_lib.get_dic_look_up_rotation_rates_id(dbname_or_dbobj=dbname)
  t2         = time.time()
  print('Fetching two look up dictionaries took {0:.2f} sec'.format(t2-t1))

  ids_models = np.array([dic_models[key] for key in list(dic_models.keys())], dtype=np.int32)
  ids_rot    = np.array([dic_rot[key] for key in list(dic_rot.keys())], dtype=np.int16)
  t3         = time.time()
  print('List comprehensions took {0:.2f} sec'.format(t3-t2))

  n_mod      = len(ids_models)
  n_eta      = len(id_rot)

  t4         = time.time()
  print('Shuffling took {0:.2f} sec'.format(t4-t3))
  combo      = []
  for id_rot in ids_rot:
    combo.extend( [(id_model, id_rot) for id_model in ids_models] )

  if self.sampling_shuffle:
    np.random.shuffle(combo)

  t5         = time.time()
  print('The combo list took {0:.2f} sec'.format(t5-t4))

  print('Total time spent is {0:.2f} sec'.format(t5-t1))
  logger.info('randomly_pick_models_and_rotation_ids: Total time spent is {0:.2f} sec'.format(t5-t1))

  return combo[:n]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

