
"""
This module prepares training/validatin/test datasets to train/validate/test an 
artificial neural network. This is achieved through the "sampling" class, which 
handles the task of collecting the models properly from the grid.

This module inherits from the "star" module, in order to sample the model frequencies
based on the observed frequencies.
"""

import sys, os, glob
import logging
import time
import itertools
import numpy as np 

import utils, db_def, db_lib, query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# S A M P L I N G   C L A S S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class sampling:
  """
  This class carries out sampling of the learning sets from the database
  """

  def __init__(self):
    # The database to retrieve samples from
    self.dbname = ''
    # Sampling function name
    self.sampling_func = None
    # Maximum sample size to slice from all possible combinations
    self.max_sample_size = 0
    # The range in log_Teff to constrain 
    self.range_log_Teff = []
    # The range in log_g to constrain
    self.range_log_g = []
    # The range in rotation rate (percentage)
    self.range_eta = []
    # The models.id for the sample
    self.ids_models = []
    # The rotation_rates.id for the sample
    self.ids_rot = []

    # Resulting sample (type numpy.recarray)
    self.sample = None
    # The sample size 
    self.sample_size = 0

    # Modes id_types (from grid.sql) to fetch frequencies from, e.g. [0, 6]
    # for radial (0) and quadrupole zonal (6) modes
    self.modes_id_types = []
    # Modes lower and upper frequency scan range
    self.modes_freq_range = []

    # Training, cross-validation and test samples
    self.training_percentage = 0
    self.training_size = 0

    self.cross_valid_percentage = 0
    self.cross_valid_size = 0

    self.test_percentage = 0
    self.test_size = 0

  ##########################
  # Setters
  ##########################
  def set_dbname(self, dbname):
    self.dbname = dbname

  def set_sampling_func(self, sampling_func):
    self.sampling_func = sampling_func

  def set_max_sample_size(self, max_sample_size):
    self.max_sample_size = max_sample_size

  def set_range_log_Teff(self, range_log_Teff):
    if not isinstance(range_log_Teff, list) or len(range_log_Teff) != 2:
      logger.error('sampling: set_range_log_Teff: Range list must have only two elements')
      sys.exit(1)
    self.range_log_Teff = range_log_Teff

  def set_range_log_g(self, range_log_g):
    if not isinstance(range_log_g, list) or len(range_log_g) != 2:
      logger.error('sampling: set_range_log_g: Range list must have only two elements')
      sys.exit(1)
    self.range_log_g = range_log_g

  def set_range_eta(self, range_eta):
    if not isinstance(range_eta, list) or len(range_eta) != 2:
      logger.error('sampling: set_range_eta: Range list must have only two elements')
      sys.exit(1)
    self.range_eta = range_eta

  def set_ids_models(self, ids_models):
    self.ids_models = ids_models

  def set_ids_rot(self, ids_rot):
    self.ids_rot = ids_rot

  def set_sample(self, sample):
    self.sample = sample

  def set_sample_size(self, sample_size):
    self.sample_size = sample_size

  def set_modes_id_types(self, modes_id_types):
    if not isinstance(modes_id_types, list):
      logger.error('sampling: set_modes_idt_types: Input must be a list of integers from grid.sql')
      sys.exit(1)
    self.modes_id_types = modes_id_types

  def set_modes_freq_range(self, modes_freq_range):
    if not isinstance(modes_freq_range, list) or len(modes_freq_range) != 2:
      logger.error('sampling: set_range_eta: Range list must have only two elements')
      sys.exit(1)
    self.modes_freq_range = modes_freq_range

  def set_training_percentage(self, percentage):
    self.training_percentage = percentage

  def set_training_size(self, size):
    self.training_size = size 

  def set_cross_valid_percentage(self, percentage):
    self.cross_valid_percentage = percentage

  def set_cross_valid_size(self, size):
    self.cross_valid_size = size 

  def set_test_percentage(self, percentage):
    self.test_percentage = percentage

  def set_test_size(self, size):
    self.test_size = size 

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    """
    General-purpose method to get the value of a canonical attribute of the object
    E.g.

    >>>MySample = MyProblem.get('sample')

    @param attr: the name of the available attribute of the class
    @type attr: string
    @return: the value of the attribute
    @rtype: float
    """
    if not hasattr(self, attr):
      logger.error('sampling: get: The attribute "{0}" is undefined'.format(attr))
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def build_learning_set(self):
    _build_learning_sets(self)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _build_learning_sets(self):
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
  # Sanity checks ...
  if not self.dbname:
    logger.error('_build_learning_sets: specify "dbname" attribute of the class')
    sys.exit(1)

  if self.sampling_func is None:
    logger.error('_build_learning_sets: specify "sampling_func" attribute of the class')
    sys.exit(1)

  if self.max_sample_size == 0:
    logger.error('_build_learning_sets: set "max_sample_size" attribute of the class greater than zero')
    sys.exit(1)

  # Get the list of tuples for the (id_model, id_rot) to fetch model attributes
  # tups_ids   = sampling_func(*sampler_args)  
  if self.sampling_func is constrained_pick_models_and_rotation_ids:

    if not self.range_log_Teff or not self.range_log_g or not self.range_eta:
      logger.error('_build_learning_sets: specify "ranges" properly')
      sys.exit(1)

    tups_ids = constrained_pick_models_and_rotation_ids(dbname=self.dbname,
                    n=self.max_sample_size, range_log_Teff=self.range_log_Teff,
                    range_log_g=self.range_log_g, range_eta=self.range_eta)

    logger.info('_build_learning_sets: constrained_pick_models_and_rotation_ids() succeeded')

  elif self.sampling_func is randomly_pick_models_and_rotation_ids:
    tups_ids = randomly_pick_models_and_rotation_ids(dbname=self.dbname, n=self.max_sample_size)

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
  self.set_ids_models( [tup[0] for tup in tups_ids] )
  self.set_ids_rot( [tup[1] for tup in tups_ids] )
  self.set_sample_size( len(self.ids_models) )

  # convert the rotation ids to actual eta values through the look up dictionary
  dic_rot    = db_lib.get_dic_look_up_rotation_rates_id(self.dbname)

  # reverse the key/values of the dic, so that the id_rot be the key, and eta the values
  # also, the eta values are floats which are improper to compare. Instead, we convert
  # eta values to two-decimal point string representation, and do the conversion like that
  dic_rot_inv= {}
  for key, val in dic_rot.items():
    str_eta  = '{0:.2f}'.format(key[0])
    dic_rot_inv[(val, )] = str_eta
  # create a 1-element tuple of eta values in f4 format to be stiched to a tuple of 
  # other attributes below
  eta_vals   = [ ( np.float32(dic_rot_inv[(id_rot,)]), ) for id_rot in self.ids_rot ]
  logger.info('_build_learning_sets: all eta values successfully collected')

  # Even if one model id is passed (repeated) several times in the following query, only the first
  # occurance is effective. Therefore, the size of the returned results from the following query
  # is a factor (len(set(ids_rot))) larger than the result of the query. Then, the problem of 1-to-1
  # matching is resolved by setting up a look-up dictionary
  the_query  = query.get_M_ini_fov_Z_logD_Xc_from_models_id(self.ids_models)
  
  with db_def.grid_db(dbname=self.dbname) as the_db:
    the_db.execute_one(the_query, None)
    params   = the_db.fetch_all()
    n_par    = len(params)
    if n_par == 0:
      logger.error('_build_learning_sets: Found no matching model attributes')
      sys.exit(1)
    else:
      logger.info('_build_learning_sets: Fetched "{0}" unique models'.format(n_par))
    
    # look-up dictionary
    dic_par  = {}
    for tup in params:
      key    = (tup[0], )   # i.e. models.id
      val    = tup[1:]      # i.e. (M_ini, fov, Z, logD, Xc)
      dic_par[key] = val
    logger.info('_build_learning_sets: Look up dictionary for models is built')

  reconst    = [dic_par[(key, )] for key in self.ids_models]
  dic_par    = []           # delete dic_par and release memory

  stiched    = [reconst[k] + eta_vals[k] for k in range(self.sample_size)]
  
  # Now, build the thoretical modes corresponding to each row in the sampled data
  # only accept those rows from the sample whose corresponding frequency row is useful
  # for our specific problem
  container  = []
  inds_keep  = []
  modes_dtype= [('id_model', 'int32'), ('id_rot', 'int16'), ('n', 'int16'), 
                ('id_type', 'int16'), ('freq', 'float32')]

  with db_def.grid_db(dbname=self.dbname) as the_db:
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
    for k, row in enumerate(stiched):
      id_model = self.ids_models[k]
      id_rot   = self.ids_rot[k]

      # pack all query constraints into a tuple
      tup_exec = (id_model, id_rot) + tuple(self.modes_id_types) + tuple(self.modes_freq_range)

      the_db.execute_one(exec_statement, tup_exec)
      this     = the_db.fetch_all()

      rec_this = utils.list_to_recarray(this, modes_dtype)

      # do the trimming of bad models (wrong num freq in the range etc)
      # decide on keeping this index k for the trainig or not 
      # trim stiched


  # Next, pack the surviving columns 
  col_dtype  = [('M_ini', 'f4'), ('fov', 'f4'), ('Z', 'f4'), ('logD', 'f4'), 
                ('Xc', 'f4'), ('eta', 'f4')] 
  rec        = utils.list_to_recarray(stiched, col_dtype)
  reconst    = []           # destroy the list, and free up memory
  stiched    = []           # destroy the list, and free up memory

  self.set_sample(rec)
  self.set_sample_size(len(rec))
  logger.info('_build_learning_sets: the attributes sampled successfully')


  return None

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  S A M P L I N G   T H E   I N P U T   M O D E L S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def constrained_pick_models_and_rotation_ids(dbname, n, 
               range_log_Teff=[3.5, 5], range_log_g=[0, 5], range_eta=[0, 51]):
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
  @param range_log_Teff: the lower and upper range of log_Teff to scan the grid. Default: [3.5, 5]
  @type range_log_Teff: list/tuple
  @param range_log_g: the lower and upper range of log_g to scan the grid. Default: [0, 5]
  @type range_log_g: list/tuple
  @param range_eta: The range of rotation rates (in percentage w.r.t to critical, e.g. 15). 
         Default: [0, 50]
  @type range_eta: list/tuple
  @return: a shuffled list of 2-element tuples, with the first element being the model id, and the
         second element being the rotation_rate id.
  @rtype: list of tuples
  """
  if n < 1:
    logger.error('constrained_pick_models_and_rotation_ids: Specify n > 1')
    sys.exit(1)

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

  # if n_mod * n_rot > n: n = n_mod * n_rot

  np.random.shuffle(ids_models)
  np.random.shuffle(ids_rot)

  combo      = [] 
  for id_rot in ids_rot:
    combo.extend( [(id_model, id_rot) for id_model in ids_models] )

  return combo[:n]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def randomly_pick_models_and_rotation_ids(dbname, n):
  """
  Return a randomly-selected models together with their rotation rates from the grid.
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
  if n < 1:
    logger.error('randomly_pick_models_and_rotation_ids: Specify n > 1')
    sys.exit(1)

  # Retrieve two look up dictionaries for the models table and the rotation table
  t1         = time.time()
  dic_models = db_lib.get_dic_look_up_models_id(dbname_or_dbobj=dbname)
  dic_rot    = db_lib.get_dic_look_up_rotation_rates_id(dbname_or_dbobj=dbname)
  t2         = time.time()
  print 'Fetching two look up dictionaries took {0:.2f} sec'.format(t2-t1)

  ids_models = np.array([dic_models[key] for key in dic_models.keys()], dtype=np.int32)
  ids_rot    = np.array([dic_rot[key] for key in dic_rot.keys()], dtype=np.int16)
  t3         = time.time()
  print 'List comprehensions took {0:.2f} sec'.format(t3-t2)

  n_mod      = len(ids_models)
  n_eta      = len(id_rot)
  # if n > n_mod*n_eta: n = n_mod * n_eta

  np.random.shuffle(ids_models)
  np.random.shuffle(ids_rot)
  t4         = time.time()
  print 'Shuffling took {0:.2f} sec'.format(t4-t3)
  combo      = []
  for id_rot in ids_rot:
    combo.extend( [(id_model, id_rot) for id_model in ids_models] )

  t5         = time.time()
  print 'The combo list took {0:.2f} sec'.format(t5-t4)

  print 'Total time spent is {0:.2f} sec'.format(t5-t1)
  logger.info('randomly_pick_models_and_rotation_ids: Total time spent is {0:.2f} sec'.format(t5-t1))

  return combo[:n]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

