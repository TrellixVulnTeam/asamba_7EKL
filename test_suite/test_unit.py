from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
from builtins import object
import sys, os, glob
import subprocess
import logging
import time
import numpy as np 
import psycopg2

from asamba import db_def, db_lib, insert_lib, read

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Error Handling and Logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    raiseExceptions=True,
                    # filename='test_unit.log',
                    # filemode='w'
                    )
formatter = logging.Formatter('%(levelname)-8s: %(name)-12s: %(message)s')
logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class LoggerWriter(object):
  def __init__(self, level):
    # self.level is really like using log.debug(message)
    # at least in my case
    self.level = level

  def write(self, message):
    # if statement reduces the amount of newlines that are
    # printed to the logger
    if message != '\n':
      self.level(message)

  def flush(self):
    # create a flush method so things can be flushed when
    # the system wants to. Not sure if simply 'printing'
    # sys.stderr is the correct way to do it, but it seemed
    # to work properly for me.
    self.level(sys.stderr)

# sys.stdout = LoggerWriter(logger.debug)
# sys.stderr = LoggerWriter(logger.warning)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# T E S T I N G   F U N C T I O N S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_09(dbname):

  logger.info('do_test_09: test db_lib.get_dic_look_up_mode_types_id()')
  dic_types = db_lib.get_dic_look_up_mode_types_id(dbname)
  if not isinstance(dic_types, dict):
    logger.error('do_test_09: return type is not dictionary')
    sys.exit(1)

  list_ids  = [0, 1, 2, 4, 5, 6]
  list_types= [(0, 0), (1, 1), (1, 0), (2, 2), (2, 1), (2, 0)]
  n_types   = len(list_ids)
  ind       = list(range(n_types))
  n_try     = 10
  results   = []

  for i_try in range(n_try):
    np.random.shuffle(ind)
    try_ids = [list_ids[j] for j in ind]
    try_types = [list_types[j] for j in ind]

    for i_type in range(n_types):
      key   = try_types[i_type]
      val   = try_ids[i_type]
      try:
        assert val == dic_types[key]
        results.append(True)
      except AssertionError:
        results.append(False)

  success = all(results)
  if success:
    logger.info('do_test_09: All tests passed')
  else:
    logger.error('do_test_09: At least one test with db_lib.get_dic_look_up_mode_types_id() failed')

  return success 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_08(dbname):

  logger.info('do_test_08: test db_lib.get_dic_look_up_rotation_rates_id()')
  dic_rot = db_lib.get_dic_look_up_rotation_rates_id(dbname)
  if not isinstance(dic_rot, dict):
    logger.error('do_test_08: return type is not dictionary')
    sys.exit(1)

  n_rot    = 11
  list_id  = np.arange(1, n_rot+1)
  list_eta = [(i*5.0, ) for i in range(n_rot)]
  n_try    = 10
  results  = []

  for i_try in range(n_try):
    ind    = list(range(n_rot))
    np.random.shuffle(ind)

    try_id = list_id[ind]
    try_eta= [list_eta[k] for k in ind]

    for i_rot in range(n_rot):
      key  = try_eta[i_rot]
      val  = try_id[i_rot]
      try:
        assert val == dic_rot[key]
        results.append(True)
      except AssertionError:
        results.append(False)

  success = all(results)
  if success:
    logger.info('do_test_08: All tests passed')
  else:
    logger.error('do_test_08: At least one test with db_lib.get_dic_look_up_rotation_rates_id() failed')

  return success 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_07(dbname, list_h5):

  n_h5 = len(list_h5)
  logger.info('do_test_07: test insert_lib.insert_gyre_output_into_modes_table(): with "{0}" files'.format(n_h5))
  try:
    t0 = time.time()
    insert_lib.insert_gyre_output_into_modes_table(dbname=dbname, list_h5=list_h5, insert_every=100000)
    dt = time.time() - t0
    logger.info('do_test_07: succeeded. "{0}" files inserted in {1} sec'.format(n_h5, dt))
  except:
    logger.info('do_test_07: failed')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_06(dbname):

  logger.info('do_test_06: test db_def.grid_db.get_table()')
  with db_def.grid_db(dbname=dbname) as the_db:
    try:
      result = the_db.get_mode_types()
      logger.info('do_test_06: succeeded')
    except:
      logger.error('do_test_06: failed')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_05(dbname, ascii_in):

  logger.info('do_test_05: test insert_lib.insert_models_from_models_parameter_file()')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_04(dbname, ascii_in):

  logger.info('do_test_04: test insert_lib.insert_tracks_from_models_parameter_file()')
  insert_lib.insert_tracks_from_models_parameter_file(dbname, ascii_in)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_03(dbname):

  logger.info('do_test_03: test db_lib.get_track_by_id()')

  # first clean up the table
  cmnd       = 'delete from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  num_tests = 10
  tolerance = 1e-5
  list_tups = []
  tests     = []

  all_id    = np.random.random_integers(1, 10000, num_tests)
  all_M_ini = np.random.random(num_tests) * 35. 
  all_fov   = np.random.random(num_tests) * 0.04
  all_Z     = np.random.random(num_tests) * 0.018
  all_logD  = np.random.random(num_tests) * 8.0

  for i_test in range(num_tests):
    orig_id    = all_id[i_test]
    orig_M_ini = all_M_ini[i_test]
    orig_fov   = all_fov[i_test]
    orig_Z     = all_Z[i_test]
    orig_logD  = all_logD[i_test]
    tup        = (orig_id, orig_M_ini, orig_fov, orig_Z, orig_logD)
    if tup in list_tups: 
      logger.warning('do_test_03: accidentally, one input tuple is repeated: i={0}'.format(i_test))
      continue

    # insert this random row in the table, and retrieve it by its id
    cmnd       = 'insert into tracks (id, M_ini, fov, Z, logD) values (%s,%s,%s,%s,%s)'
    with db_def.grid_db(dbname=dbname) as the_db:
      the_db.execute_one(cmnd, tup)

    params     = db_lib.get_track_by_id(dbname, orig_id)
    res_M_ini  = params[0]
    res_fov    = params[1]
    res_Z      = params[2]
    res_logD   = params[3]

    check_01   = assert_approximately_equal('M_ini', orig_M_ini, res_M_ini, tolerance)
    check_02   = assert_approximately_equal('fov', orig_fov, res_fov, tolerance)
    check_03   = assert_approximately_equal('Z', orig_Z, res_Z, tolerance)
    check_04   = assert_approximately_equal('logD', orig_logD, res_logD, tolerance)
    this_test  = all([check_01, check_02, check_03, check_04])

    tests.append(this_test)


  test       = all(tests)
  if test is True:
    logger.info('do_test_03: All "{0}" checks passed'.format(num_tests))
  else:
    logger.error('do_test_03: At least one check failed')

  cmnd       = 'delete from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  return test 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_02(dbname):

  logger.info('do_test_02: test db_lib.get_track_id()')

  # first clean up the table
  cmnd       = 'delete from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  num_tests = 10
  list_tups = []
  tests     = []

  for i_test in range(num_tests):
    orig_id    = np.random.randint(1, 10000)
    orig_M_ini = np.random.random(1)[0] * 35. 
    orig_fov   = np.random.random(1)[0] * 0.04
    orig_Z     = np.random.random(1)[0] * 0.018
    orig_logD  = np.random.random(1)[0] * 8.0
    tup        = (orig_id, orig_M_ini, orig_fov, orig_Z, orig_logD)
    if tup in list_tups: 
      logger.warning('do_test_02: accidentally, one input tuple is repeated: i={0}'.format(i_test))
      continue

    # insert this random row in the table, and get back the id
    insert_lib.insert_row_into_tracks(dbname, orig_id, orig_M_ini, orig_fov, orig_Z, orig_logD)
    # retrieve the id
    the_id     = db_lib.get_track_id(dbname, M_ini=orig_M_ini, 
                        fov=orig_fov, Z=orig_Z, logD=orig_logD)
    # assert the retrieved id is the same as the original one
    try:
      assert the_id == orig_id
      logger.info('   ... Check {0} OK'.format(i_test))
      tests.append(True)
    except AssertionError:
      logger.error('   ... Check {0} failed: {1} != {2}'.format(i_test, the_id, orig_id))
      tests.append(False)

  test       = all(tests)
  if test is True:
    logger.info('do_test_02: All "{0}" checks passed'.format(num_tests))
  else:
    logger.error('do_test_02: At least one check failed')

  cmnd       = 'delete from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  return test 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_01(dbname):

  logger.info('do_test_01: Insert & retrieve a row into tracks, ensure 32bit identical')
  
  # first clean up the table
  cmnd       = 'delete from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  # original test values  
  if True:
    # using ordinary input
    orig_id    = 1001
    orig_M_ini = 12.3456
    orig_fov   = 0.12312
    orig_Z     = 0.04554
    orig_logD  = 98.7654
  else:
    # testing small and large exponential numbers
    orig_id    = 1001
    orig_M_ini = 12.3456
    orig_fov   = 1.23456e-8
    orig_Z     = 1.23456e15
    orig_logD  = -1.23456e-20

  tup        = (orig_id, orig_M_ini, orig_fov, orig_Z, orig_logD)

  cmnd       = 'insert into tracks (id, M_ini, fov, Z, logD) values (%s,%s,%s,%s,%s)'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, tup)

  cmnd       = 'select id, M_ini, fov, Z, logD from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)
    result   = the_db.fetch_one()
  res_id     = result[0]
  res_M_ini  = result[1]
  res_fov    = result[2]
  res_Z      = result[3]
  res_logD   = result[4]

  test_id    = assert_exactly_equal('id', orig_id, res_id)
  test_M_ini = assert_exactly_equal('M_ini', orig_M_ini, res_M_ini)
  test_fov   = assert_exactly_equal('fov', orig_fov, res_fov)
  test_Z     = assert_exactly_equal('Z', orig_Z, res_Z)
  test_logD  = assert_exactly_equal('logD', orig_logD, res_logD)

  test       = all([test_id, test_M_ini, test_fov, test_Z, test_logD])
  if test:
    logger.info('do_test_01: All five columns retrieved successfully')
  else:
    logger.error('do_test_01: At least one column was not retrieved successfully')

  cmnd       = 'delete from tracks'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  return test

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# C O N V E N I E N C E   &   C O M P A R I S O N   F U N C T I O N S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def assert_exactly_equal(attribute, original, retrieved):  
  """
  Define a local function to test each column value, and log the result properly
  """
  try:
    assert original == retrieved
    logger.info('        ... "{0}" exactly OK'.format(attribute))
    return True
  except AssertionError:
    logger.error('       XXX "{0}" failed'.format(attribute))
    return False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def assert_approximately_equal(attribute, original, retrieved, tolerance):
  """
  Assert the two values (original and retrived) of the attribute are identical within the round-off
  as specified by the tolerance.
  """
  try:
    assert np.abs(original - retrieved) <= tolerance
    logger.info('        ... {0} approximately OK'.format(attribute))
    return True 
  except AssertionError:
    logger.error('       XXX "{0}" failed'.format(attribute))
    # print original, retrieved, np.abs(original-retrieved)
    return False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# P O S T G R E S Q L   F U N C T I O N S 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def operator_overloading_function(dbname):
  """
  The operator " ~ " is overloaded to mean "approximately equals to", for a single-precision floating
  point comparison
  """
  with db_def.grid_db(dbname=dbname) as the_db:
    if the_db.has_function('approximately_equals_to'):
      logger.warning('operator_overloading_function: skipping: function "{0}" already exists'.format('approximately_equals_to'))
      return True

  if not db_def.exists(dbname=dbname): 
    logger.error('operator_overloading_function: Database "{0}" does not exist'.format(dbname))
    sys.exit(1)

  cmnd = 'create function approximately_equals_to  \
          (in x real, in y real) returns boolean \
          as $$ select abs(x-y) <= 1e-5*abs(x) $$ \
          language sql stable strict;'
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)
          
  cmnd = 'create operator ~ ( \
                 procedure = approximately_equals_to, \
                 leftarg = real, \
                 rightarg = real \
                 );\
          '
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(cmnd, None)

  logger.info('operator_overloading_function: the new operator "~" overloaded.')

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_table_modes(dbname):
  """
  Create the "modes" table identical to the "grid.modes" table
  """
  if not db_def.exists(dbname=dbname):
    logger.error('make_table_modes: Database "{0}" does not exist'.format(dbname))
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as my_db:
    if my_db.has_table('modes'):
      logger.warning('make_table_modes: Database "{0}" already has table "modes"'.format(dbname))
      return


  tbl = 'create table modes (\
          id             bigserial, \
          id_model       int not null, \
          id_rot         smallint not null, \
          id_type        smallint not null, \
          n              int not null, \
          freq           real not null, \
          primary key (id), \
          foreign key (id_model) references models (id), \
          foreign key (id_rot)   references rotation_rates (id), \
          foreign key (id_type)  references mode_types (id), \
          constraint positive_freq check (freq > 0) ); \
          create index index_freq_n on modes (n, freq);'

  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(tbl, None)
  logger.info('make_table_modes: the "modes" table created successfully in database "{0}"'.format(dbname))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_table_rotation_rates(dbname):
  """
  Create the "rotation_rates" table
  """
  if not db_def.exists(dbname=dbname): 
    logger.error('make_table_rotation_rates: Database "{0}" does not exist'.format(dbname))
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as my_db:
    if my_db.has_table('rotation_rates'):
      logger.warning('make_table_rotation_rates: Database "{0}" already has table "mode_types"'.format(dbname))
      return

  tbl = 'create table rotation_rates ( \
        id             serial, \
        eta            real, \
        primary key (id), \
        unique (eta), \
        constraint positive_eta check (eta >= 0) );'

  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(tbl, None)
  logger.info('make_table_rotation_rates: the "rotation_rate" table created in database "{0}".'.format(dbname))

  # static insertions
  with db_def.grid_db(dbname=dbname) as the_db:
    cmnd = 'prepare ins_rot_rat (int, real) as \
            insert into rotation_rates (id, eta) values ($1, $2)'
    the_db.execute_one(cmnd, None)

    cmnd = 'execute ins_rot_rat (%s, %s)'
    tups = [(i+1, i*5) for i in range(11)]
    the_db.execute_many(cmnd, tups)

  logger.info('make_table_rotation_rates: the insertions succeeded')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_table_mode_types(dbname):
  """
  Create the "mode_types" table identical to the "grid.mode_types" table
  """
  if not db_def.exists(dbname=dbname): 
    logger.error('make_table_mode_types: Database "{0}" does not exist'.format(dbname))
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as my_db:
    if my_db.has_table('mode_types'):
      logger.warning('make_table_mode_types: Database "{0}" already has table "mode_types"'.format(dbname))
      return

  tbl = 'create table mode_types( \
        id            serial, \
        l             int not null, \
        m             int not null, \
        primary key (id), \
        unique (l, m), \
        constraint positive_l check (l >= 0), \
        constraint bounded_m  check (m >= -l and m <= l)\
        );'

  with db_def.grid_db(dbname=dbname) as my_db:
    my_db.execute_one(tbl, None)
  logger.info('make_table_mode_types: the "mode_types" table created in database "{0}".'.format(dbname))

  # static insertions
  with db_def.grid_db(dbname=dbname) as my_db:
    cmnd = 'prepare ins_md_tp (int, int, int) as \
            insert into mode_types (id, l, m) values ($1, $2, $3)'
    my_db.execute_one(cmnd, None)

    cmnd = 'execute ins_md_tp (%s, %s, %s)'
    tups = [(0, 0, 0), (1, 1, 1), (2, 1, 0), (3, 1, -1), (4, 2, 2),
            (5, 2, 1), (6, 2, 0), (7, 2, -1), (8, 2, -2)]
    my_db.execute_many(cmnd, tups)

  logger.info('make_table_mode_types: the insertions succeeded')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_table_models(dbname):
  """
  Create the "models" table identical to the "grid.models" table
  """
  if not db_def.exists(dbname=dbname): 
    logger.error('make_table_models: Database "{0}" does not exist'.format(dbname))
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as my_db:
    if my_db.has_table('models'):
      logger.warning('make_table_models: Database "{0}" already has table "models"'.format(dbname))
      return

  tbl = 'create table models ( \
        id             serial, \
        id_track       int not null, \
        Xc             real not null, \
        model_number   int not null, \
        star_mass      real, \
        radius         real, \
        log_Teff       real, \
        log_g          real, \
        log_L          real, \
        log_Ledd       real, \
        log_abs_mdot   real, \
        mass_conv_core real, \
        star_age       real, \
        dynamic_timescale real, \
        kh_timescale   real, \
        nuc_timescale  real, \
        log_center_T   real, \
        log_center_Rho real, \
        log_center_P   real, \
        center_h1      real, \
        center_h2      real, \
        center_he3     real, \
        center_he4     real, \
        center_c12     real, \
        center_c13     real, \
        center_n14     real, \
        center_n15     real, \
        center_o16     real, \
        center_o18     real, \
        center_ne20    real, \
        center_ne22    real, \
        center_mg24    real, \
        surface_h1     real, \
        surface_h2     real, \
        surface_he3    real, \
        surface_he4    real, \
        surface_c12    real, \
        surface_c13    real, \
        surface_n14    real, \
        surface_n15    real, \
        surface_o16    real, \
        surface_o18    real, \
        surface_ne20   real, \
        surface_ne22   real, \
        surface_mg24   real, \
        delta_nu       real, \
        nu_max         real, \
        acoustic_cutoff real, \
        delta_Pg       real, \
        Mbol           real, \
        bcv            real, \
        U_B            real, \
        B_V            real, \
        V_R            real, \
        V_I            real, \
        V_K            real, \
        R_I            real, \
        I_K            real, \
        J_H            real, \
        H_K            real, \
        K_L            real, \
        J_K            real, \
        J_L            real, \
        J_Lp           real, \
        K_M            real,\
        primary key (id), \
        foreign key (id_track) references tracks (id), \
        constraint positive_Xc check (Xc >= 0), \
        check (model_number >= 0) ); \
        create index index_logTeff_logg on models (log_Teff, log_g); \
        create index index_Xc on models (Xc desc); \
        create index index_age on models (star_age asc); \
        create index index_delta_Pg on models (delta_Pg desc);'

  with db_def.grid_db(dbname=dbname) as my_db:
    my_db.execute_one(tbl, None)

  logger.info('make_table_models: the "models" table created in database "{0}".'.format(dbname))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_table_tracks(dbname):
  """
  Create the "tracks" table identical to the "grid.tracks" table
  """
  if not db_def.exists(dbname=dbname): 
    logger.error('make_table_tracks: Database "{0}" does not exist'.format(dbname))
    sys.exit(1)

  with db_def.grid_db(dbname=dbname) as my_db:
    if my_db.has_table('tracks'):
      logger.warning('make_table_tracks: Database "{0}" already has table "tracks"'.format(dbname))
      return

  tbl =  'create table tracks ( \
          id             serial, \
          M_ini          real not null, \
          fov            real not null, \
          Z              real not null, \
          logD           real not null, \
          primary key (id), \
          unique (M_ini, fov, Z, logD), \
          constraint positive_mass check (M_ini > 0), \
          constraint positive_ov check (fov >= 0), \
          constraint positive_Z check (Z > 0), \
          constraint positive_log_D check (logD >= 0) \
        ); \
        create index index_track_id on tracks (id asc);  \
        create index index_M_ini on tracks (M_ini asc); \
        create index index_fov on tracks (fov asc); \
        create index index_Z on tracks (Z asc); \
        create index index_log_D on tracks (logD asc);'

  with db_def.grid_db(dbname=dbname) as my_db:
    my_db.execute_one(tbl, None)

  logger.info('make_table_tracks: the "tracks" table created in database "{0}".'.format(dbname))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_schema(dbname):
  """
  Create a schema for the database, like the following: "create schema grid;"
  """
  if db_def.exists(dbname): return True

  schema = 'create schema {0}'.format(dbname)
  with db_def.grid_db(dbname=dbname) as the_db:
    the_db.execute_one(schema, None)
  logger.info('make_schema: "{0}" done'.format(schema))

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_test_database(dbname):
  """
  Create a new database by calling shell commands.
  Returns false if failed to create the database, and returns True if the database already existed 
  or successfully created
  """
  present = db_def.exists(dbname)
  if present:
    logger.warning('create_test_database: Database "{0}" already existed.'.format(dbname))
    return True

  cmnd    = 'createdb {0}'.format(dbname)
  execute = subprocess.Popen(cmnd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout  = execute.stdout.read().rstrip('\n')
  stderr  = execute.stderr.read()
  err     = execute.returncode
  if err is not None:
    logger.error('create_test_database: stdout: {0}'.format(stdout))
    logger.error('create_test_database: stderr: {0}'.format(stderr))
    logger.error('create_test_database: failed to create the database')
    return False
  else:
    logger.info('create_test_database: database "{0}" created'.format(dbname))
    return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def drop_test_database(dbname):
  """
  Delete/drop the test database by calling a shell command
  Returns False if fails to drop the dataase, and returns True if the database was already not there,
  or when the database is successfully dropped.
  """
  present = db_def.exists(dbname)
  if not present:
    logger.warning('drop_test_database: Database "{0}" does not exist anyway. Cannot drop it.')
    return True

  cmnd    = 'dropdb {0}'.format(dbname)
  execute = subprocess.Popen(cmnd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout  = execute.stdout.read().rstrip('\n')
  stderr  = execute.stderr.read()
  err     = execute.returncode
  if err is not None:
    logger.error('drop_test_database: stdout: {0}'.format(stdout))
    logger.error('drop_test_database: stderr: {0}'.format(stderr))
    logger.error('drop_test_database: failed to drop the database "{0}"'.format(dbname))
    return False
  else:
    logger.info('drop_test_database: successfully droped the database "{0}"'.format(dbname))
    return True


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_string(dbname):
  str_s = 4*'%s,'
  str_s = str_s[:-1]
  cmnd  = 'insert into tracks (M_ini, fov, Z, logD) values (' + str_s + ')'
  tup   = (1.1, 2.2, 3.3, 4.4)
  with db_def.grid_db(dbname) as the_db: 
    the_db.execute_one('delete from tracks', None)
    print(the_db.execute_one(cmnd, tup))

  with db_def.grid_db(dbname) as the_db: 
    print(the_db.fetch_one())

  with db_def.grid_db(dbname) as the_db: 
    the_db.execute_one('delete from tracks', None)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_gyre_h5(filename):
  logger.info('test_gyre_h5: test read.get_minimal_gyre_output()')
  modes  = read.gyre_h5(filename)
  if modes:
    logger.info('test_gyre_h5: Succeeded')
  else:
    logger.error('test_gyre_h5: Failed')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main(ascii_in):
  """
  Execute the tests one after the other

  @param ascii_in: full path to where the ASCII model parameter file is stored. This is a large file
         (~ 2-3 GB), and has >3.8 million rows, and about 65 columns.
  """
  logger.info('Main: Start the tests')

  my_db   = 'test_grid' 

  status  = create_test_database(dbname=my_db)
  if status is not True:
    logger.error('main: create_test_database failed')
    return status

  status  = make_schema(dbname=my_db)
  status  = operator_overloading_function(dbname=my_db)

  make_table_tracks(dbname=my_db)

  # status  = do_test_01(dbname=my_db)

  # status  = do_test_02(dbname=my_db)

  # status  = do_test_03(dbname=my_db)

  if ascii_in:
    status= do_test_04(dbname=my_db, ascii_in=ascii_in)

    status= do_test_05(dbname=my_db, ascii_in=ascii_in)

  # test_string(dbname=my_db)

  make_table_models(dbname=my_db)

  make_table_mode_types(dbname=my_db)

  make_table_rotation_rates(dbname=my_db)

  make_table_modes(dbname=my_db)

  status  = do_test_06(dbname=my_db)

  dir_    = '/STER/mesa-gyre/asamba-grid/'
  # dir_    = '/Users/ehsan/programs/asamba-grid/'
  dir_    += 'M35.000/eta05.00/'
  list_h5 = sorted(glob.glob(dir_ + '*.h5'))
  n_h5    = len(list_h5)

  if False:
    test_gyre_h5(list_h5[0])

  if True:
    status  = do_test_07(dbname='copy_grid', list_h5=list_h5)

  status  = do_test_08(dbname=my_db)

  status  = do_test_09(dbname=my_db)

  status  = drop_test_database(dbname=my_db)
  if status is not True:
    logger.error('main: drop_test_db failed')
    return status

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  args     = sys.argv
  if len(args) == 2:
    ascii_in = args[1]
  else:
    ascii_in = ''
  status   = main(ascii_in=ascii_in)
  sys.exit(status)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
