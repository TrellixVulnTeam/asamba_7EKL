
import sys, os, glob
import subprocess
import logging
import numpy as np 
import psycopg2

from grid import db_def, db_lib, insert_lib

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Error Handling and Logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    # filename='test_unit.log',
                    # filemode='w'
                    )
formatter = logging.Formatter('%(levelname)-8s: %(name)-12s: %(message)s')
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# T E S T I N G   F U N C T I O N S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
  orig_id    = 1001
  orig_M_ini = 12.3456
  orig_fov   = 0.12312
  orig_Z     = 0.04554
  orig_logD  = 98.7654
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
    print the_db.execute_one(cmnd, tup)

  with db_def.grid_db(dbname) as the_db: 
    print the_db.fetch_one()

  with db_def.grid_db(dbname) as the_db: 
    the_db.execute_one('delete from tracks', None)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
  """
  Execute the tests one after the other
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

  status  = do_test_01(dbname=my_db)

  status  = do_test_02(dbname=my_db)

  status  = do_test_03(dbname=my_db)

  # test_string(dbname=my_db)

  status  = drop_test_database(dbname=my_db)
  if status is not True:
    logger.error('main: drop_test_db failed')
    return status

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  status = main()
  sys.exit(status)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
