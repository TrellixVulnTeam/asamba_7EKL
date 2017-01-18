
import sys, os, glob
import logging
import numpy as np 
import subprocess
import psycopg2

from grid import db_def

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Error Handling and Logging

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    # filename='test_unit.log',
                    # filemode='w'
                    )
# define a Handler which writes INFO messages or higher to the sys.stderr
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(levelname)-8s: %(name)-12s: %(message)s')
# tell the handler to use this format
# console.setFormatter(formatter)
# # add the handler to the root logger
# logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_test_01():

  logger.info('do_test_01')

  return None

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_table_tracks(dbname):
  """
  Create the "tracks" table identical to the "grid.tracks" table
  """
  tbl =  'create table grid.tracks ( \
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
        create index index_track_id on grid.tracks (id asc);  \
        create index index_M_ini on grid.tracks (M_ini asc); \
        create index index_fov on grid.tracks (fov asc); \
        create index index_Z on grid.tracks (Z asc); \
        create index index_log_D on grid.tracks (logD asc);'

  my_db = db_def.grid_db(dbname=dbname)
  with temp_db as my_db:
    temp_db.execute_one(tbl)

  logger.info('make_table_tracks: the "tracks" table created in database "{0}".'.format(dbname))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_test_database(dbname):
  """
  Create a new database by calling shell commands.
  """
  cmnd    = 'createdb {0}'.format(dbname)
  execute = subprocess.Popen(cmnd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout  = execute.stdout.read().rstrip('\n')
  stderr  = execute.stderr.read()
  err     = execute.returncode
  if err is not None:
    logger.error('create_test_database: stdout: {0}'.format(stdout))
    logger.error('create_test_database: stderr: {0}'.format(stderr))
    logger.error('create_test_database: failed to create the database')
  else:
    logger.info('create_test_database: database "{0}" created'.format(dbname))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def drop_test_database(dbname):
  """
  Delete/drop the test database by calling a shell command
  """
  cmnd    = 'dropdb {0}'.format(dbname)
  execute = subprocess.Popen(cmnd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout  = execute.stdout.read().rstrip('\n')
  stderr  = execute.stderr.read()
  err     = execute.returncode
  if err is not None:
    logger.error('drop_test_database: stdout: {0}'.format(stdout))
    logger.error('drop_test_database: stderr: {0}'.format(stderr))
    logger.error('drop_test_database: failed to drop the database "{0}"'.format(dbname))
  else:
    logger.info('drop_test_database: successfully droped the database "{0}"'.format(dbname))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
  """
  Execute the tests one after the other
  """
  logger.info('Main: Start the tests')

  my_db   = 'test_grid' 

  create_test_database(dbname=my_db)
  make_table_tracks(dbname=my_db)

  # do_test_01()
  drop_test_database(dbname=my_db)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  status = main()
  sys.exit(status)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
