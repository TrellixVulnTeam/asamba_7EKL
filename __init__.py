from __future__ import print_function
from __future__ import unicode_literals

import sys
try:
  import h5py
except ImportError:
  print('ImportError: Please try installing the "h5py" Python module first:')
  print('http://docs.h5py.org/en/latest/build.html')
  sys.exit(1)
try:
  import psycopg2
except ImportError:
  print('ImportError: Please try installing psycopg2 first:')
  print('<https://pypi.python.org/pypi/psycopg2>')
  sys.exit(1)
