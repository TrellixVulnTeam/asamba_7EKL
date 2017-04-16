
"""
This backend serves as a facade between the underlying functionalities built around the grid database, 
and the user's frontend (GUI). The idea is that the user uses the mouse and keybord to specify the inputs;
then, those inputs are immediately communicated to the backend. The backend imports the "grid", and passes
the user's choices to the underlying functions, and calls them properly. There is a huge potential of 
extention here, which can be provided gradually as new needs emerge.
"""

import sys, os, glob
import logging
import numpy as np 
from grid import *

def connection_manager(event):
  print 'hello from connection manager: var={0}'.format(dir(event))
  # print but_conn.get()
  # for attr in dir(event):
  #   print attr, getattr(event, attr)