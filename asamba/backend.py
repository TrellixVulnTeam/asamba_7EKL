
"""
This backend serves as a facade between the underlying functionalities built around the grid database, 
and the user's frontend (GUI). The idea is that the user uses the mouse and keybord to specify the inputs;
then, those inputs are immediately communicated to the backend. The backend imports the "grid", and passes
the user's choices to the underlying functions, and calls them properly. There is a huge potential of 
extention here, which can be provided gradually as new needs emerge.
"""
from __future__ import unicode_literals

import sys, os, glob
import logging
import numpy as np 
from asamba import star, db_def 
from asamba import sampler as smpl
from asamba import artificial_neural_network as ann
from asamba import interpolator as interp

logger = logging.getLogger(__name__)

####################################################################################
# U S E R  -  C O N T R O L L E D   P A R A M E T E R S :
# B A C K E N D    O B J E C T S   T H A T   D O   T H E   R E A L   W O R K
####################################################################################

class ModellingSession(interp.interpolation, ann.neural_net, smpl.sampling, star.star):
  """
  The ModellingSession is a derived class from the underlying modules in the package. 
  Concretely, the parent classes which are used here are below, in the following "Method
  Resolution Order (MRO)":

    - interpolator.interpolation
    - artificial_neural_network.neural_net
    - sampler.sampling
    - star.star
  
  With the bundling of the above classes, we create a derived class which takes care of 
  the observatinal data, the theoretical models in the database, the interface between 
  the underlying routine and the PostgreSQL database, and the high-level machine learning
  analysis machinery.
  """

  def __init__(self):
    """ Constructor """
    super(ModellingSession, self).__init__()

  def set(self, attr, val):
    """ Setter """
    super(ModellingSession, self).set(attr, val)

  def get(self, attr):
    """ Getter """
    return super(ModellingSession, self).get(attr)

####################################################################################


####################################################################################
# B A C K E N D   F U N C T I O N S
####################################################################################

####################################################################################
def do_connect(dbname):
  """
  Make a trial attempt to the connection port, passed as "dbname", and assert if the 
  connection is possible (returns True) or not (returns False). If successful, we set
  the connection name in the backend instance of the sampling() class.

  @param dbname: The full name of the connection port, e.g. 'grid' for local machine. 
         This value is passed by the frontend.GUI.dbname attribute
  @type dbname: str
  @return: True if the connection is possible, and False, otherwise
  @rtype: bool
  """
  if not isinstance(dbname, str):
    logger.error('do_connect: The input argument must be a string')
    sys.exit(1)

  if db_def.exists(dbname):
    # bk_sample.set('dbname', dbname)
    BackEndSession.set('dbname', dbname)
    return True
  else:
    return False

####################################################################################
def set_input_freq_file(filename):
  """
  Set the modes file for reading by star.load_modes_from_file()
  @param filename: full path to the local frequency list file
  @type filename: str
  """
  if not os.path.exists(filename):
    logger.error('set_input_freq_file: The file "{0}" not found'.format(filename))
    sys.exit()

  # modes = star.load_modes_from_file(filename, delimiter=',')
  # bk_star.set('modes', modes)
  # BackEndSession.set('modes', modes)
  BackEndSession.load_modes_from_file(filename, delimiter=',')

####################################################################################
def get_example_input_freq():
  """
  Return a long string that gives an example of how the input frequency list must be structured
  @return: example text
  @rtype: str
  """

  ex_lines =  '\n'
  ex_lines += 'amplitude, freq,  freq_err, freq_unit, l,   m,   g_mode, in_dP, p_mode, in_df \n'
  ex_lines += 'float,          float, float,         str,     int,    int, bool,   bool,  bool,   bool \n'
  ex_lines += '148.7,        2.472, 0.019,    cd,        0,   0,   0,        0,     1,      0 \n'
  ex_lines += '162.6 ,       3.086, 0.021,    cd,        1,   1,   0,        0,     1,      1 \n'
  ex_lines += '218.4 ,       0.986, 0.016,    cd,        1,   0,   1,        1,     0,      0 \n'
  ex_lines += '... \n\n\n'
  ex_lines += 'Notes: \n' 
  ex_lines += ' - The input must be an ASCII machine-readable file. \n'
  ex_lines += ' - All fields are comma-delimited. This is a mandatory format. \n'
  ex_lines += ' - The first line gives the column names. \n'
  ex_lines += ' - The second line gives the format of the corresponding column. \n'
  ex_lines += ' - The amplitude information (first column) can be left with zeros. \n'
  ex_lines += ' - The preferred frequency unit is "per day" noted as "cd" as a string. \n'
  ex_lines += ' - The degree (l) and azimuthal order (m) of the modes are integers. \n'
  ex_lines += ' - g_mode? if yes, then insert "1", else insert "0". \n'
  ex_lines += ' - Is this mode part of a g-mode period spacing? If so put "1", else put "0". \n'
  ex_lines += ' - p_mode? if yes, then insert "1", else insert "0". \n'
  ex_lines += ' - Is this mode part of a p-mode frequency spacing? If so put "1", else put "0". \n'
  ex_lines += ' - The last four columns have type "bool" but given values 0/1. The values are \n'
  ex_lines += '   internally converted to True/False using the "bool" operator. \n'
  ex_lines += '\n'
  ex_lines += 'What do the three example lines above mean? \n'
  ex_lines += ' - The first one is a radial mode. \n'
  ex_lines += ' - The second one is a dipole radial p-mode, which is also part of a frequency \n'
  ex_lines += '   spacing series. \n'
  ex_lines += ' - The last one is a dipole zonal g-mode, which is a member of a period \n'
  ex_lines += '   spacing series. \n'
  ex_lines += '\n'

  return ex_lines

####################################################################################
def set_obs_log_Teff(val, err):
  """
  Set using the observed effective temperature 
  """
  # bk_star.set('log_Teff', val)
  # bk_star.set('log_Teff_err_lower', err)
  # bk_star.set('log_Teff_err_upper', err)
  BackEndSession.set('log_Teff', val)
  BackEndSession.set('log_Teff_err_lower', err)
  BackEndSession.set('log_Teff_err_upper', err)

####################################################################################
def set_obs_log_g(val, err):
  """
  Set using the observed surface gravity
  """
  # bk_star.set('log_g', val)
  # bk_star.set('log_g_err_lower', err)
  # bk_star.set('log_g_err_upper', err)
  BackEndSession.set('log_g', val)
  BackEndSession.set('log_g_err_lower', err)
  BackEndSession.set('log_g_err_upper', err)

####################################################################################
def set_sampling_function(choice):
  """
  Set the one of the two sampling functions from the sampler module. True means choosing
  the "sampler.constrained_pick_models_and_rotation_ids" function and False means 
  selecting "sampler.randomly_pick_models_and_rotation_ids"
  """
  if choice is True:
    BackEndSession.set('sampling_func', smpl.constrained_pick_models_and_rotation_ids)
  else:
    BackEndSession.set('sampling_func', smpl.randomly_pick_models_and_rotation_ids)

####################################################################################
def set_shuffling(choice):
  """
  Set the sampling shuffling mode. choice=True means apply the shuffling of the learning
  set, and False means otherwise.
  """
  # bk_sample.set('sampling_shuffle', choice)
  BackEndSession.set('sampling_shuffle', choice)

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################


####################################################################################
# B A C K E N D   W O R K I N G   S E S S I O N
####################################################################################
BackEndSession = ModellingSession()
####################################################################################





