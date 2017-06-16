
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
from asamba import machine_learning as ml
from asamba import interpolator as interp

logger = logging.getLogger(__name__)

####################################################################################
# U S E R  -  C O N T R O L L E D   P A R A M E T E R S :
# B A C K E N D    O B J E C T S   T H A T   D O   T H E   R E A L   W O R K
####################################################################################

class ModellingSession(interp.interpolation, ml.learner, smpl.sampling, star.star):
  """
  The ModellingSession is a derived class from the underlying modules in the package. 
  Concretely, the parent classes which are used here are below, in the following "Method
  Resolution Order (MRO)":

    - interpolator.interpolation
    - machine_learning.learner
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
  ex_lines += ' - See the template in <asamba>/data/input_templates/pulsation.freq'
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
def read_star_inlist(filename):
  """
  Read the star inlist, and load the available information to the BackEndSession object
  """  
  BackEndSession.load_star_from_inlist(filename)

####################################################################################
def get_example_star_inlist():
  """
  Return a long string that gives an example of how the star parameter inlist file must be structured
  @return: example text
  @rtype: str
  """

  ex_lines =  '\n'
  ex_lines += 'name = "beta Cephei" \n'
  ex_lines += 'Teff = 27e+3 \n'
  ex_lines += 'Teff_err_lower = 450. \n' 
  ex_lines += 'Teff_err_upper = 450. \n'
  ex_lines += 'log_g = 4.05 \n'
  ex_lines += 'log_g_err_lower = 4.05 \n'
  ex_lines += 'log_g_err_upper = 4.05 \n'
  ex_lines += 'Z = 0.0132 \n'
  ex_lines += 'Z_err = 0.0025 \n'
  ex_lines += 'mass = 12.2 \n'
  ex_lines += 'mass_err = 0.4 \n'
  ex_lines += 'log_L = 4.18 \n'
  ex_lines += 'references="Nieva & Przybilla (2014, A&A)"\n'
  ex_lines += '... \n \n'
  ex_lines += 'Notes: \n'
  ex_lines += ' - See the valid variables in <asamba>/data/input_templates/parameters.star. \n'
  ex_lines += ' - The valid variables are also attributes of the star.star() class \n'
  ex_lines += '   (see documentations in star.py).\n'
  ex_lines += ' - The units of the physical quantities are either in CGS, or w.r.t. to the Sun \n'
  ex_lines += '\n'

  return ex_lines

####################################################################################
def read_sampling_inlist(filename):
  """
  Read the sampling inlist, and load the instructions to the BackEndSession object
  """ 
  BackEndSession.load_sampling_from_inlist(filename)

####################################################################################
def get_example_sampling_inlist():
  """
  Return a long string that gives an example of how the sampling inlist file must be structured
  @return: example text
  @rtype: str
  """
  ex_lines =  '\n'
  ex_lines += 'use_constrained_sampling = True \n'
  ex_lines += 'sampling_shuffle = True \n'
  ex_lines += 'max_sample_size = 5000 \n'
  ex_lines += 'range_log_Teff = [3.95, 4.11] \n'
  ex_lines += 'range_log_g = [3.9, 4.3] \n'
  ex_lines += 'range_eta = [8, 32] \n'
  ex_lines += 'exclude_eta_column = False \n'
  ex_lines += 'modes_id_types = [2] \n'
  ex_lines += 'search_strictly_for_dP = True \n'
  ex_lines += 'trim_delta_freq_factor = 0.25 \n'
  ex_lines += 'training_percentage = 0.80 \n'
  ex_lines += 'cross_valid_percentage = 0.15 \n'
  ex_lines += 'test_percentage = .05 \n'
  ex_lines += '... \n \n'
  ex_lines += 'Notes: \n'
  ex_lines += ' - See the valid variables in <asamba>/data/input_templates/instructions.sampling. \n'
  ex_lines += ' - Some of the valid variables are attributes of the sampler.sampling() class \n'
  ex_lines += ' - This inlists accepts Boolean (True/False), integer, and float inputs, in addition \n'
  ex_lines += '   to a list or tuple of values. \n'
  ex_lines += ' - An example of input lists is "range_eta = [8, 32]."\n'
  ex_lines += ' \n'

  return ex_lines

####################################################################################
def do_call_build_learning_set():
  """ This is a basic wrapper around the sampler method: build_learning_set() """
  BackEndSession.build_learning_set()

####################################################################################
def get_samp_results():
  """ Grab several useful information after the learning set is built """
  lx    = BackEndSession.learning_x
  ly    = BackEndSession.learning_y
  tx    = BackEndSession.training_x
  ty    = BackEndSession.training_y
  cvx   = BackEndSession.cross_valid_x
  cvy   = BackEndSession.cross_valid_y
  tsx   = BackEndSession.test_x
  tsy   = BackEndSession.test_y
  tp    = BackEndSession.training_percentage
  cvp   = BackEndSession.cross_valid_percentage
  tsp   = BackEndSession.test_percentage
  flag1 = BackEndSession.learning_done
  flag2 = BackEndSession.training_set_done
  flag3 = BackEndSession.cross_valid_set_done
  flag4 = BackEndSession.test_set_done

  lines =  'Max. num. rows (set by user): {0} \n'.format(BackEndSession.max_sample_size)
  if flag1:
    lines += 'Number of rows retrieved: {0} \n'.format(BackEndSession.sample_size)
    lines += '\n'
    lines += 'Number of features returned: {0} \n'.format(BackEndSession.num_features)
    lines += 'Names of the feature columns: {0} \n'.format(BackEndSession.feature_names)
    lines += 'Shape of the frequency matrix: {0} \n'.format(ly.shape)
  lines += '\n'
  if flag2:
    lines += 'Percentage of rows kept for training: {0:.2f} % \n'.format(tp*100.)
    lines += 'Training set matrix X: {0}, Y: {1} \n\n'.format(tx.shape, ty.shape)
  if flag3:
    lines += 'Percentage of rows kept for cross-validation: {0:.2f} % \n'.format(cvp*100.)
    lines += 'Cross-validation set matrix X: {0}, Y: {1} \n\n'.format(cvx.shape, cvy.shape)
  if flag4:
    lines += 'Percentage of rows kept for testing: {0:.2f} % \n'.format(tsp*100.)
    lines += 'Test set matrix X: {0}, Y: {1} \n\n'.format(tsx.shape, tsy.shape)
  lines += '\n'

  return lines

####################################################################################
def do_split_sample():
  """ a wrapper around the sampler method split_learning_sets() """
  BackEndSession.split_learning_sets()

####################################################################################
def save_sampling_h5(self, filename, include_periods):
  """ a wrapper around sampler method write_sample_to_hdf5() """

  BackEndSession.write_sample_to_h5(filename=filename, include_periods=include_periods)

####################################################################################
def do_normal_eq():
  """ A wrapper around ml.solve_normal_equation() method """
  if not BackEndSession.learning_done:
    logger.warning('do_normal_eq: You must first build your learning set! Try again')
    return False

  BackEndSession.solve_normal_equation()

####################################################################################
def get_norm_eq_result():
  """ 
  Parse the results of solving the normal equation. By results, we mean the set of 
  regression parameters \f$\theta\f$ which minimize the cost function (normally the 
  chi square function). For further details, you can refer to the docmunetion below
  the following method: machine_learning.solve_normal_equation(). 
  """
  if not BackEndSession.normal_equation_done:
    logger.warning('get_norm_eq_result: You must first solve analytically! Try again')
    return False

  X_Neq = BackEndSession.get('normal_equation_features')
  names = BackEndSession.get('feature_names')
  names = ['Intercept'] + names

  lines =  'The set of attributes which minimize the cost function \n'
  for key_val in zip(names, X_Neq):
    lines += '{0}: {1:0.4f} \n'.format(key_val[0], key_val[1])
  lines += '\n' 
  lines += 'Cost function: J(theta) = {0:0.2e} \n'.format(BackEndSession.normal_equation_cost)

  return lines

####################################################################################
# def set_obs_log_Teff(val, err):
#   """
#   Set using the observed effective temperature 
#   """
#   # bk_star.set('log_Teff', val)
#   # bk_star.set('log_Teff_err_lower', err)
#   # bk_star.set('log_Teff_err_upper', err)
#   BackEndSession.set('log_Teff', val)
#   BackEndSession.set('log_Teff_err_lower', err)
#   BackEndSession.set('log_Teff_err_upper', err)

####################################################################################
# def set_obs_log_g(val, err):
#   """
#   Set using the observed surface gravity
#   """
#   # bk_star.set('log_g', val)
#   # bk_star.set('log_g_err_lower', err)
#   # bk_star.set('log_g_err_upper', err)
#   BackEndSession.set('log_g', val)
#   BackEndSession.set('log_g_err_lower', err)
#   BackEndSession.set('log_g_err_upper', err)

####################################################################################
# def set_sampling_function(choice):
#   """
#   Set the one of the two sampling functions from the sampler module. True means choosing
#   the "sampler.constrained_pick_models_and_rotation_ids" function and False means 
#   selecting "sampler.randomly_pick_models_and_rotation_ids"
#   """
#   if choice is True:
#     BackEndSession.set('sampling_func', smpl.constrained_pick_models_and_rotation_ids)
#   else:
#     BackEndSession.set('sampling_func', smpl.randomly_pick_models_and_rotation_ids)

####################################################################################
# def set_shuffling(choice):
#   """
#   Set the sampling shuffling mode. choice=True means apply the shuffling of the learning
#   set, and False means otherwise.
#   """
#   # bk_sample.set('sampling_shuffle', choice)
#   BackEndSession.set('sampling_shuffle', choice)

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





