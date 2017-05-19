
"""
This module provides basic classes and methods to construct a star (given its observed quantities), 
and construct the list of observed modes (including various features) if the star is pulsating. Other
modules (e.g. sampling) inherit the "mode" and "star" objects from this moduele.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from builtins import object
from past.utils import old_div
import sys, os, glob
import logging
import time
import itertools
import numpy as np 

from asamba import utils, read, db_def, db_lib, query

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
Conversion between various frequency and time units within CGS units
"""
d_to_sec  = 24.0 * 3600.
sec_to_d  = old_div(1.0, d_to_sec)

Hz_to_uHz = 1e6
uHz_to_Hz = 1e-6

cd_to_Hz  = old_div(1.0, d_to_sec)
cd_to_uHz = cd_to_Hz * Hz_to_uHz
Hz_to_cd  = old_div(1.0, cd_to_Hz)
uHz_to_cd = old_div(1.0, cd_to_uHz)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# M O D E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class mode(object):
  """
  Container for a single pulsation mode of a star. E.g. 

  >>>from star import mode
  >>>mode_1 = star.mode()
  >>>mode_1.set('freq_unit', 'cd')

  Note that the frequency unit is converted to "per day" if not already in this unit. This is convenient
  because the typical frequencies of massive stars are around a day, and further feature normalization
  is not really needed for frequencies. Whereas with Hertz, there is roughly a factor of 10^{-6} always
  hanging around without any added value.
  """
  def __init__(self):
    super(mode, self).__init__()

    # Mode frequency
    self.freq = 0
    # Error on mode frequency
    self.freq_err = 0
    # Unit of the mode frequency
    self.freq_unit = ''

    # Mode amplitude
    self.amplitude = 0
    # Error on mode amplitude
    self.amplitude_err = 0
    # Unit of the mode amplitude
    self.amplitude_unit = ''

    # Radial order (i.e. n_pg = n_p - n_g), negative for g-modes
    self.n = -999
    # Degree of the mode
    self.l = -999
    # Azimuthal order of the mode
    self.m = -999
    # Is this mode a confirmed p-mode?
    self.p_mode = False
    # Is this mode a confirmed g-mode?
    self.g_mode = False
    # Does this mode form a frequency-spacing (df) series? 
    self.in_df = False
    # Does this mode form a period-spacing (dP) series?
    self.in_dP = False

  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    if not hasattr(self, attr):
      logger.error('mode: set: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)
    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('mode: get: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)

    return getattr(self, attr)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class modes(mode):

  def __init__(self):
    super(modes, self).__init__()

    self.modes = []
    self.num_modes = 0
  
  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    super(modes, self).set(attr, val)
    if not hasattr(self, attr):
      logger.error('modes: set: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)
    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('mode: get: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)

    try:
      super(modes, self).get(attr)
    except:
      return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def load_modes_from_file(self, filename, delimiter=''):
    """
    For detailed documentation, please refer to the function read.modes_from_file()
    """
    modes = read.modes_from_file(filename=filename, delimiter=delimiter)
    n     = len(modes)
    self.set('modes', modes)
    self.set('num_modes', n)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# S T A R 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class star(modes):
  """
  Container for all possible observables of a star, and the uncertainties, if available. This class
  inherits the "mode()" class.
  """
  def __init__(self):
    super(star, self).__init__()

    #.............................
    # Basic/General Info
    #.............................
    # Star name, e.g. Sirius
    self.name = ''
    # Other given names, e.g. HD number, KIC number, TYCO number, etc.
    self.other_names = ['']
    # Is this a member of a binary system?
    self.is_binary = False
    # Spectroscopic designation
    self.spectral_type = ''
    # Luminosity class
    self.luminosity_class = ''
    # Magnetic star?
    self.is_magnetic = False
    # Magnetic field strength in Gauss
    self.B_mag = 0       
    # Error in magnetic field strength in Gauss
    self.B_mag_err = 0
    # Variability type
    self.variability_type = ''

    #.............................
    # Global properties
    #.............................
    # Effective temperature (K)
    self.Teff = 0
    # Lower error on Teff
    self.Teff_err_lower = 0
    # Upper error on Teff
    self.Teff_err_upper = 0
    # logarithm of effective temperature
    self.log_Teff = 0
    # Lower error on log_Teff
    self.log_Teff_err_lower = 0
    # Upper error on log_Teff
    self.log_Teff_err_upper = 0

    # Surface gravity (m/sec^2)
    self.log_g = 0
    # Lower error on log_g
    self.log_g_err_lower = 0
    # upper error on log_g
    self.log_g_err_upper = 0

    # Parallax in milli arc seconds
    self.parallax = 0
    # Error on parallax
    self.parallax_err = 0

    # Bolometric luminosity
    self.luminosity = 0
    # Error on luminosity
    self.luminosity_err = 0

    # Projected rotational velocity (km / sec)
    self.v_sini = 0
    # Error on projected rotational velocity
    self.v_sini_err = 0

    # Rotation frequency (Hz)
    self.freq_rot = 0
    # Error on rotation frequency 
    self.freq_rot_err = 0

    # Microturbulent broadening
    self.v_micro = 0
    # Macroturbulent broadening
    self.v_macro = 0

    #.............................
    # Inferred global quantities
    #.............................
    # Star mass
    self.mass = 0
    # Error on star mass
    self.mass_err = 0

    # Metallicity
    self.Z = 0
    self.Fe_H = 0
    # Error on metallicity
    self.Z_err = 0
    self.Fe_H_err = 0

    # Step overshooting parameter
    self.alpha_ov = 0
    # Error on step overshooting parameter
    self.alpha_ov_err = 0
    # Exponential overshooting parameter
    self.f_ov = 0
    # Error on exponential overshooting parameter
    self.f_ov_err = 0

    #.............................
    # Surface abundances
    #.............................
    # of Helium
    self.surface_He = 0
    self.surface_He_err = 0
    # of Carbon
    self.surface_C = 0
    self.surface_C_err = 0
    # of Nitrogen 
    self.surface_N = 0
    self.surface_N_err = 0
    # of Oxygen 
    self.surface_O = 0
    self.surface_O_err = 0

    #.............................
    # Center abundances
    #.............................
    # of H
    self.center_H = 0
    self.center_H_err = 0
    # of He
    self.center_He = 0
    self.center_He_err = 0

    #.............................
    # Extra Information
    #.............................
    # Does this star has a PI as part of a main project?
    self.principal_investigator = ''
    # Literature references, e.g. ['Smith et al. (2000)', 'Jones et al. (2001)']
    self.references = ['']
    # Hyperlink to relevant publications or Simbad pages, etc
    self.url = ['']

  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    """ Setter """
    if not hasattr(self, attr):
      logger.error('star: set: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)
    setattr(self, attr, val)
    _do_extra(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    """ Getter """
    if not hasattr(self, attr):
      logger.error('star: get: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def load_star_from_inlist(self, filename):
    """
    One can set the attributes of a specific star all from an inlist. This method is used by the 
    backend module, but can be widely used in any script.

    @param self: an instance of the star.star() class 
    @type self: object
    @param filename: full path to the star inlist
    @type filename: str
    @return: None
    """
    options = read.read_inlist(filename)
    for tup in options:
      attr = tup[0]
      val  = tup[1]
      try:
        self.set(attr, val)
      except:
        logger.error('load_star_from_inlist: Failed to set (attr, val) = ({0}, {1})'.format(attr, val))
        sys.exit(1)

    logger.info('load_star_from_inlist: Succeeded.')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _do_extra(self, attr, val):
  """
  Some assignemnt to the star() object requires taking care of other attributes decently. This routine
  ensures the consistency of the assignments. E.g. if you set Teff, the log_Teff will be assigend here.
  Note: for asymmetric errors in logarithmic scale, we use the taylor expnsion of the following form

  \f[ 
      \log(x\pm\epsilon)\approx\log(x)\pm\frac{\epsilon}{(\ln10)x}-\frac{\epsilon^2}{(2\ln10)x^2}.
  \f]
  """
  if attr == 'Teff' and self.log_Teff == 0:
    self.log_Teff = np.log10(val)

  if attr == 'log_Teff' and self.Teff == 0:
    self.Teff = np.power(10, val)
    print('1')

  if attr == 'Teff_err_lower' and self.log_Teff_err_lower == 0:
    if self.Teff == 0:
      logger.error('_do_extra: Specify Teff first')
      sys.exit(1)
    x, eps, ln10 = self.Teff, val, np.log(10.)
    self.log_Teff_err_lower = old_div(eps,(ln10 * x)) + old_div(eps**2,(2*ln10*x**2))

  if attr == 'Teff_err_upper' and self.log_Teff_err_upper == 0:
    if self.Teff == 0:
      logger.error('_do_extra: Specify Teff first')
      sys.exit(1)
    x, eps, ln10 = self.Teff, val, np.log(10.)
    self.log_Teff_err_upper = old_div(eps,(ln10 * x)) - old_div(eps**2,(2*ln10*x**2))

  if attr == 'log_Teff_err_lower' and self.Teff_err_lower == 0:
    if self.log_Teff == 0:
      logger.error('_do_extra: Specify log_Teff and log_Teff_err_lower first')
      sys.exit(1)
    self.Teff_err_lower = np.power(10, self.log_Teff) - \
                          np.power(10, self.log_Teff - val)

  if attr == 'log_Teff_err_upper' and self.Teff_err_upper == 0:
    if self.log_Teff == 0:
      logger.error('_do_extra: Specify log_Teff and log_Teff_err_upper first')
      sys.exit(1)
    self.Teff_err_upper = np.power(10, self.log_Teff + val) - \
                          np.power(10, self.log_Teff) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

