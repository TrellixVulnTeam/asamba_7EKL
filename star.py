
"""

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

  dic['list_dic_freq'] = [{'freq':0, 'freq_err':0, 'freq_unit':'cd', 'l':0, 'm':0, 'pg':''}] # see HD50230 for an example

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# M O D E
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class mode(object):
  """
  Container for a single pulsation mode of a star
  """
  def __init__(self):
    super(ClassName, self).__init__()

    # Mode frequency
    self.freq = 0
    # Error on mode frequency
    self.freq_err = 0
    # Unit of the mode frequency
    self.freq_unit = ''
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
  def setter(self, attr):
    if not hasattr(self, attr):
      logger.error('mode: setter: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)
    setattr(self, attr)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('mode: get: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)

    return getattr(self, attr)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# S T A R 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class star(object):
  """
  Container for all possible observables of a star, and the uncertainties, if available
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
    # Asteroseismic properties
    #.............................
    self.list_modes = [mode()]

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
  def setter(self, attr):
    if not hasattr(self, attr):
      logger.error('star: setter: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)
    setattr(self, attr)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    if not hasattr(self, attr):
      logger.error('star: get: Attribute "{0}" is unavailable'.format(attr))
      sys.exit(1)

    return getattr(self, attr)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
