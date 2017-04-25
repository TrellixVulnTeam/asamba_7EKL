#! /usr/bin/python

import sys, os, glob
import logging
import numpy as np

# from test_unit_sampling import main as tus_main
import sampler
from test_unit_ann import main as tun_main 
import interpolator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def main():

  # Easy/fast way to provide the dependencies
  TheANN    = tun_main()
  # TheSample = TheANN.get('sampling')


  # Get an instance of the interpolation class to work with
  TheInterp = interpolator.interpolation() 
  TheInterp.set('dbname', TheANN.get('dbname'))

  # Insert few attributes where the order of insertion matters
  TheInterp.set('Teff', TheANN.get('Teff'))
  TheInterp.set('Teff_err_lower', TheANN.get('Teff_err_lower'))
  TheInterp.set('Teff_err_upper', TheANN.get('Teff_err_upper'))
  TheInterp.set('log_g', TheANN.get('log_g'))
  TheInterp.set('log_g_err_lower', TheANN.get('log_g_err_lower'))
  TheInterp.set('log_g_err_upper', TheANN.get('log_g_err_upper'))

  # Inserting all attributes through a dictionary
  # dic_intrp = TheANN.__dict__
  samp_dic  = sampler.sampling().__dict__
  samp_keys = samp_dic.keys()
  for key in samp_keys:
    TheInterp.set(key, TheANN.get(key))
  # for key, val in dic_intrp.items(): TheInterp.set(key, val)
  
  TheInterp.set('inputs_around_anchor', True)
  TheInterp.set('inputs_around_anchor_M_ini_n', 5)
  TheInterp.set('inputs_around_anchor_fov_n', 7)
  TheInterp.set('inputs_around_anchor_Z_n', 3)
  TheInterp.set('inputs_around_anchor_logD_n', 10)
  TheInterp.set('inputs_around_anchor_Xc_n', 20)
  TheInterp.set('inputs_around_anchor_eta_n', 0)

  TheInterp.set('anchor_param_names', TheANN.get('feature_names'))
  TheInterp.set('anchor_param_values', TheANN.get('marginal_features'))

  TheInterp.set('anchor_frequencies', TheANN.get('MAP_frequencies'))
  TheInterp.set('anchor_radial_orders', TheANN.get('MAP_radial_orders'))

  TheInterp.do_interpolate()

  return TheInterp

if __name__ == '__main__':
  status = main()
  sys.exit(status)

