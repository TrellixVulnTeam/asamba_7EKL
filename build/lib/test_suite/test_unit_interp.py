#! /usr/bin/python

from __future__ import absolute_import
from __future__ import unicode_literals
import sys, os, glob
import logging
import numpy as np

from asamba import sampler, interpolator, plot_interpolator
from test_unit_ann import main as tun_main 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
  samp_keys = list(samp_dic.keys())
  for key in samp_keys:
    TheInterp.set(key, TheANN.get(key))
  # for key, val in dic_intrp.items(): TheInterp.set(key, val)
  
  TheInterp.set('inputs_around_anchor', True)
  TheInterp.set('inputs_around_anchor_M_ini_n', 6)
  TheInterp.set('inputs_around_anchor_fov_n', 7)
  TheInterp.set('inputs_around_anchor_Z_n', 3)
  TheInterp.set('inputs_around_anchor_logD_n', 10)
  TheInterp.set('inputs_around_anchor_Xc_n', 20)
  TheInterp.set('inputs_around_anchor_eta_n', 0)

  TheInterp.set('anchor_param_names', TheANN.get('feature_names'))
  TheInterp.set('anchor_param_values', TheANN.get('marginal_features'))

  TheInterp.set('anchor_frequencies', TheANN.get('MAP_frequencies'))
  TheInterp.set('anchor_radial_orders', TheANN.get('MAP_radial_orders'))

  # Set the interpolation specifications, and ranges
  TheInterp.set('interp_M_ini', True)
  TheInterp.set('interp_M_ini_from', 2.5)
  TheInterp.set('interp_M_ini_to', 4)
  TheInterp.set('interp_M_ini_steps', 21)

  TheInterp.set('interp_fov', True)
  TheInterp.set('interp_fov_from', 0.01)
  TheInterp.set('interp_fov_to', 0.03)
  TheInterp.set('interp_fov_steps', 7)

  TheInterp.set('interp_Z', True)
  TheInterp.set('interp_Z_from', 0.01)
  TheInterp.set('interp_Z_to', 0.02)
  TheInterp.set('interp_Z_steps', 5)

  TheInterp.set('interp_logD', True)
  TheInterp.set('interp_logD_from', 0.0)
  TheInterp.set('interp_logD_to', 3.0)
  TheInterp.set('interp_logD_steps', 7)

  TheInterp.set('interp_Xc', True)
  TheInterp.set('interp_Xc_from', 0.5)
  TheInterp.set('interp_Xc_to', 0.6)
  TheInterp.set('interp_Xc_steps', 11)

  if True:
    # Only prepare the inputs, and plot the frequencies as a test
    TheInterp.collect_inputs()
    TheInterp.check_inputs()
    TheInterp.prepare()

    figure_name = 'test_suite/plots/KIC-10526294-interp-wrt-M.png'
    plot_interpolator.input_frequencies_wrt(TheInterp, wrt='M_ini', figure_name=figure_name)

    # figure_name = 'test_suite/plots/KIC-10526294-interp-wrt-fov.png'
    # plot_interpolator.input_frequencies_wrt(TheInterp, wrt='fov', figure_name=figure_name)

    # figure_name = 'test_suite/plots/KIC-10526294-interp-wrt-Z.png'
    # plot_interpolator.input_frequencies_wrt(TheInterp, wrt='Z', figure_name=figure_name)

    # figure_name = 'test_suite/plots/KIC-10526294-interp-wrt-logD.png'
    # plot_interpolator.input_frequencies_wrt(TheInterp, wrt='logD', figure_name=figure_name)

    # figure_name = 'test_suite/plots/KIC-10526294-interp-wrt-Xc.png'
    # plot_interpolator.input_frequencies_wrt(TheInterp, wrt='Xc', figure_name=figure_name)

  if False:
    # Do interpolate!
    TheInterp.do_interpolate()

    # Check the outcome
    if TheInterp.get('interp_meshgrid_OK'):
      logger.info('interp_meshgrid_OK: Succeeded')
    else:
      logger.warning('interp_meshgrid_OK: Failed')

  # return "self"
  return TheInterp

if __name__ == '__main__':
  status = main()
  sys.exit(status)

