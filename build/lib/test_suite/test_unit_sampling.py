#! /usr/bin/python

from __future__ import print_function
from __future__ import unicode_literals
import sys, os, glob
import logging
import numpy as np 

from asamba import sampler
from asamba import star
from asamba import plot_sampler

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():

  print(' - Load the mode list from a file')
  mode_file = 'asamba/stars/KIC_10526294.freq'

  print(' - Get an instance of the "sampling" class.')
  TheSample = sampler.sampling()

  print(' - Attach the modes to a star object')
  # TheStar   = star.star()
  TheSample.set('name', 'KIC_10526294')

  TheSample.set('Teff', 11500.)
  TheSample.set('Teff_err_lower', 500.)
  TheSample.set('Teff_err_upper', 500.)
  TheSample.set('log_g', 4.1)
  TheSample.set('log_g_err_lower', 0.2)
  TheSample.set('log_g_err_upper', 0.2)

  # TheSample.set('modes', modes)
  TheSample.load_modes_from_file(filename=mode_file, delimiter=',')

  TheSample.set('dbname', 'grid')
  TheSample.set('sampling_func', sampler.constrained_pick_models_and_rotation_ids)
  TheSample.set('max_sample_size', 5000)
  TheSample.set('range_log_Teff', [3.95, 4.11])
  TheSample.set('range_log_g', [3.9, 4.3])
  TheSample.set('range_eta', [0, 0])

  # TheSample.set('star', TheStar)

  # seismic constraints
  TheSample.set('modes_id_types', [2])   # for l=1, m=0: dipole zonal modes  

  # search plan for matching frequencies
  TheSample.set('search_strictly_for_dP', True)
  TheSample.set('trim_delta_freq_factor', 0.25)

  # For non-rotating models, exclude eta column (which is just 0.0) to avoid singular X matrix
  TheSample.set('exclude_eta_column', True)

  # Now, build the learning sets
  TheSample.build_learning_set()

  # Get the sample
  learning_x  = TheSample.get('learning_x')
  print('   Size of the retrieved sample is: "{0}"'.format(TheSample.sample_size))
  print('   Names of the sampled columns: ', learning_x.dtype.names)

  # Get the corresponding frequencies
  learning_y = TheSample.get('learning_y')
  print('   Shape of the synthetic frequencies is: ', learning_y.shape) 
  print('   ')

  # Plot the histogram of the learning Y sample
  if False:
    plot_sampler.hist_learning_x(TheSample, 'test_suite/plots/KIC-10526294-hist-X.png')
    plot_sampler.hist_learning_y(TheSample, 'test_suite/plots/KIC-10526294-hist-Y.png')

  # Set percentages for training, cross-validation and test sets
  TheSample.set('training_percentage', 0.80)
  TheSample.set('cross_valid_percentage', 0.15)
  TheSample.set('test_percentage', 0.05)

  # Now, create the three sets from the learning set
  TheSample.split_learning_sets()

  # Print sizes of each learning sets
  print('   The Training set: X:{0}, Y:{1}'.format(TheSample.training_x.shape, TheSample.training_y.shape))
  print('   The Cross-Validation set: X:{0}, Y:{1}'.format(TheSample.cross_valid_x.shape, TheSample.cross_valid_y.shape))
  print('   The Test set: X:{0}, Y:{1}'.format(TheSample.test_x.shape, TheSample.test_y.shape))
  print() 

  return TheSample

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  status = main()
  sys.exit(status)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
