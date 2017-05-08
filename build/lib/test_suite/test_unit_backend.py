#! /usr/bin/python

from __future__ import print_function
from __future__ import unicode_literals
import sys, os, glob
import logging
import numpy as np 

from asamba import star, sampler
from asamba import backend

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():
  """ Test unit for the BackEnd """

  this      = backend.ModellingSession()
  logger.info('Got a working instance of the ModellingSession')

  this.set('name', 'KIC_10526294')
  this.set('Teff', 11500.)
  this.set('Teff_err_lower', 500.)
  this.set('Teff_err_upper', 500.)
  this.set('log_g', 4.1)
  this.set('log_g_err_lower', 0.2)
  this.set('log_g_err_upper', 0.2)
  logger.info('Observational data inserted')

  mode_file = 'stars/KIC_10526294.freq'
  # modes     = star.load_modes_from_file(filename=mode_file, delimiter=',')
  # this.set('modes', modes)
  this.load_modes_from_file(filename=mode_file, delimiter=',')
  logger.info('Loaded the mode list from file: {0}'.format(mode_file))
  print(this.get('num_modes'))

  this.set('dbname', 'grid')
  this.set('sampling_func', sampler.constrained_pick_models_and_rotation_ids)
  this.set('max_sample_size', 5000)
  this.set('range_log_Teff', [3.95, 4.11])
  this.set('range_log_g', [3.9, 4.3])
  this.set('range_eta', [0, 0])
  logger.info('Sampling settings plugged in')

  this.set('modes_id_types', [2])   # for l=1, m=0: dipole zonal modes  
  this.set('search_strictly_for_dP', True)
  this.set('trim_delta_freq_factor', 0.25)
  # For non-rotating models, exclude eta column (which is just 0.0) to avoid singular X matrix
  this.set('exclude_eta_column', True)
  logger.info('Asteroseismic search plan for matching frequencies defined')

  # Now, build the learning sets
  this.build_learning_set()

  # Get the sample
  learning_x  = this.get('learning_x')
  logger.info('Size of the retrieved sample is: "{0}"'.format(this.sample_size))
  logger.info('Names of the sampled columns: {0}'.format(learning_x.dtype.names))

  # Get the corresponding frequencies
  learning_y = this.get('learning_y')
  logger.info('Shape of the synthetic frequencies is: {0}'.format(learning_y.shape) )

  # Plot the histogram of the learning Y sample
  if False:
    this.hist_learning_x(this, 'test_suite/plots/KIC-10526294-hist-X.png')
    this.hist_learning_y(this, 'test_suite/plots/KIC-10526294-hist-Y.png')

  # Set percentages for training, cross-validation and test sets
  this.set('training_percentage', 0.80)
  this.set('cross_valid_percentage', 0.15)
  this.set('test_percentage', 0.05)
  # Now, create the three sets from the learning set
  this.split_learning_sets()
  logger.info('The Training set: X:{0}, Y:{1}'.format(this.training_x.shape, this.training_y.shape))
  logger.info('The Cross-Validation set: X:{0}, Y:{1}'.format(this.cross_valid_x.shape, this.cross_valid_y.shape))
  logger.info('The Test set: X:{0}, Y:{1}'.format(this.test_x.shape, this.test_y.shape))

  return 0

if __name__ == '__main__':
  stat = main()
  sys.exit(stat)
