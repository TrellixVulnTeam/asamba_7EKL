#! /usr/bin/python

from __future__ import print_function
from __future__ import unicode_literals
import sys, os, glob, time
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

  logger.info('Start time: {0}'.format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
  logger.info('Load the mode list from a file')
  mode_file = '../asamba/stars/KIC_10526294.freq'

  logger.info('Get an instance of the "sampling" class.')
  TheSample = sampler.sampling()

  TheSample.set('name', 'KIC_10526294')

  TheSample.set('Teff', 11550.)
  TheSample.set('Teff_err_lower', 500.)
  TheSample.set('Teff_err_upper', 500.)
  TheSample.set('log_g', 4.1)
  TheSample.set('log_g_err_lower', 0.2)
  TheSample.set('log_g_err_upper', 0.2)

  # TheSample.set('modes', modes)
  TheSample.load_modes_from_file(filename=mode_file, delimiter=',')

  TheSample.set('dbname', 'grid')

  TheSample.set('use_6D_feature_box', True)
  # TheSample.set('use_constrained_sampling', True)
  # TheSample.set('use_random_sampling', True)

  # TheSample.set('range_log_Teff', [3.95, 4.11]) # from paper: [3.95, 4.11]
  # TheSample.set('range_log_g', [3.9, 4.3])      # from paper: [3.9, 4.3]

  TheSample.set('range_M_ini', [2.5, 5])
  TheSample.set('range_fov', [0, 0.04])
  TheSample.set('range_Z', [0.009, 0.02])
  TheSample.set('range_logD', [0, 7])
  TheSample.set('range_Xc', [0.3, 0.71])
  TheSample.set('range_eta', [0, 0])

  # seismic constraints
  TheSample.set('modes_id_types', [2])   # for l=1, m=0: dipole zonal modes  

  # Set the maximum of returned sample size
  # TheSample.set('max_sample_size', 5000)
  # search plan for matching frequencies
  TheSample.set('sampling_shuffle', True)
  TheSample.set('search_strictly_for_dP', True)
  TheSample.set('trim_delta_freq_factor', 0.25)

  # For non-rotating models, exclude eta column (which is just 0.0) to avoid singular X matrix
  TheSample.set('exclude_eta_column', True)

  # Now, build the learning sets
  if True: 
    TheSample.build_learning_set()
    TheSample.write_sample_to_h5(filename='/Users/ehsan/Desktop/learning_set.h5', include_periods=True)
  else:
    TheSample.load_sample_from_hdf5(filename='/Users/ehsan/Desktop/learning_set.h5')

  # Get the sample
  learning_x  = TheSample.get('learning_x')
  logger.info('Size of the retrieved sample is: "{0}"'.format(TheSample.sample_size))
  logger.info('Names of the sampled columns: '.format(TheSample.feature_names))

  # Get the corresponding frequencies
  learning_y = TheSample.get('learning_y')
  print('Shape of the synthetic frequencies is: ', learning_y.shape, '\n') 

  # Plot the histogram of the learning Y sample
  if False:
    plot_sampler.hist_learning_x(TheSample, 'plots/KIC-10526294-hist-X.png')
    plot_sampler.hist_learning_y(TheSample, 'plots/KIC-10526294-hist-Y.png')

  # Set percentages for training, cross-validation and test sets
  TheSample.set('training_percentage', 0.80)
  TheSample.set('cross_valid_percentage', 0.15)
  TheSample.set('test_percentage', 0.05)

  # Now, create the three sets from the learning set
  TheSample.split_learning_sets()

  # Print sizes of each learning sets
  logger.info('The Training set: X:{0}, Y:{1}'.format(TheSample.training_x.shape, TheSample.training_y.shape))
  logger.info('The Cross-Validation set: X:{0}, Y:{1}'.format(TheSample.cross_valid_x.shape, TheSample.cross_valid_y.shape))
  logger.info('The Test set: X:{0}, Y:{1}\n'.format(TheSample.test_x.shape, TheSample.test_y.shape))

  # Get the tagged representation of the learning/training/CV/test sets
  TheSample.set('path_Xc_tags_ascii', '../asamba/data/tags/Xc-tags.txt')
  TheSample.convert_features_to_tags()
  learning_tags = TheSample.get('learning_tags')

  return TheSample

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  logger.info('Start time: {0}'.format(time.strftime("%a, %d %b %Y, %H:%M:%S", time.gmtime())))
  status = main()
  logger.info('End time:   {0}'.format(time.strftime("%a, %d %b %Y, %H:%M:%S", time.gmtime())))
  sys.exit(status)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
