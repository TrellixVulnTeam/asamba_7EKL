
import sys, os, glob
import logging
import numpy as np 
from grid import sampler
from grid import star


def main():

  print ' - Load the mode list from a file'
  mode_file = 'stars/KIC_10526294.freq'
  modes     = star.load_modes_from_file(filename=mode_file, delimiter=',')

  print ' - Attach the modes to a star object'
  TheStar   = star.star()
  TheStar.setter('name', 'KIC_10526294')
  TheStar.setter('modes', modes)

  print ' - Get an instance of the "sampling" class.'
  TheSample = sampler.sampling()
  TheSample.setter('dbname', 'grid')
  TheSample.setter('sampling_func', sampler.constrained_pick_models_and_rotation_ids)
  TheSample.setter('max_sample_size', 1000)
  TheSample.setter('range_log_Teff', [3.5, 4.0])
  TheSample.setter('range_log_g', [3.0, 4.0])
  TheSample.setter('range_eta', [0, 0])

  TheSample.setter('star', TheStar)

  # seismic constraints
  TheSample.setter('modes_id_types', [2])   # for l=1, m=0: dipole zonal modes  

  # search plan for matching frequencies
  TheSample.setter('search_strictly_for_dP', True)

  # Now, build the learning sets
  TheSample.build_learning_set()

  # Get the sample
  learning_x  = TheSample.get('learning_x')
  print '   Size of the retrieved sample is: "{0}"'.format(TheSample.sample_size)
  print '   Names of the sampled columns: ', learning_x.dtype.names

  # Get the corresponding frequencies
  learning_y = TheSample.get('learning_y')
  print '   Shape of the synthetic frequencies is: ', learning_y.shape 
  print '   '

  # Set percentages for training, cross-validation and test sets
  TheSample.setter('training_percentage', 80)
  TheSample.setter('cross_valid_percentage', 15)
  TheSample.setter('test_percentage', 5)

  # Now, create the three sets from the learning set
  TheSample.split_learning_sets()

if __name__ == '__main__':
  status = main()
  sys.exit(status)
