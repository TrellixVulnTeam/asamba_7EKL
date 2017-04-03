
import sys, os, glob
import logging
import numpy as np 
from grid import sampler


def main():

  print ' - Get an instance of the "sampling" class.'
  TheSample = sampler.sampling()
  TheSample.set_dbname('grid')
  TheSample.set_sampling_func(sampler.constrained_pick_models_and_rotation_ids)
  TheSample.set_max_sample_size(1000)
  TheSample.range_log_Teff = [3.5, 4.0]
  TheSample.range_log_g = [3.0, 4.0]
  TheSample.range_eta = [0, 0]

  # seismic constraints
  TheSample.set_modes_id_types([2])  # for l=1, m=0: dipole zonal modes  

  # Now, build the learning sets
  TheSample.build_learning_set()

  # Get the sample
  sample    = TheSample.get('sample')
  print '   Size of the retrieved sample is: "{0}"'.format(TheSample.sample_size)
  print '   Names of the sampled columns: ', sample.dtype.names

  # Set percentage of different sets

if __name__ == '__main__':
  status = main()
  sys.exit(status)