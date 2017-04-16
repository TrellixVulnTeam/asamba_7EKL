
import sys, os, glob
import logging
import numpy as np

# from test_unit_sampling import main as tus_main
from test_unit_ann import main as tun_main 
import interpolator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def main():

  # Easy/fast way to provide the dependencies
  TheANN    = tun_main()
  TheSample = TheANN.get('sampling')

  # Get an instance of the interpolation class to work with
  TheInterp = interpolator.interpolation() 
  TheInterp.set('anchor_param_names', TheSample.get('feature_names'))
  TheInterp.set('', TheSample.get('learning_radial_orders'))

  TheInterp.set('anchor_param_values', TheANN.get('marginal_features'))
  TheInterp.set('anchor_frequencies', TheANN.get('MAP_frequencies'))
  TheInterp.set('anchor_radial_orders', TheANN.get('MAP_radial_orders'))
