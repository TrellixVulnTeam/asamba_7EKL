import sys, os, glob
import logging
import numpy as np 

from test_unit_sampling import main as main_sampling
import artificial_neural_network as ann

def main():
  """
  This test unit is built on top of the test_unit_sampling, because it calls that, and uses the "sampling"
  object returned from that test.
  """
  TheSample = main_sampling()

  # Get an instance of the ANN class
  TheANN    = ann.neural_net()

  # Insert the TheSample into TheANN object
  TheANN.setter('sampling', TheSample)

  # Set the relevant attributes
  TheANN.solve_normal_equation()

  theta_Neq = TheANN.get('theta_normal_equation')

  return TheANN

if __name__ == '__main__':
  stat = main()
  sys.exit(stat)