#! /usr/bin/python

import sys, os, glob
import logging
import numpy as np 

from asamba import artificial_neural_network as ann

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger  = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
  num_neurons = [7, 5, 3]
  y_in   = [0.8, 0.87, 0.92, 0.98, 1.03, 1.08, 1.12]
  y_out  = [0.12, 0.29, 0.78]
  NNet   = ann.neural_network(num_neurons, y_in, y_out)

  NNet.initialize()

  NNet.feedforward()  
  NNet.backpropagate()

  # b_corr = [_[:,0] for _ in NNet.get('corrections')[1:]]
  # print(b_corr)
  print(NNet.cost)

  return 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  status = main()
  sys.exit(status)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
