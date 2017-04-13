import sys, os, glob
import logging
import numpy as np 

from test_unit_sampling import main as main_sampling
import artificial_neural_network as ann

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def main():
  """
  This test unit is built on top of the test_unit_sampling, because it calls that, and uses the "sampling"
  object returned from that test.
  """
  TheSample = main_sampling()

  # Get an instance of the ANN class
  TheANN    = ann.neural_net()

  # Insert the TheSample into TheANN object
  TheANN.set('sampling', TheSample)

  # Set the relevant attributes
  TheANN.solve_normal_equation()

  theta_Neq = TheANN.get('normal_equation_theta')

  X_Neq     = TheANN.get('normal_equation_features')
  print ' - Solution from the Normal Equation:'
  print '   Intercept:{0:0.4f}, mass:{1:0.3f}, fov:{2:0.3f}'.format(X_Neq[0], X_Neq[1], X_Neq[2])
  print '   Z:{0:0.3f}, logD:{1:0.2f}, Xc:{2:0.4f}'.format(X_Neq[3], X_Neq[4], X_Neq[5])
  if not TheSample.exclude_eta_column:
    print '   eta:{0:0.2f}'.format(X_Neq[-1])
  print '   Cost is: J={0:0.2e} \n'.format(TheANN.normal_equation_cost)
  
  # compare the observed and predicted frequencies from the normal equation
  if False:
    theta     = TheANN.normal_equation_theta
    g         = TheANN.normal_equation_features
    h_theta   = np.dot(g.T, theta)           # (n+1, 1).T x (n+1, K) = (1, K)
    K         = len(h_theta)
    modes     = TheSample.star.modes
    obs_freqs = np.array([mode.freq for mode in modes])
    for j in range(K):
      print '   mode {0}: Obs:{1}, modeled:{2}'.format(j+1, obs_freqs[j], h_theta[j])

  # Maximum a posteriori analysis
  TheANN.set('MAP_use_log_Teff_log_g_prior', True)
  TheANN.set('frequency_sigma_factor', 1.)
  TheANN.set('rescale_chi_square', True)
  MAP       = TheANN.max_a_posteriori()
  marg_M_ini= TheANN.marginalize(wrt='M_ini')
  print marg_M_ini

  return TheANN

if __name__ == '__main__':
  stat = main()
  sys.exit(stat)