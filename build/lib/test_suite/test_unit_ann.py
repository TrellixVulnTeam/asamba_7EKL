#! /usr/bin/python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import range

import sys, os, glob
import logging
import numpy as np 

from test_unit_sampling import main as main_sampling
from asamba import artificial_neural_network as ann

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
  """
  This test unit is built on top of the test_unit_sampling, because it calls that, and uses the "sampling"
  object returned from that test.
  """
  TheSample = main_sampling()

  # Get an instance of the ANN class
  TheANN    = ann.neural_net()

  # Insert few attributes where the order of insertion matters
  TheANN.set('Teff', TheSample.get('Teff'))
  TheANN.set('Teff_err_lower', TheSample.get('Teff_err_lower'))
  TheANN.set('Teff_err_upper', TheSample.get('Teff_err_upper'))
  TheANN.set('log_g', TheSample.get('log_g'))
  TheANN.set('log_g_err_lower', TheSample.get('log_g_err_lower'))
  TheANN.set('log_g_err_upper', TheSample.get('log_g_err_upper'))

  # Insert the TheSample attributes into TheANN object
  dic_sample= TheSample.__dict__
  for key, val in list(dic_sample.items()):
    TheANN.set(key, val)

  # Set the relevant attributes
  TheANN.solve_normal_equation()

  theta_Neq = TheANN.get('normal_equation_theta')

  X_Neq     = TheANN.get('normal_equation_features')
  print(' - Solution from the Normal Equation:')
  print('   Intercept:{0:0.4f}, mass:{1:0.3f}, fov:{2:0.3f}'.format(X_Neq[0], X_Neq[1], X_Neq[2]))
  print('   Z:{0:0.3f}, logD:{1:0.2f}, Xc:{2:0.4f}'.format(X_Neq[3], X_Neq[4], X_Neq[5]))
  if not TheANN.exclude_eta_column:
    print('   eta:{0:0.2f}'.format(X_Neq[-1]))
  print('   Cost is: J={0:0.2e} \n'.format(TheANN.normal_equation_cost))
  
  # compare the observed and predicted frequencies from the normal equation
  if False:
    theta     = TheANN.normal_equation_theta
    g         = TheANN.normal_equation_features
    h_theta   = np.dot(g.T, theta)           # (n+1, 1).T x (n+1, K) = (1, K)
    K         = len(h_theta)
    modes     = TheANN.star.modes
    obs_freqs = np.array([mode.freq for mode in modes])
    for j in range(K):
      print('   mode {0}: Obs:{1}, modeled:{2}'.format(j+1, obs_freqs[j], h_theta[j]))

  # Maximum a posteriori analysis
  TheANN.set('MAP_use_log_Teff_log_g_prior', True)
  TheANN.set('frequency_sigma_factor', 1.)
  TheANN.set('rescale_ln_likelihood', True)
  MAP       = TheANN.max_a_posteriori()
  
  ln_prior  = TheANN.get('MAP_ln_prior')
  ln_L      = TheANN.get('MAP_ln_likelihood')
  ln_evid   = TheANN.get('MAP_ln_evidence')
  ln_post   = TheANN.get('MAP_ln_posterior')
  #
  print(' - Maximum Likelihood Results:')
  print('   ln(P(h)):   min:{0:.2f}, max:{1:.2f}'.format(np.min(ln_prior), np.max(ln_prior)))
  print('   ln(P(D|h)): min:{0:.2f}, max:{1:.2f}'.format(np.min(ln_L), np.max(ln_L)))
  print('   ln(P(D)) = {0:.2f}'.format(ln_evid))
  print('   ln(P(h|D)): min:{0:.2f}, max:{1:.2f}\n'.format(np.min(ln_post), np.max(ln_post)))

  print(' - Radial orders of the MAP model:')
  print(TheANN.get('MAP_radial_orders'), '\n')

  if True:
    all_n_pg= TheANN.get('learning_radial_orders')
    Catch   = False
    for k, row in enumerate(all_n_pg):
      diffs = row[1:] - row[:-1]
      if np.mean(diffs) != 1:
        Catch = True
        print(k, row)
    if Catch:
      logging.warning('At least one instance found where n_pg are not contiguous \n')
    else:
      logging.info('All radial orders in the learning set are contiguous \n')
  
  print('\n - Marginalized features') 
  TheANN.marginalize()
  features  = TheANN.get('feature_names')
  marg_vals = TheANN.get('marginal_features')
  for i, name in enumerate(features):
    print('   {0} = {1:.4f}'.format(name, marg_vals[i]))
  print()

  return TheANN

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
  stat = main()
  sys.exit(stat)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
