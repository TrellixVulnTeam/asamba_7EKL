from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from builtins import range
from past.utils import old_div
import sys, os, glob
import logging
import numpy as np 
import pylab as plt 

from asamba import star, sampler

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def hist_learning_x(self, figure_name):
  """
  The learning set goes through a (strict) filtering, adapted to the star (e.g. based on the observed
  frequency/period regularities/spacing, etc.). As a result of that, only a subset of models survive
  this. Therefore, it is very intuitive to take a look at a histogram/distribution of the input parameters
  that survived this, and perhaps make a very rough/preliminary conclusion on the range of accepted
  parameters that roughly reproduces the observations. In principle, this plotting routine provides 
  this functionality.

  @param self: an instance of the sampler.sampling class 
  @type self: object
  @param figure_out: the full path to the figure to be saved on disk.
  @type figure_out: str
  """
  if not self.learning_done:
    logger.error('hist_learning_x: Call build_learning_sets() method first')
    sys.exit(1)

  x     = self.get('learning_x')
  m, n  = x.shape
  names = self.get('feature_names')

  fig, tup_ax = plt.subplots(2, 3, figsize=(9, 6))
  plt.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.97, wspace=0.2, hspace=0.25)
  tup_ax      = tup_ax.reshape(-1)

  for i, name in enumerate(names):
    ax      = tup_ax[i]
    feature = x[:, i]
    mn, mx  = np.min(feature), np.max(feature)
    ax.hist(feature, bins=old_div(m,10), histtype='stepfilled')

    # cosmetics
    ax.set_xlim(mn*0.95, mx*1.02)
    ax.set_xlabel(r'{0}'.format(name))

  plt.savefig(figure_name)
  print(' - plot_sampler: hist_learning_x: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def hist_learning_y(self, figure_name):
  """
  Plot the histogram of the verctor of learning set Y for all frequancies there. Note that Y has
  dimensions of (m x K), where m is the number of rows of the learning set, and K is the number 
  of modes. The plotted histogram will loop over the K columns, and overplots the histogram for 
  each mode.

  @param self: an instance of the sampler.sampling class 
  @type self: object
  @param figure_out: the full path to the figure to be saved on disk.
  @type figure_out: str
  """
  # observed values
  modes = self.get('modes')
  freqs = np.array([mode.freq for mode in modes])

  # Learning set
  y = self.get('learning_y') 
  m, K = y.shape

  # Prepare the figure
  fig, ax = plt.subplots(1, figsize=(10, 4))
  plt.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.97)
  
  max_hist = -1
  for i_col in range(K):
    y_i = y[:, i_col]
    tup_hist = ax.hist(y_i, bins=old_div(m,10), histtype='stepfilled', alpha=0.5, zorder=1)
    vals     = tup_hist[0]
    max_hist = np.max([max_hist, np.max(vals)])

    ax.axvline(x=freqs[i_col], ymin=0, ymax=m, linestyle='dashed', color='k', lw=1.5, zorder=2)

  # # include the predicted frequencies from solving the normal equation
  # theta = self.normal_equation_theta
  # g     = self.normal_equation_features
  # h_theta = np.dot(g.T, theta)           # (n+1, 1).T x (n+1, K) = (1, K)
  # ax.scatter(h_theta, np.ones(len(h_theta)) * max_hist * 0.9, s=50, color='red', marker='o', zorder=3)

  # Cosmetics
  ax.annotate('N={0}'.format(self.sample_size), xy=(0.92, 0.90), xycoords='axes fraction', ha='center')
  ax.annotate('Total={0}'.format(self.max_sample_size), xy=(0.92, 0.83), xycoords='axes fraction', ha='center')

  ax.set_xlim(np.min(y)*0.95, np.max(y)*1.02)
  ax.set_xlabel(r'Frequency [d$^{-1}$]')
  ax.set_ylabel(r'Count')

  plt.savefig(figure_name)
  print(' - plot_sampler: hist_learning_y: saved {0}'.format(figure_name))
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
