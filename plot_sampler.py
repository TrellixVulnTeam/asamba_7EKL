
import sys, os, glob
import logging
import numpy as np 
import pylab as plt 

from grid import star, sampler

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
  modes = self.star.modes
  freqs = np.array([mode.freq for mode in modes])

  # Learning set
  y = self.get('learning_y') 
  m, K = y.shape

  # Prepare the figure
  fig, ax = plt.subplots(1, figsize=(10, 4))
  plt.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.97)
  
  for i_col in range(K):
    y_i = y[:, i_col]
    ax.hist(y_i, bins=m/10, histtype='stepfilled', alpha=0.5, zorder=1)

    ax.axvline(x=freqs[i_col], ymin=0, ymax=m, linestyle='dashed', color='k', lw=1.5, zorder=2)

  # Cosmetics
  ax.annotate('N={0}'.format(self.sample_size), xy=(0.92, 0.90), xycoords='axes fraction', ha='center')
  ax.annotate('Total={0}'.format(self.max_sample_size), xy=(0.92, 0.83), xycoords='axes fraction', ha='center')

  ax.set_xlim(np.min(y)*0.95, np.max(y)*1.02)
  ax.set_xlabel(r'Frequency [d$^{-1}$]')
  ax.set_ylabel(r'Count')

  plt.savefig(figure_name)
  print ' - hist_learning_y: saved {0}'.format(figure_name)
  plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
