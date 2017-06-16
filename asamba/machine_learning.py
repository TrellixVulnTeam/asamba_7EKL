
"""
This module provides various functionalities for carrying out Machine Learning (ML for short)
analysis (modelling) using the asteroseismic database, and a given set of observations. This
module builds heavily on the "sampler" module.

One of the nice features is that the marginalization of posterior probabilities are carried 
out over attribute tags, rather than their own values. This facilitates a unique representation
of different states of an attribute; e.g. logD can have only 5 states (0.0 to max(logD)), and 
the tag captures the attribute state, rather than its absolute value!
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from past.utils import old_div
import sys, os, glob
import logging
import numpy as np 

from asamba import utils, star, sampler

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger  = logging.getLogger(__name__)
is_py3x = sys.version_info[0] >= 3 # to handle unicode encoding for Python v2.7

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    ###  ### ######  ###     ###   #####
    #    #    #    #  #     #  #       #   #     #
    #     #   #    #  #     #  #       #  #
    #    #    #    #  ######   #       #  #
    #####     #    #  #     #  #       #  #
    #         #    #  #     #  #    #  #   #     #
    #          ####   ######  ####### ###   #####


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class learner(sampler.sampling):
  """
  This class extends the sampler.sampling() class
  """
  def __init__(self):
    super(learner, self).__init__()

    #.............................
    # Normal Equation
    #.............................
    # Status of the procedure
    self.normal_equation_done = False
    # Analytic solution to the coefficients
    self.normal_equation_theta = 0
    # Solution of the features given analytic theta, and the observed y
    self.normal_equation_features = 0
    # The cost from the optimized theta and features
    self.normal_equation_cost = 0

    #.............................
    # Maximum a posteriori (MAP)
    #.............................
    # The status of the MAP algorithm
    self.MAP_done = False
    # Working feature matrix for MAP
    self.MAP_work_features = 0
    # Working frequency matrix for MAP
    self.MAP_work_frequencies = 0

    # Use uniform prior distribution or not
    self.MAP_uniform_prior = False
    # Set prior based on log_Teff and log_g
    self.MAP_use_log_Teff_log_g_prior = False
    # The prior values, i.e. P(h)
    self.MAP_prior = 0
    # The prior in natural log scale
    self.MAP_ln_prior = 0

    # Status of the chi square computations
    self.chi_square_done = False
    # Enhance observed frequencies by this factor for chi square computation
    self.frequency_sigma_factor = 1.0
    # 2-D matrix of chi^2_i(j) for the scaled squared frequency difference
    # for the i-th mode compared with that of the i-th mode of the j-th model
    self.MAP_chi_square_matrix = 0
    # sum over MAP_chi_square_matrix along the 2nd axis to give the total 
    # chi square for each input model. This is ln( P(D|h) )
    self.MAP_chi_square = 0

    # The likelihood probability distribution, P(D\h)
    self.MAP_likelihood = 0
    # The likelihood in natural log scale
    self.MAP_ln_likelihood = 0
    # Rescale ln(likelihood) + ln(prior) so that for the MAP model,
    # the sum of the two is zero
    self.rescale_ln_probabilities = True

    # The evidence
    self.MAP_evidence = 0
    # The evidence in natural log scale
    self.MAP_ln_evidence = 0

    # The posterior
    self.MAP_posterior = 0
    # The posterior in natural log scale
    self.MAP_ln_posterior = 0

    # The best-model feature that maximize the likelihood (MAP)
    # That's what you want to look at!
    self.MAP_feature = 0
    # The frequencies corresponding to MAP_feature (from GYRE output)
    self.MAP_frequencies = 0
    # The models.id_track for the MAP model
    self.MAP_id_track = 0
    # The models.id for the MAP model
    self.MAP_id_model = 0
    # The radial orders corresponing to the MAP_feature
    self.MAP_radial_orders = 0
    # The mode identification types (from grid.sql) corresponding the MAP_feature
    self.MAP_mode_types = 0

    #.............................
    # Marginalization
    #.............................
    # Flag to specify if all features are marginalized
    self.marginalize_done  = False
    # List of marginalized feature arrays with identical order as in
    # sampling.feature_names, dtype: list of tuples (two ndarrays each)
    self.marginal_results  = []
    # List of marginalized feature values with identical order as in
    # sampling.feature_names, i.e. basically the MAP value from marginal_results
    self.marginal_features = []

  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    super(learner, self).set(attr, val)
    if not hasattr(self, attr):
      logger.error('learner: set: Attribute "{0}" is unavailable.')
      sys.exit(1)

    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    super(learner, self).get(attr)
    if not hasattr(self, attr):
      logger.error('learner: get: Attribute "{0}" is unavailable.')
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def solve_normal_equation(self):
    """
    Find the analytic solution for the unknown hypothesis coefficients \f$\theta\f$, which minimizes the
    cost function \f$ J(\theta) \f$ as defined below.
    
    \f[ J(\theta)= \frac{1}{2m} (\theta^T X-y)^T \cdot (\theta^T X-y) \f]
    
    For more information refer to: 
    <a href="http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression">Click to Open</a> 
    Consequently, the analytic solution to \f$\theta\f$ is:
    
    \f[ \theta_0 = (X^T \cdot X)^{-1} \cdot X^{-1} \cdot y. \f]

    A brief remark on the dimensionality of the terms: For a learning set of size \f$ m\f$, with \f$ n+1\f$ features
    (including the intercept coefficient), and for the observed/trained output \f$ y \f$ being a matrix of \f$ m\times K\f$
    (for \f$ K\f$ modes), then the coefficient matrix \f$ \theta_0\f$ is \f$ (n+1) \times K \f$.

    Once \f$\theta_0\f$ is analytically derived, then the cost function is minimized. If we assume
    this set of coefficients make the cost function approach zero \f$J(\theta_0)\approx 0\f$, intuitively 
    \f$ \theta_0^T\cdot X \approx y \f$. 

    One can immediately solve for the unknown feature vector \f$ X \f$, which reproduces the observations \f$ y_0\f$, 
    given the corresponding coefficients \f$ \theta_0 \f$. To that end, we multiply both sides of the last equation 
    by \f$ \theta \f$, followed by a multiplication with \f$ (\theta_0 \cdot \theta_0^T)^{-1} \f$ to yield \f$ X \f$:
    
    \f[ X_0 \approx (\theta_0 \cdot \theta_0^T)^{-1} \cdot (\theta \cdot y_0) \f]

    Needless to highlight that \f$X_0\f$ is a vector of size \f$(n+1)\f$, for an intercept and \f n\f$ features.
    
    Notes:
    - The resulting coefficients are saved as the following attribute self.normal_equation_theta, and the resulting
      feature vector \f$ X_0 \f$ is stored as the attribute self.normal_equation_features.
    - The model frequencies \f$ y \f$ and the observed frequencies \f$ y_0 \f$ are converted to the per day 
      (\f$ d^{-1} \f$) unit for a fair comparison.
    - "A major drawback of the Maximum Likelihood approach is that it is vulnerable to overfitting, because no care
       is taken for cmplex models that try to learn the specificities of the particular training set (Theodoridis, S.
       2015, Machine Learning book)."

    @param self: instance of the learner class
    @type self: object
    """
    _solve_normal_equation(self)
    self.set('normal_equation_done', True)

  ##########################
  def chi_square(self):
    """
    We define the \f$ \chi^2\f$ score between the i-th observed frequency \f$ f_i^{\rm (obs)}, and the model
    frequency \f$ f_i^{\rm (mod)}\f$ (coming from e.g. the learning set) as the following

    \f[ \chi^2 = \sum_{i=1}^{K} \chi^2_i = \sum_{i=1}^{K} \frac{1}{2}
                 \left(\frac{f_i^{\rm (obs) - f_i^{\rm (mod)}{\sigma_i^2}\right)^2. 
    \f]

    Here, \f$ K\f$ is the total number of observed modes, and \f$\sigma_i\f$ is the 1-\f$\sigma\f$ uncertainty
    around each observed frequency.

    @param self: An instance of the learner() class
    @type self: obj
    @return: the "MAP_chi_square_matrix" and "MAP_chi_square" attributes of the class will be set
    @rtype: None
    """
    _chi_square(self)
    self.set('chi_square_done', True)

  ##########################
  def max_a_posteriori(self):
    """
    This routine finds the attributes of the model which maximizes the posterior likelihood function, hence MAP. 
    It consists of the following steps:
    
    1. Priors: Either set uniformly, or are set based on a comparison between log_Teff and log_g of the model and
       the star. For each hypothesis \f$h\f$, we return the prior information \f$P(h)\f$, and \f$\ln P(h)\f$.
    
    2. LogLikelihood, or chi square \f$\chi^2\f$: the natural logarithm of the probability density of the data 
       given the hypothesis, i.e. 
       \f$\chi^2=\ln P(D|h) = \frac{1}{2K}\sum_{i=1}^{K} \left((f_i^{\rm (obs)} - f_i^{\rm (mod)})/sigma_i \right)^2 \f$.
       A recommended option here is to "rescale" the chi-square values to avoid numerical overflow.

    3. Evidence, which is basically an inner dot product between the prior vector and the likelihood vector:
       \f$P(D)=\sum_h P(D\h) P(h) \f$. We return both \f$P(D)\f$ and \f$\ln P(D)\f$.

    4. Posterior \f$P(h|D)\f$: Based on the Bayes Theorem, the posterior is 

       \f[
           \frac{P(h|D)}{s}=\frac{\frac{P(D|h)}{s}P(h)}{P(D)},
       \f]

       where \f$s\f$ is an optional "scaling" factor used to "rescale" the loglikelihood. Indeed, setting \f$s=1\f$
       recovers the Bayes theorem in its original form. This scaling is allowed, since we only make relative 
       comparison between the models.

    @param self: an instance of the learner() class    
    @type self: obj
    """
    _max_a_posteriori(self)
    self.set('MAP_done', True)

  ##########################
  def marginalize_wrt_x_y(self, wrt_x, wrt_y):
    """
    Marginalize the learning features (MAP_work_features), with respect to two (hence "wrt_x" and "wrt_y")
    feature columns (whose names are available as self.sampling.feature_names).
    We internally assert that the the sum of all marginal probabilities add sufficiently close to one.

    @param self: an instance of the learner() class
    @type self: obj
    @param wrt_x, wrt_y: marginalize with respect to wrt_x and wrt_y
    @type wrt_x, wrt_y: str
    @return: a list of tuples, where each tuple corresponds to one marginal probability with the following
         three elements:
         - val_x: which is the value of the marginal parameter (wrt_x) used 
         - val_y: which is the value of the marginal parameter (wrt_y) used
         - marg_x_y: the marginal probability (sum=1) with respect to wrt_x and wrt_y
    @rtype: list of tuples
    """
    return _marginalize_wrt_x_y(self, wrt_x, wrt_y)

  ##########################
  def marginalize_wrt(self, wrt):
    """
    Marginalize the learning features (MAP_work_features), with respect to one (hence "wrt") of the 
    feature columns (whose names are available as self.sampling.feature_names)
    We internally assert that the the sum of all marginal probabilities add sufficiently close to one.

    @param self: an instance of the learner() class
    @type self: obj
    @param wrt: marginalize with respect to
    @type wrt: str
    @return: list of tuples, where each member tuple has the following two elements
         - val_wrt, which is the value of the marginal parameter (assigned as wrt)
         - marg_wrt, the marginal probability corresponding to val_wrt
    @rtype: list of tuples
    """
    return _marginalize_wrt(self, wrt)

  ##########################
  def marginalize(self):
    """
    Iterate over all features in the learning set (whose names are stored in self.sampling.feature_names),
    and marginalize with respect to each of these quantities. The outcome of the marginalization will be stored in
    two attribute of the learner class:
    
    1. marginal_results: which is a list of tuples; read the documentation of marginalize_wrt() method for more info.
    
    2. marginal_features: which is basically cherrypicking of the most likely value from the marginal_features list.
       The order of the outputs here matches exactly that of sampling.features_names

    The return structure of "self.marginal_results" is noteworthy. Because we iteratively call the
    marginalize_wrt() method and collect its results (tuple with two ndarrays), the "self.marginal_results"
    is a list of tuples.
    """
    _marginalize(self)

  ##########################
  def sort_by_posterior(self):
    """
    This method sorts all input feature and frequency lists, in addition to the prior/likelihood/posterior
    arrays based on the ln_posterior in *decreasing* order, so that the maximum a posteriori (MAP) model is 
    always the zeroth-element of all those arrays, after this method is called.
    """
    _sort_by_posterior(self)

  ##########################


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    #####    ###  ###  ###    ###    ###########  ######### 
    #    #   #    #    #    #    #     # #    #    #    #  #       #
    #     #  #     #   #    #    #     # #         #       #
    #    #   #    #    #     #  #     #   #        #       ####
    #####    #####     #     #  #     #####        #       #
    #        #  #      #      #      #     #       #       #       #
    #        #    #   ###    ###    ###   ###      #       #########


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# A N A L Y T I C   S O L U T I O N
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _solve_normal_equation(self):
  """
  Refer to the documentation of solve_normal_equation() for further details.
  """
  # Get THE instance of the sampling class already stored in learner object 
  # sample = self.get('sampling') 
  # Check if the sampling is already done or not
  if not self.learning_done:
    logger.error('_solve_normal_equation: The sampling is not done yet. Call sampler.sampling.build_sampling_sets() first')
    sys.exit(1)

  x = self.learning_x                  # (m, n)
  if self.exclude_eta_column: x = x[:, :-1]
  x = utils.prepend_with_column_1(x)   # (m, n+1)
  y = self.learning_y                  # (m, K)

  a = np.dot(x.T, x)                   # (n+1, n+1)
  b = np.linalg.inv(a)                 # (n+1, n+1)
  c = np.dot(x.T, y)
  d = np.dot(b, c)

  theta = d[:]                         # (n+1, K)
  self.set('normal_equation_theta', theta)

  # observed frequency from list of modes
  modes = self.modes

  # Now, solve for the "best-value" features using theta from above
  freqs = np.array([ mode.freq for mode in modes ]).T # (K, 1)

  e = np.linalg.inv(np.dot(theta, theta.T)) # (n+1, n+1)
  f = np.dot(theta, freqs)             # (n+1, K) x (K, 1) == (n+1, 1)
  g = np.dot(e, f)                     # (n+1, n+1) x (n+1, 1)  == (n+1, 1)

  self.set('normal_equation_features', g)

  h = np.dot(g.T, theta) - freqs
  J = old_div(np.dot(h.T, h), (2 * len(freqs)))

  self.set('normal_equation_cost', J)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# M A X I M U M   L I K E L I H O O D   E S T I M A T I O N 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _max_a_posteriori(self):
  """
  Refer to the documentation below the max_a_posteriori() method for further details
  """
  _set_work_matrixes(self)

  _set_priors(self)
  _chi_square(self)
  _set_likelihood(self)
  _rescale_ln_probabilities(self)
  _set_evidence(self)
  _set_posterior(self)
  
  ln_posterior     = self.get('MAP_ln_posterior')
  ind_max_post     = np.argmax(ln_posterior)
  MAP_work_x       = self.get('MAP_work_features')
  MAP_work_y       = self.get('MAP_work_frequencies')
  learning_n_pg    = self.get('learning_radial_orders')
  learning_types   = self.get('learning_mode_types')
  MAP_feature      = MAP_work_x[ind_max_post]
  MAP_frequencies  = MAP_work_y[ind_max_post]
  MAP_n_pg         = learning_n_pg[ind_max_post]
  MAP_mode_types   = learning_types[ind_max_post]
  self.set('MAP_feature', MAP_feature)
  self.set('MAP_frequencies', MAP_frequencies)
  self.set('MAP_radial_orders', MAP_n_pg)
  self.set('MAP_mode_types', MAP_mode_types)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_work_matrixes(self):
  """
  This routine copies the learning_tags and learning_y matrixes into self.MAP_work_features and 
  self.MAP_work_frequencies, and the rest of the routines only use the MAP_work_* matrixes for 
  the posterior calculations
  """
  if not self.tagging_done: self.convert_features_to_tags()
  self.set('MAP_work_features', self.get('learning_tags'))
  self.set('MAP_work_frequencies', self.get('learning_y'))
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_priors(self):
  """
  Refer to the documentation below the max_a_posteriori() method for further details
  """
  bool_arr    = np.array([self.MAP_uniform_prior, self.MAP_use_log_Teff_log_g_prior])
  n_bool      = np.sum(bool_arr * 1)
  if n_bool   != 1:
    logger.error('_set_priors: set only one of "MAP_uniform_prior" or "MAP_use_log_Teff_log_g_prior" to True:')
    sys.exit(1)

  work_y     = self.get('MAP_work_frequencies')
  m, K       = work_y.shape

  if self.MAP_uniform_prior:
    prior    = old_div(np.ones(m), float(m))
  elif self.MAP_use_log_Teff_log_g_prior:
    # get observed log_Teff and log_g together with their errors from the star
    if self.log_Teff_err_lower == 0 or self.log_Teff_err_upper == 0:
      logger.error('_set_priors: set log_Teff_err_lower and log_Teff_err_upper first')
      sys.exit(1)

    obs_log_Teff = self.log_Teff
    obs_log_Teff_err = np.max([ self.log_Teff_err_lower, self.log_Teff_err_upper ])
    obs_log_g    = self.log_g
    obs_log_g_err= np.max([ self.log_g_err_lower, self.log_g_err_upper ])
    if not all([ obs_log_Teff != 0, obs_log_Teff_err != 0, 
                 obs_log_g != 0, obs_log_g_err != 0 ]):
      logger.error('_set_priors: Specify the log_Teff, log_g and their errors properly')
      sys.exit(1)

    lrn_log_Teff = self.learning_log_Teff[:]
    lrn_log_g    = self.learning_log_g[:]

    prior_log_Teff = utils.gaussian(x=lrn_log_Teff, mu=obs_log_Teff, sigma=obs_log_Teff_err)
    prior_log_g    = utils.gaussian(x=lrn_log_g, mu=obs_log_g, sigma=obs_log_g_err)

    prior     = prior_log_Teff * prior_log_g

  if utils.has_nan(prior):
    logger.error('_set_priors: NaN detected')
    sys.exit(1)

  prior       = utils.substitute_inf(prior)
  ln_prior    = np.log(prior)

  self.set('MAP_prior', prior)
  self.set('MAP_ln_prior', ln_prior)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _chi_square(self):
  """
  Refer to the documentation below the chi_square() method for further details  
  """
  modes  = self.modes
  freqs  = np.array([ mode.freq for mode in modes ])
  sigma  = np.array([ mode.freq_err for mode in modes ])
  sigma  *= self.frequency_sigma_factor
  n_freq = len(freqs)                  

  work_y = self.get('MAP_work_frequencies')  # (m, K)
  m, K   = work_y.shape
  try:
    assert n_freq == K
  except AssertionError:
    logger.error('_chi_square: The number of observed frequencies ({0}) not equal to the number of columns of learning_y ({1})'.format(n_freq, K))
    sys.exit(1)

  # duplicate/reapeat the frequency vector m times to form identical matrix to learning_y for a 
  # vectorized calculation of chi square
  freqs_matrix = np.tile(freqs, m).reshape(m, K) # (m, K)
  sigma_matrix = np.tile(sigma, m).reshape(m, K) # (m, K)

  # calculate the chi^2 in a vectorized form, i.e. element-by-element operation in abstract form
  chi_2_matrix = old_div(np.power(old_div((freqs_matrix - work_y),sigma_matrix), 2.0), 2.0)
  if utils.has_nan(chi_2_matrix):
    logger.error('_chi_square: NaN detected in chi_2_matrix')
    sys.exit(1)

  chi_2        = np.sum(chi_2_matrix, axis=1) 
  if utils.has_nan(chi_2):
    logger.error('_chi_square: NaN detected in chi_2')
    sys.exit(1)

  # save the above two chi squares as attributes of the class
  self.set('MAP_chi_square_matrix', chi_2_matrix)
  self.set('MAP_chi_square', chi_2)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_likelihood(self):
  """
  For the full documentation refer to the max_a_posteriori() method
  """
  modes = self.get('modes')
  errs  = np.array([ mode.freq_err for mode in modes ])
  chi_2 = self.get('MAP_chi_square')
  K     = len(modes)
  ln_L  = -K/2.0 * np.log(2.0*np.pi) - np.sum(np.log(errs)) - chi_2 

  if utils.has_nan(ln_L):
    logger.error('_set_likelihood: NaN detected in ln_L')
    sys.exit(1)

  L     = utils.substitute_inf(np.exp(ln_L))

  self.set('MAP_ln_likelihood', ln_L)
  self.set('MAP_likelihood', L)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _rescale_ln_probabilities(self):
  """
  Rescale the product of the ln(likelihood) x ln(prior), to avoid numerical underflow when evaluating
  the evidence. Note that one cannot rescale, e.g. only the ln_likelihood, else the weighting of the 
  rescaled ln(likelihood) with unscaled ln(prior) will render the problem as wrong.

  The scaling factor is in natural logarithm. The value of ln_scale is chosen so that the sum of 
  unscaled ln(likelihood) + lin(prior) adds up to zero, after rescaling. Once ln_scale is found, then
  ln(likelihood) and ln(prior) are each rescaled by 0.5 x ln_scale. Subsequently, ln(likelihood), 
  likelihood, ln(prior) and prior are all updated.
  """
  if not self.rescale_ln_probabilities: return 

  ln_P_h       = self.get('MAP_ln_prior')
  ln_P_D_h     = self.get('MAP_ln_likelihood')
  ln_prod      = ln_P_D_h + ln_P_h

  ln_scale     = np.max(ln_P_D_h)
  ln_P_D_h     -= ln_scale

  P_D_h        = utils.substitute_inf(np.exp(ln_P_D_h))

  self.set('MAP_ln_likelihood', ln_P_D_h)
  self.set('MAP_likelihood', P_D_h)

  logger.info('_set_likelihood: rescaling ln(likelihood) x ln(prior) with ln_scale={0:.2f}'.format(ln_scale))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_evidence(self):
  """
  Refer to the documentation below the max_a_posteriori() method for further details
  """
  P_h      = self.get('MAP_prior')      # == P(h)
  P_D_h    = self.get('MAP_likelihood') # P(D|h)
  P_D      = np.dot(P_D_h, P_h)
  if np.isinf(P_D):
    logger.error('_set_evidence: The evidence has diverged!')
    sys.exit(1)

  self.set('MAP_evidence', P_D)
  self.set('MAP_ln_evidence', np.log(P_D))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_posterior(self):
  """
  Refer to the documentation below the max_a_posteriori() method for further details
  """
  ln_prior      = self.get('MAP_ln_prior')
  ln_likelihood = self.get('MAP_ln_likelihood') 
  ln_evidence   = self.get('MAP_ln_evidence')

  ln_posterior  = ln_likelihood + ln_prior - ln_evidence
  posterior     = utils.substitute_inf(np.exp(ln_posterior))

  self.set('MAP_posterior', posterior)
  self.set('MAP_ln_posterior', ln_posterior)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# M A R G I N A L I Z A T I O N   R O U T I N E S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _marginalize_wrt_x_y(self, wrt_x, wrt_y):
  """
  For the full documentation refer to the marginalize_wrt_x_y() method
  """
  _names  = self.get('feature_names')
  if wrt_x not in _names:
    logger.error('_marginalize_wrt_x_y: The column "{0}" is not among the available features'.format(wrt_x))
    sys.exit(1)
  if wrt_y not in _names:
    logger.error('_marginalize_wrt_x_y: The column "{0}" is not among the available features'.format(wrt_y))
    sys.exit(1)
  
  # ndarray of feature tags (not their absolute values)
  _work   = self.get('MAP_work_features')
  dtypes  = [(name, np.int16) for name in _names]
  rec     = utils.ndarray_to_recarray(_work, dtypes)

  set_x   = set(list(rec[wrt_x]))
  set_y   = set(list(rec[wrt_y]))
  nx      = len(set_x)
  ny      = len(set_y)
  arr_x   = np.array(list(set_x))
  arr_x.sort()
  arr_y   = np.array(list(set_y))
  arr_y.sort()
  _grid_xy= np.meshgrid(arr_x, arr_y, indexing='ij')
  all_x   = _grid_xy[0].reshape(-1).astype(np.int16, casting='same_kind')
  all_y   = _grid_xy[1].reshape(-1).astype(np.int16, casting='same_kind')
  tup_xy  = list(zip(all_x, all_y))

  post    = self.get('MAP_posterior')

  tol     = 1e-4
  p_vals  = np.zeros(nx * ny)
  outcome = []
  for i_wrt, _tup in enumerate(tup_xy):
    xtag  = _tup[0]
    ytag  = _tup[1]

    ind   = np.where((rec[wrt_x] == xtag) & (rec[wrt_y] == ytag))[0]
    # ind   = np.where((np.abs(rec[wrt_x] - xval) <= tol) & 
    #                  (np.abs(rec[wrt_y] - yval) <= tol))[0]
    n_ind = len(ind)
    if n_ind == 0: continue

    p_val = np.sum(post[ind])
    tup   = ( xtag, ytag, p_val )
    outcome.extend([ tup ])
    p_vals[i_wrt] = p_val

  try:
    assert np.abs(1.0 - np.sum(p_vals)) < 1e-8
  except:
    logger.error('_marginalize_wrt_x_y: p-values do not sum up to 1 (+/- 1e-8)')
    sys.exit(1)

  nout    = len(outcome)
  logger.info('_marginalize_wrt_x_y: Returning "{0}" points for x:"{1}" and y:"{2}"'.format(nout, wrt_x, wrt_y))
  
  return outcome 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _marginalize_wrt(self, wrt):
  """
  For the full documentation refer to the marginalize_wrt() method.
  """
  x_names = self.get('feature_names')
  if wrt not in x_names:
    logger.error('_marginalize_wrt: The column "{0}" is not among the available features'.format(wrt))
    sys.exit(1)
  
  work_x  = self.get('MAP_work_features')   # ndarray of feature tags (not their absolute values)
  dtypes  = [(name, np.int16) for name in x_names]
  # print(work_x.shape, type(work_x))
  # print(dtypes)
  # sys.exit()
  rec     = utils.ndarray_to_recarray(work_x, dtypes)

  set_wrt = set(list(rec[wrt]))
  n_wrt   = len(set_wrt)
  arr_wrt = np.array(list(set_wrt))
  arr_wrt = np.sort(arr_wrt)
  p_vals  = np.zeros(n_wrt)
  post_wrt= []

  post    = self.get('MAP_posterior')

  for i_wrt, tag_wrt in enumerate(arr_wrt):
    k_wrt = np.where(rec[wrt] == tag_wrt)[0]
    p_val = np.sum(post[k_wrt])
    tup   = (tag_wrt, p_val)

    p_vals[i_wrt] = p_val
    post_wrt.extend([ tup ])

  try:
    assert np.abs(1.0 - np.sum(p_vals)) < 1e-8
  except:
    logger.error('_marginalize_wrt: p-values do not sum up to 1 (+/- 1e-8)')
    sys.exit(1)

  n_post  = len(post_wrt)
  logger.info('_marginalize_wrt: Returning "{0}" points for wrt:"{1}"'.format(n_post, wrt))

  return post_wrt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _marginalize(self):
  """
  For the full documentation refer to the marginalize() method.
  """
  names  = self.get('feature_names')
  n_names= len(names)
  result = []
  vals   = np.zeros(n_names)

  for i, name in enumerate(names):
    res  = self.marginalize_wrt(wrt=name)
    result.append(res)

    par  = np.array([_tup[0] for _tup in res])
    marg = np.array([_tup[1] for _tup in res])
    ind  = np.argmax(marg)
    vals[i] = par[ind] # res[0][ np.argmax(res[1]) ]

  self.set('marginal_results', result)
  self.set('marginal_features', vals)
  self.set('marginalize_done', True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _sort_by_posterior(self):
  """
  For detailed documentation, please refer to the docs below the method sort_by_posterior().
  """
  ln_posterior = self.get('MAP_ln_posterior')
  ind          = np.argsort(ln_posterior)[::-1]
  self.set('MAP_prior', self.MAP_prior[ind])
  self.set('MAP_ln_prior', self.MAP_ln_prior[ind])
  self.set('MAP_likelihood', self.MAP_likelihood[ind])
  self.set('MAP_ln_likelihood', self.MAP_ln_likelihood[ind])
  self.set('MAP_posterior', self.MAP_posterior[ind])
  self.set('MAP_ln_posterior', self.MAP_ln_posterior[ind])

  logger.info('_sort_by_posterior: all arrays sorted by ln_posterior (in decreasing order)')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# G E N E R I C   R O U T I N E S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

