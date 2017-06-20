
"""
This module provides a basic Artificial Neural Network to build iterative learning routines, and
carry out e.g. feedforward and back propagation learning of the model frequencies to infer the 
model attributes from the model frequencies, together with their uncertainties. The neural_network
class is derived from the machine_learning.learner class, and extends its functionalities.

Some of the functionalities are developed following strictly the chapter 18 in the following book:
   Theodoridis Sergios, 2015, "Machine Learning. A Baysian and optimization perspective", Elsevier
In such functions/methods, the reference is provided to the relevant equation in the book, e.g. 
TS:12.34, meaning the equation 34 from chapter 12 of the Theodoridis book.
"""

import sys, os, glob
import logging
import numpy as np 

from asamba import utils, star, sampler
from asamba import machine_learning as ml

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logger  = logging.getLogger(__name__)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    ###  ### ######  ###     ###   #####
    #    #    #    #  #     #  #       #   #     #
    #     #   #    #  #     #  #       #  #
    #    #    #    #  ######   #       #  #
    #####     #    #  #     #  #       #  #
    #         #    #  #     #  #    #  #   #     #
    #          ####   ######  ####### ###   #####


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class neural_network(object):
  """
  This class extends the machine_learning.learner() class by stacking N neural layers where each layer
  has n_neurons number of neurons. The architecture of the network follows closely the instructions 
  given in chapter 18 of the ST book. All neural layers present in this network are stacked after one 
  another in a list (called layers). 

  The idea is that each neural net
  """
  def __init__(self, n_neurons_per_layer, inputs, outputs):
    """
    Instantiate a neural network by providing the number of neurons per each layer. The total number of
    layers (input, hidden, output) is determined by the length of the input list/tuple. Obviousely, 
    the first (last) element of the list determines the number of neurons in the first (last) layer, 
    and the rest determine the hidden layers.

    @param n_neurons_per_layer: iterable which specifies the number of neurons per each layer.
    @type n_neurons_per_layer: list/tuple
    """
    # super(neural_network, self).__init__()

    #.............................
    # Check input type
    #.............................
    self.set_init_argument(n_neurons_per_layer, 'n_neurons_per_layer')
    self.set_init_argument(inputs, 'inputs')
    self.set_init_argument(outputs, 'outputs')

    # Instantiate the layers, with the input layer standing upfront
    self.n_layers = len(self.n_neurons_per_layer)

    #.............................
    # Responses of neurons in layers
    #.............................
    # List of all reponses, y=f(z)
    self.responses   = []
    # List of all response derivatives, df(z)/dz
    self.derivatives = []

    #.............................
    # Interconnection weights 
    #.............................
    # Shape tuples (None for input layer)
    self.weight_shapes = []
    # STD of the initial normally distributed weights
    self.std_initial_weights = 0.10
    # Weights
    self.weights = []

    #.............................
    # Linear combiners z_{nj}^{r}
    #.............................
    # List of combiners (TS:18.17)
    # (None for input layer, else ndarray)
    self.combiners = []

    #.............................
    # Errors
    #.............................
    # The difference between expected and 
    # actual neuron outputs
    self.errors    = []
    # The weight correction integrands delta_{nj}^{r}
    self.deltas    = []
    # Corrections to the weights for one training example
    self.corrections = []

    #.............................
    # Cost
    #.............................
    # The cost for the n-th training example
    self.cost      = 0.0

    #.............................
    # Preparation & Initialization
    #.............................
    # status of the initialization step
    self.allocate_responses_done = False
    self.initialize_weights_done = False
    self.initialization_done     = False

    #.............................
    # Global network constants
    #.............................
    # The (initial) step size
    self.step_size_mu   = 0.01
    # The scale constant of the sigmoid function
    self.sigmoid_a      = 1.0

    logger.info('neural_network: initialized.')

  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    # super(neural_network, self).set(attr, val)
    if not hasattr(self, attr):
      logger.error('neural_network: does not have the attribute: "{0}"'.format(attr))
      sys.exit(1)

    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    # super(neural_network, self).get(attr)
    if not hasattr(self, attr):
      logger.error('neural_network: get: Attribute "{0}" is unavailable.')
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def set_init_argument(self, arg, name):
    """
    Check and set the input argument to __init__ is an iterable (list, tuple or ndarray), and if it 
    is not an ndarray, convert to it.

    @param self: an instance of the neural_network() class
    @type self: object
    @param arg: The iterable argument used to initialize the neural_network() object
    @type: arg: list, tuple or ndarray
    @param name: The name of the argument, as found in the argumet list of the __init__ method
    @type name: str
    """
    _set_init_argument(self, arg, name)

  ##########################
  def initialize(self):
    """
    Follow the initialization steps in Algorithm 18.3 in the TS:page891, as the following:
    - initialize all synaptic weights and biases randomly with small, but not very small values.
    - select step size, mu
    - set the inputs of the n-th training example, where n=1, ..., N
    """
    if self.initialization_done: return 

    _allocate_responses(self)
    _shape_the_weights(self)
    _initialize_weights(self)
    _initialize_combiners(self)
    _initialize_derivatives(self)
    _initialize_corrections(self)
    _allocate_errors(self)
    _set_inputs(self)
    _set_outputs(self)

    status = all([self.allocate_responses_done, self.initialize_weights_done]) 
    if status:
      self.set('initialization_done', True)
      logger.info('initialize: done.')
    else:
      logger.error('initialize: failed.')
      sys.exit(1)

  ##########################
  def feedforward(self):
    """
    Based on the TS18.3 algorithm, the feedforward step for one given example set (x_n, y_n) consists
    of two steps, the computation of the combiners and the responses for all neurons in all layers. 
    This method carries out both these steps, respectively.
    """
    _calculate_combiners(self)
    _calculate_response(self)

  ##########################
  def backpropagate(self):
    """ 
    Based on TS18.3 algorithm, the backpropagation consists of the following steps, which this method 
    applies, respectively:
    - computation of the delta for the output layer
    - backward computation of all deltas for hidden layers
    """
    _calculate_delta_output_layer(self)
    _backpropagate_deltas(self)
    _calculate_corrections(self)
    _calculate_cost(self)

  ##########################


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class neural_training(sampler.sampling):
  """
  This class builds up on the sampler.sampling class and has access to the whole learning dataset there.
  It integrates also with the neural_network class in order to import the architecture of the network from
  there, and carry out the training one per example at a time.
  """
  def __init__(self, n_neurons_per_layer):
    """
    Initialize an instance of the neural_training class after providing the architecture of the neural 
    network (i.e. the number of neurons per each neural layer).
    """
    super(neural_training, self).__init__()
    self.n_neurons_per_layer = n_neurons_per_layer

    #.............................
    # The learning data to be normalized
    #.............................
    self.work_inputs  = []
    self.work_outputs = []

    #.............................
    # Cost
    #.............................
    self.cost = 0

  ##########################
  # Setter
  ##########################
  def set(self, attr, val):
    super(neural_training, self).set(attr, val)
    if not hasattr(self, attr):
      logger.error('neural_training: does not have the attribute: "{0}"'.format(attr))
      sys.exit(1)

    setattr(self, attr, val)

  ##########################
  # Getter
  ##########################
  def get(self, attr):
    super(neural_training, self).get(attr)
    if not hasattr(self, attr):
      logger.error('neural_training: get: Attribute "{0}" is unavailable.')
      sys.exit(1)

    return getattr(self, attr)

  ##########################
  # Methods
  ##########################
  def run_one(self):
    """
    Feed the neural network with only one example row from the learning/training dataset, and get the 
    cost and the weight corrections back from that single example.
    """
    _run_one(self)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #####    #####    ###  ###  ###    ###    ###########  ######### 
    #    #   #    #    #    #    #     # #    #    #    #  #       #
    #     #  #     #   #    #    #     # #         #       #
    #    #   #    #    #     #  #     #   #        #       ####
    #####    #####     #     #  #     #####        #       #
    #        #  #      #      #      #     #       #       #       #
    #        #    #   ###    ###    ###   ###      #       #########

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   T H E   N E U R A L   N E T W O R K   C L A S S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _calculate_cost(self):
  """
  For the n-the training example, the corresponding cost after the forward step is the difference 
  between the ground-truth (self.outputs), and the predicted output at the output layer (L), as
  in TS:18.23
  """
  e_L = self.get('errors')[-1]
  J_n = np.sum(e_L * e_L) / 2.0

  self.set('cost', J_n)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _calculate_corrections(self):
  """
  Following TS:18.21, the correction for each interconnection weight is basically the inner product
  of the deltas in layer r with the response from layer r-1. 
  """
  responses   = self.get('responses')
  deltas      = self.get('deltas')
  corrections = self.get('corrections')
  n_layers    = self.get('n_layers')

  for r in range(1, n_layers):
    delta   = deltas[r]             # k_r 
    y       = responses[r-1]        # 1 + k_{r-1}
    corr    = np.outer(delta, y)    # (k_r, 1 + k_{r-1})
    self.corrections[r][:,:] = corr[:,:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _backpropagate_deltas(self):
  """
  Walk through the layers backwards, starting from the output layer until the 2nd layer, and update
  the errors and deltas for the hidden layers. Here you hear the backpropagation's heartbeat. See also
  TS18.32
  """
  weights = self.get('weights')
  errors  = self.get('errors')
  deltas  = self.get('deltas')
  deriv   = self.get('derivatives')
  n_layers= self.get('n_layers')
  ind     = range(2, n_layers)[::-1]

  for k, r in enumerate(ind):
    delta = deltas[r]               # k_r
    weight= weights[r][:,1:]        # (k_r, k_{r-1}) : here we have excluded the bias terms
    dfz   = deriv[r-1]              # k_{r-1}
    error = np.dot(delta, weight)
    self.errors[r-1][:] = error[:]  # k_{r-1}
    delta_= error * dfz 
    self.deltas[r-1][:] = delta_[:] # k_{r-1}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _calculate_delta_output_layer(self):
  """
  The error at the output layer is basically the difference between the expected and computed 
  response of the neurons. For the last layer in the network see TS:18.24
  """
  y_L = self.get('responses')[-1][1:]   # excluding the constant term in the zeroth element
  y   = self.get('outputs')
  e_L = y_L - y                         # k_L
  self.errors[-1][:] = e_L[:]          
  dz  = self.get('derivatives')[-1][:] 
  d_L = e_L * dz
  self.deltas[-1][:] = d_L[:]           # excluding the constant term in the zeroth element

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _calculate_combiners(self):
  """
  Following TS:18.17, the linear combiners z_{nj}^{r} are computed by a dot product between the 
  weights (from left), and the response of the preceeding layer (from right). Because the input layer
  by definition has no combiner, we put a None instead.

  As a return, the self.combiners will be allocated with a list of combiners (None or 1D ndarray) for 
  all layers in the network.
  """
  n_layers  = self.get('n_layers')
  responses = self.get('responses')
  weights   = self.get('weights')
  combiners = self.get('combiners')

  for r in range(1, n_layers):  # sizes are commented below
    theta   = weights[r]        # k_r x (1 + k_{r-1})
    y       = responses[r-1]    # (1 + k_{r-1})
    z       = np.dot(theta, y)  # k_r
    self.combiners[r][:] = z[:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _calculate_response(self):
  """
  The response of the hidden layers (i.e. all layers excluding the 0-th/input layer) is the activation
  of each neuron given its combiner as an input argument. Because for each layer, the combiners are 
  accumulated as ndarray vectors, we calculate the activation value of the layer by applying the activation
  function to the array of combiners for each layer. As a return, list of responses will be updated, except
  for the 0-th/input array.

  On top of the responses, we get the derivatives of each response w.r.t the combiner (df/dz) for free,
  which we store as self.derivatives attribute. These derivatives are used during the backpropagation
  to estimate the errors of each neuron.
  """
  n_layers  = self.get('n_layers')
  combiners = self.get('combiners')
  a         = self.get('sigmoid_a')

  for r in range(1, n_layers):
    z       = combiners[r]            # k_r
    fz, dfz = sigmoid(a, z)           # k_r
    self.responses[r][1:]   = fz[:]
    self.derivatives[r][:] = dfz[:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_init_argument(self, arg, name):
  """
  Refer to set_init_argument() method for further details
  """
  is_list    = isinstance(arg, list)
  is_tuple   = isinstance(arg, tuple)
  is_ndarray = isinstance(arg, np.ndarray)

  is_iterable= any([is_list, is_tuple, is_ndarray])
  if not is_iterable:
    logger.error('_check_init_argument: "{0}" argument is not iterable'.format(name))
    sys.exit(1)

  if not is_ndarray: 
    setattr(self, name, np.array(arg))
  else:
    setattr(self, name, arg)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_inputs(self):
  """
  Set the input for the n-th training example. The response from each layer has a mandatory first
  element of value unity; therefore, the input y is put starting from the 1-st element of the response
  vector until the end, i.e. response[1:] = y[:]
  """
  if not self.allocate_responses_done: return False
  y = self.get('inputs')
  n = self.get('n_neurons_per_layer')[0]
  if len(y) != n:
    logger.error('_set_inputs: The "inputs" must have {0} elements (for {0} neurons)'.format(n))
    sys.exit(1)
  self.responses[0][1:] = y[:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _set_outputs(self):
  """
  Set the output for the n-th training example. The response from each layer has a mandatory first
  element of value unity; therefore, the output y is put starting from the 1-st element of the response
  vector until the end, i.e. response[1:] = y[:]
  """
  if not self.allocate_responses_done: return False
  y = self.get('outputs')
  n = self.get('n_neurons_per_layer')[-1]
  if len(y) != n:
    logger.error('_set_inputs: The "outputs" must have {0} elements (for {0} neurons)'.format(n))
    sys.exit(1)
  self.responses[-1][1:] = y[:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _allocate_responses(self):
  """
  The response from each layer with "n" neurons is a vector of "1 + n" elements, where the zeroth 
  element is forced to be unity. This routine, walks through the number of requested neurons per 
  layer, and allocates the appropriate size and value for the response vectors. 
  As a return, the responses will be stored as a list of ndarrays to the self.responses attribute
  """
  nnpl = self.get('n_neurons_per_layer')
  responses = []
  for k, n in enumerate(nnpl):
    response = np.zeros(1 + n)
    response[0] = 1.0
    responses.append(response)

  self.set('responses', responses)
  self.set('allocate_responses_done', True)

  logger.info('_allocate_responses: done')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _allocate_errors(self):
  """
  The "errors" and "deltas" are basically vectors for each neural layer. They are allocated and set
  to zeros in this function, and their proper values will be iteratively modified per each iteration
  during the backpropagation procedure.
  """
  nnpl    = self.get('n_neurons_per_layer')
  errors  = []
  deltas  = []
  for k, n in enumerate(nnpl):
    errors.append(np.zeros(n)) # including the bias term
    deltas.append(np.zeros(n)) # including the bias term

  self.set('errors', errors)
  self.set('deltas', deltas)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _initialize_derivatives(self):
  """
  The derivatives of the activation of each neural layer is stored as an array of length n+1, where
  n is the number of neurons in that specific layer. Note that the input layer has no derivative for
  the activation function, hence "None" is put in that place.
  """
  nnpl    = self.get('n_neurons_per_layer')
  n_layers= self.get('n_layers')

  derivatives = []
  for r, n in enumerate(nnpl):
    if r == 0:
      der = None 
    else: 
      der = np.zeros(n)
    derivatives.append(der)

  self.set('derivatives', derivatives)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _initialize_combiners(self):
  """
  Allocate and set the initial value of the combiners (TS:18.17) to zero. Note that the first (input)
  layer has no combiner, hence we place None for that. The combiners have a length n, for a layer 
  with n neurons. 
  """
  nnpl    = self.get('n_neurons_per_layer')
  combiners = []
  for k, n in enumerate(nnpl):
    if k == 0:
      cmb = None 
    else:
      cmb = np.zeros(n)
    combiners.append(cmb)

  self.set('combiners', combiners)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _shape_the_weights(self):
  """
  Set the shapes of the layer weights based on the number of neurons in that layer, and the number of 
  neurons in the preceeding layer. The weights will be randomized and initialized later (after a call 
  to the _initialize_weights() method). This method specifies the shape (tuple) of the weight matrixes
  for each layer. Noting that the zeroth (input) layer has no weight, we put None in place.

  As a return, a list of weight shapes (list of tuples) will be returned where each tuple specifies 
  the weight shape for that layer, with the first element set to None.

  The weights (row arrays) are stacked along the axis=0, so that they look like the following
      [[bias_1, w_11, w_12, ..., w_1,k_{r-1}],
       [bias_2, w_21, w_22, ..., w_2,k_{r-1}],
       ...,
       [bias_k_{r}, w_k_{r},1, w_k_{r},2, ..., w_k_{r},k_{r-1}]]
  Therefore, for the r-th layer with k_{r} neurons, the weight matrix has size k_{r} x (k_{r-1}+1), 
  where k_{r-1} is the number of neurons in the preceeding layer.
  """
  nnpl    = self.get('n_neurons_per_layer') # iterable
  shapes  = []
  n_shapes= 0
  for k, n in enumerate(nnpl):
    if k  == 0: 
      shape = None 
    else:
      prev  = nnpl[k-1]
      shape = (n, 1 + prev)
      n_shapes += n * (1 + prev)
    shapes.append(shape)

  self.set('weight_shapes', shapes)
  logger.info('_shape_the_weights: Total weight elements = {0}'.format(n_shapes))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _initialize_weights(self):
  """
  Set the weights to randomly distributed values with zero mean and small variance. The input layer 
  has no weights, and hence is set to None. As a return, the self.weights attribute is set to a list
  of None or 2D ndarrays for layer weights.
  """
  weights = []
  shapes  = self.get('weight_shapes')
  std     = self.get('std_initial_weights')

  for k, shape in enumerate(shapes):
    if shape is None: 
      w   = None 
    else:
      w = np.random.normal(loc=0.0, scale=std, size=shape)
    weights.append(w)

  self.set('weights', weights)
  self.set('initialize_weights_done', True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _initialize_corrections(self):
  """
  Set the corrections for weights of each neural network to ndarrays with zero values, to make sure
  the size stays consistent during all iterations. 
  """
  # nnpl = self.get('n_neurons_per_layer')
  shapes = self.get('weight_shapes')

  corrections = []
  for k, shape in enumerate(shapes):
    if k == 0: 
      corr = None
    else:
      corr = np.zeros(shape)  # 1 for correcting the bias term
    corrections.append(corr)

  self.set('corrections', corrections)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _adapt_step_size(self):
  """
  Select a fixed step size or an adaptable step size
  """
  mu = self.get('step_size_mu')
  self.set('step_size_mu', mu)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R O U T I N E S   F O R   T H E    E P O C H   C L A S S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# G E N E R A L   P U R P O S E   R O U T I N E S
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sigmoid(a, z):
  """
  Return the sigmoid function (ST:18.9)
     f(a, z) = (1 + exp(-a z))^{-1}
  together with its derivative (ST:18.33)
     df(a, z)/dz = a f(a, z) (1 - f(a, z)) 
  @param a: sigmoid scale parameter
  @type a: float
  @param z: sigmoid argument
  @type z: float
  @return: tuple with the value of the sigmoid as the first element, and its derivative as the second
        element
  @rtype: tuple
  """
  f  = 1.0 / (1.0 + np.exp(-a * z))
  df = a * f * (1.0 - f)

  return (f, df)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def normalize(u):
  """
  Normalize u based on the following formula, so that after normalization -1 <= u <= +1. We designate
  the normalized form of the u with w:
     w = [2 * u - (b + a)] / (b - a)
  where "b" and "a" are the maximum and minimum values of the u, respectively.

  @param u: the array to normalize. It can be a vector or a matrix
  @type u: np.ndarray
  @return: tuple of three quantities: the minimum "a", the maximum "b" and the normalized array "w".
  @rtype: tuple
  """
  a = np.min(u)
  b = np.max(u)
  w = (2 * u - (b + a)) / (b - a)

  return (a, b, w)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def denormalize(w, a, b):
  """
  Denormalize the input array "w", given minimum "a" and maximum "b" values of the outcome array. This
  function complements the normalize() method.
  """
  try:
    assert b > a
  except AssertionError:
    logger.error('_denormalize: b must be strictly greater than a.')
    sys.exit(1)

  return (b - a) * (w + 1) / 2.0 + a

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



