import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################

  # Define ReLU
  ReLU = lambda x: np.maximum(np.zeros(x.shape), x)

  # Long way
  # # Concatenate W1 and b1
  # W1b1 = np.vstack((W1, b1))
  #
  # # Add ones to X for bias
  # X_1 = np.hstack((X, np.ones((N, 1))))
  #
  # # Dot product X1 * W1b1 with ReLU activation
  # h1 = ReLU(X_1.dot(W1b1))
  #
  # # Concatenate W2 and b2
  # W2b2 = np.vstack((W2, b2))
  #
  # # Add ones to h1 for bias
  # h1_1 = np.hstack((h1, np.ones((h1.shape[0], 1))))
  #
  # # Dot product h1_1 * W2b2
  # h2 = h1_1.dot(W2b2)

  # Succinct way
  h1 = X.dot(W1) + b1
  h1_activation = ReLU(h1)
  h2 = h1_activation.dot(W2) + b2

  # Scores are the output of the second layer without activation
  scores = h2

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################

  ############
  # Data Loss
  ############
  # Avoid numeric instability
  scores -= np.max(scores)

  # Compute unnormalized probabilities
  exp_scores = np.exp(scores)

  # Compute the normalized probabilities
  normalized_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # Cross-entropy loss is the log probabilities of the correct class
  log_probs = -np.log(normalized_probs[np.arange(N), y])

  # Average probabilities to get final data loss
  data_loss = np.average(log_probs)

  ######################
  # Regularization Loss
  ######################
  # Square weights
  W1_squared = np.square(W1)
  W2_squared = np.square(W2)

  # Sum across first dimension
  W1_sum = np.sum(W1_squared, axis=1)
  W2_sum = np.sum(W2_squared, axis=1)

  # Sum across second dimension yields L2 regularization
  W1_L2 = np.sum(W1_sum, axis=0)
  W2_L2 = np.sum(W2_sum, axis=0)

  # Sum L2 regularization for both weights
  L2_regularization = W1_L2 + W2_L2

  # Multiply by lambda
  regularization_loss = reg * L2_regularization

  #############
  # Final Loss
  #############
  loss = data_loss + (regularization_loss * 0.5)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}

  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################

  # backprop softmax (see http://cs231n.github.io/neural-networks-case-study/)
  dscores = normalized_probs
  dscores[np.arange(N), y] -= 1
  dscores /= N

  # backprop scores (h2) = h1_activation.dot(W2) + b2
  dW2 = h1_activation.T.dot(dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)

  # backprop from second layer
  dh1 = dscores.dot(W2.T)
  dh1[h1_activation <= 0] = 0 # ????

  # backprop h1 = X.dot(W1) + b1
  dW1 = X.T.dot(dh1)
  db1 = np.sum(dh1, axis=0, keepdims=True)

  # regularization loss backprop
  dW1 += reg * W1
  dW2 += reg * W2

  # store gradients
  grads['W1'] = np.squeeze(dW1)
  grads['b1'] = np.squeeze(db1)
  grads['W2'] = np.squeeze(dW2)
  grads['b2'] = np.squeeze(db2)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads
