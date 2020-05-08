# import tensorflow as tf
# from tensorflow import keras
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import sklearn
# import sklearn.datasets
# from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
# from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec
# #
#
# plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# # load image dataset: blue/red dots in circles
# train_X, train_Y, test_X, test_Y = load_dataset()
#
# def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
#     """
#     Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
#
#     Arguments:
#     X -- input data, of shape (2, number of examples)
#     Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
#     learning_rate -- learning rate for gradient descent
#     num_iterations -- number of iterations to run gradient descent
#     print_cost -- if True, print the cost every 1000 iterations
#     initialization -- flag to choose which initialization to use ("zeros","random" or "he")
#
#     Returns:
#     parameters -- parameters learnt by the model
#     """
#
#     grads = {}
#     costs = [] # to keep track of the loss
#     m = X.shape[1] # number of examples
#     layers_dims = [X.shape[0], 10, 5, 1]
#
#     # Initialize parameters dictionary.
#     if initialization == "zeros":
#         parameters = initialize_parameters_zeros(layers_dims)
#     elif initialization == "random":
#         parameters = initialize_parameters_random(layers_dims)
#     elif initialization == "he":
#         parameters = initialize_parameters_he(layers_dims)
#
#     # Loop (gradient descent)
#
#     for i in range(0, num_iterations):
#
#         # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
#         a3, cache = forward_propagation(X, parameters)
#
#         # Loss
#         cost = compute_loss(a3, Y)
#
#         # Backward propagation.
#         grads = backward_propagation(X, Y, cache)
#
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
#
#         # Print the loss every 1000 iterations
#         if print_cost and i % 1000 == 0:
#             print("Cost after iteration {}: {}".format(i, cost))
#             costs.append(cost)
#
#     # plot the loss
#     plt.plot(costs)
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per hundreds)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
#
#     return parameters
#
# # GRADED FUNCTION: initialize_parameters_he
#
# def initialize_parameters_he(layers_dims):
#     """
#     Arguments:
#     layer_dims -- python array (list) containing the size of each layer.
#
#     Returns:
#     parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
#                     W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
#                     b1 -- bias vector of shape (layers_dims[1], 1)
#                     ...
#                     WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
#                     bL -- bias vector of shape (layers_dims[L], 1)
#     """
#
#     np.random.seed(3)
#     parameters = {}
#     L = len(layers_dims) - 1 # integer representing the number of layers
#
#     for l in range(1, L + 1):
#         ### START CODE HERE ### (≈ 2 lines of code)
#         parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
#         parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
#         ### END CODE HERE ###
#
#     return parameters


# parameter = model(train_X, train_Y, initialization = "he")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameter)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# Regularization

# import packages
# import numpy as np
# import matplotlib.pyplot as plt
# from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
# from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
# from reg_utils import compute_cost_with_regularization, backward_propagation_with_regularization
# from reg_utils import forward_propagation_with_dropout, backward_propagation_with_dropout
#
# import sklearn
# import sklearn.datasets
# import scipy.io
# from testCases import *
# from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec
# #
#
# plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# # load image dataset: blue/red dots in circles
# train_X, train_Y, test_X, test_Y = load_dataset()
#
#
#
# def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
#     """
#     Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
#
#     Arguments:
#     X -- input data, of shape (input size, number of examples)
#     Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
#     learning_rate -- learning rate of the optimization
#     num_iterations -- number of iterations of the optimization loop
#     print_cost -- If True, print the cost every 10000 iterations
#     lambd -- regularization hyperparameter, scalar
#     keep_prob - probability of keeping a neuron active during drop-out, scalar.
#
#     Returns:
#     parameters -- parameters learned by the model. They can then be used to predict.
#     """
#
#     grads = {}
#     costs = []                            # to keep track of the cost
#     m = X.shape[1]                        # number of examples
#     layers_dims = [X.shape[0], 20, 3, 1]
#
#     # Initialize parameters dictionary.
#     parameters = initialize_parameters(layers_dims)
#
#     # Loop (gradient descent)
#
#     for i in range(0, num_iterations):
#
#         # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
#         if keep_prob == 1:
#             a3, cache = forward_propagation(X, parameters)
#         elif keep_prob < 1:
#             a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
#
#         # Cost function
#         if lambd == 0:
#             cost = compute_cost(a3, Y)
#         else:
#             cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
#
#         # Backward propagation.
#         assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout,
#                                             # but this assignment will only explore one at a time
#         if lambd == 0 and keep_prob == 1:
#             grads = backward_propagation(X, Y, cache)
#         elif lambd != 0:
#             grads = backward_propagation_with_regularization(X, Y, cache, lambd)
#         elif keep_prob < 1:
#             grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
#
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
#
#         # Print the loss every 10000 iterations
#         if print_cost and i % 10000 == 0:
#             print("Cost after iteration {}: {}".format(i, cost))
#         if print_cost and i % 1000 == 0:
#             costs.append(cost)
#
#     # plot the cost
#     plt.plot(costs)
#     plt.ylabel('cost')
#     plt.xlabel('iterations (x1,000)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
#
#     return parameters

#
# parameter = model(train_X, train_Y)
# print ("On the training set:")
# predictions_train = predict(train_X, train_Y, parameter)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameter)


# parameter = model(train_X, train_Y, lambd = 0.2)
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameter)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameter)
#
# parameter = model(train_X, train_Y, keep_prob = 0.8, learning_rate = 0.3)
#
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameter)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameter)

# Packages

# Gradient Checking
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters

#
#
# # GRADED FUNCTION: forward_propagation
#
# def forward_propagation(x, theta):
#     """
#     Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
#
#     Arguments:
#     x -- a real-valued input
#     theta -- our parameter, a real number as well
#
#     Returns:
#     J -- the value of function J, computed using the formula J(theta) = theta * x
#     """
#
#     ### START CODE HERE ### (approx. 1 line)
#     J = theta * x
#     ### END CODE HERE ###
#
#     return J
#
# # GRADED FUNCTION: backward_propagation
#
# def backward_propagation(x, theta):
#     """
#     Computes the derivative of J with respect to theta (see Figure 1).
#
#     Arguments:
#     x -- a real-valued input
#     theta -- our parameter, a real number as well
#
#     Returns:
#     dtheta -- the gradient of the cost with respect to theta
#     """
#
#     ### START CODE HERE ### (approx. 1 line)
#     dtheta = x
#     ### END CODE HERE ###
#
#     return dtheta
#
# # GRADED FUNCTION: gradient_check
#
# def gradient_check(x, theta, epsilon = 1e-7):
#     """
#     Implement the backward propagation presented in Figure 1.
#
#     Arguments:
#     x -- a real-valued input
#     theta -- our parameter, a real number as well
#     epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
#
#     Returns:
#     difference -- difference (2) between the approximated gradient and the backward propagation gradient
#     """
#
#     # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
#     ### START CODE HERE ### (approx. 5 lines)
#     thetaplus = theta + epsilon                               # Step 1
#     thetaminus = theta - epsilon                              # Step 2
#     J_plus = forward_propagation(x, thetaplus)                                  # Step 3
#     J_minus = forward_propagation(x, thetaminus)                                 # Step 4
#     gradapprox = (J_plus - J_minus) / (2 * epsilon)                              # Step 5
#     ### END CODE HERE ###
#
#     # Check if gradapprox is close enough to the output of backward_propagation()
#     ### START CODE HERE ### (approx. 1 line)
#     grad = backward_propagation(x, theta)
#     ### END CODE HERE ###
#
#     ### START CODE HERE ### (approx. 1 line)
#     numerator = np.linalg.norm(grad - gradapprox)                               # Step 1'
#     denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                            # Step 2'
#     difference = numerator / denominator                              # Step 3'
#     ### END CODE HERE ###
#
#     if difference < 1e-7:
#         print ("The gradient is correct!")
#     else:
#         print ("The gradient is wrong!")
#
#     return difference

# x_test, theta_test = 2, 4
# diff = gradient_check(x_test, theta_test)
# print("difference = " + str(diff))

# GRADED FUNCTION: gradient_check_n

























































