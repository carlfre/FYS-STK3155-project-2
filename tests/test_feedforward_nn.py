from feedforward_nn import NeuralNetwork

import numpy as np

#print working directory

import os
print(os.getcwd())

def test_flatten_weights():
    nn = NeuralNetwork([2, 2, 2, 1], "relu")
    nn.weights = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10]])]
    nn.biases = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5]])]
    wb = nn.flatten_weights_and_biases()
    assert np.array_equal(wb, np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]]).T)

#def test_gradient_2_layers():
#    """Computes analytical gradient for a 3-layer network and compares it to
#    numerical gradient.
#    """
#    layers = [2, 2]
#    nn = NeuralNetwork(layers, "relu")
#    W = nn.weights
#    b = nn.biases
#    wb = nn.wb()
#
#    X = np.array([[1, 2], [3, 4]])
#    y = np.array([[1], [0]])

