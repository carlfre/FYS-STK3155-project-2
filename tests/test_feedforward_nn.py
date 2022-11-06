from feedforward_nn import NeuralNetwork

import numpy as np

#print working directory

import os
print(os.getcwd())

def test_initialization():
    nn = NeuralNetwork([3, 2, 6, 8, 1], "relu")
    assert len(nn.weights) == 4
    assert len(nn.biases) == 4

    assert nn.weights[0].shape == (3, 2)
    assert nn.weights[1].shape == (2, 6)
    assert nn.weights[2].shape == (6, 8)
    assert nn.weights[3].shape == (8, 1)

    assert nn.biases[0].shape == (1, 2)
    print(nn.biases[0])
    assert nn.biases[1].shape == (1, 6)
    assert nn.biases[2].shape == (1, 8)
    assert nn.biases[3].shape == (1, 1)


def test_flatten_weights():
    nn = NeuralNetwork([2, 2, 2, 1], "relu")
    nn.weights = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10]])]
    nn.biases = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5]])]
    wb = nn.wb()
    assert np.array_equal(wb, np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]]).T)


def test_unflatten_weights():
    nn = NeuralNetwork([2, 2, 2, 1], "relu")
    wb = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]]).T
    weights, biases = nn.unflatten_weights_and_biases(wb)
    print(weights)
    print(biases)
    assert np.array_equal(weights[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(weights[1], np.array([[5, 6], [7, 8]]))
    assert np.array_equal(weights[2], np.array([[9, 10]]).T)
    assert np.array_equal(biases[0], np.array([[1, 2]]))
    assert np.array_equal(biases[1], np.array([[3, 4]]))
    assert np.array_equal(biases[2], np.array([[5]]))


def test_forward_propagation():
    nn = NeuralNetwork([2, 2, 2, 1], "relu")
    nn.weights = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10]])]
    nn.biases = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5]])]
    X = np.array([[1, 2], [3, 4]])
    wb = nn.wb()

    activations, zs = None, None#nn._forward_propagation(X)
    assert np.array_equal(activations[0], X)
    assert np.array_equal(activations[1], np.array([[13, 16], [29, 36]]))
    assert np.array_equal(activations[2], np.array([[69, 80], [157, 184]]))
    assert np.array_equal(activations[3], np.array([[389, 410], [901, 952]]))
    assert np.array_equal(zs[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(zs[1], np.array([[13, 16], [29, 36]]))
    assert np.array_equal(zs[2], np.array([[69, 80], [157, 184]]))
    assert np.array_equal(zs[3], np.array([[389, 410], [901, 952]]))






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

