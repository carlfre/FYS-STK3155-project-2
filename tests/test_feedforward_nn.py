from feedforward_nn import NeuralNetwork

import numpy as np


def test_initialization():
    nn = NeuralNetwork([3, 2, 6, 8, 1], "relu")
    assert len(nn.weights) == 4
    assert len(nn.biases) == 4

    assert nn.weights[0].shape == (3, 2)
    assert nn.weights[1].shape == (2, 6)
    assert nn.weights[2].shape == (6, 8)
    assert nn.weights[3].shape == (8, 1)

    assert nn.biases[0].shape == (1, 2)
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
    assert np.array_equal(weights[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(weights[1], np.array([[5, 6], [7, 8]]))
    assert np.array_equal(weights[2], np.array([[9, 10]]).T)
    assert np.array_equal(biases[0], np.array([[1, 2]]))
    assert np.array_equal(biases[1], np.array([[3, 4]]))
    assert np.array_equal(biases[2], np.array([[5]]))


# TODO: add tests for activation functions (particularly derivatives)
def test_relu():
    pass


def test_sigmoid():
    pass


def test_leaky_relu():
    pass


def test_forward_propagation():
    X = np.array([[1, 2]])
    nn = NeuralNetwork([2, 2, 2, 1], "sigmoid")

    nn.weights = [np.array([[1, 2], [3, 4]]) * 0.1, np.array([[5, 6], [7, 8]]) * 0.1, np.array([[9, 10]]).T * 0.1]
    nn.biases = [np.array([[1, 2]])* 0.1, np.array([[3, 4]]) * 0.1, np.array([[5]]) * 0.1]
    nn.X = X

    activations, zs = nn._forward_propagation()

    atol = 1e-5
    # Compare with hand-caluclated values
    assert np.allclose(zs[0], X, atol=atol)
    assert np.allclose(activations[0], X, atol=atol)
    assert np.allclose(zs[1], np.array([[0.8, 1.2]]), atol=atol)
    assert np.allclose(activations[1], np.array([[0.68997448, 0.76852478]]), atol=atol)
    assert np.allclose(zs[2], np.array([[1.182954586, 1.428804512]]), atol=atol)
    assert np.allclose(activations[2], np.array([[0.76547863027, 0.80671497654]]), atol=atol)
    assert np.allclose(zs[3], np.array([[1.995645743783]]), atol=atol)
    assert np.allclose(activations[3], np.array([[0.880339150445242]]), atol=atol)

def test_predict():
    X = np.array([[1, 2]])
    nn = NeuralNetwork([2, 2, 2, 1], "sigmoid")

    nn.weights = [np.array([[1, 2], [3, 4]]) * 0.1, np.array([[5, 6], [7, 8]]) * 0.1, np.array([[9, 10]]).T * 0.1]
    nn.biases = [np.array([[1, 2]])* 0.1, np.array([[3, 4]]) * 0.1, np.array([[5]]) * 0.1]
    wb = nn.wb()
    

    y_pred = nn.predict(X, wb)
    assert y_pred.shape == (1, 1)
    assert np.allclose(y_pred, np.array([[0.88033915]]), atol=1e-5)

    X = np.array([[1, 2], [3, 4]])
    y_pred = nn.predict(X, wb)
    assert y_pred.shape == (2, 1)

    X = np.array([8, 3, 2, 1, 4, 5, 6, 7]).reshape(4, 2)
    y_pred = nn.predict(X, wb)
    assert y_pred.shape == (4, 1)

def test_gradient_3_layers():
    """Numericallly estimates the gradient, and compares it to the gradient calculated by backpropagation
    """
    np.random.seed(90) # Set numpy rng seed

    layers = [2, 2, 1]
    nn = NeuralNetwork(layers, "sigmoid")
    wb = nn.wb()
    n = len(wb) # 6 in this case

    X = np.array([[1, 2], [3, 4]])
    y = np.array([[1], [0]])

    tol = 1e-4
    eps = 1e-4
    grad = nn.gradient(X, wb, y)
    for i in range(n):
        one_i = np.zeros((n, 1))
        one_i[i] = 1
        wb_plus = wb + one_i * eps
        wb_minus = wb - one_i * eps

        partial_i = (nn.cost(X, wb_plus, y) - nn.cost(X, wb_minus, y)) / (2 * eps)

        # Verify that relative error is lower than tolerance
        denom = partial_i if partial_i != 0 else 1
        assert abs((partial_i - grad[i][0]) / denom) < tol


def test_gradient_4_layers():
    """Numericallly estimates the gradient, and compares it to the gradient calculated by backpropagation
    """
    np.random.seed(673) # Set numpy rng seed

    layers = [3, 4, 3, 1]
    nn = NeuralNetwork(layers, "sigmoid")
    wb = nn.wb()
    n = len(wb)

    X = np.array([
        [1, 2, 3], 
        [3, 4, 5], 
        [10, 10, 10]
        ])
    y = np.array([
        [1], 
        [0],
        [0]
        ])

    tol = 1e-4
    eps = 1e-4
    grad = nn.gradient(X, wb, y)
    for i in range(n):
        one_i = np.zeros((n, 1))
        one_i[i] = 1
        wb_plus = wb + one_i * eps
        wb_minus = wb - one_i * eps

        partial_i = (nn.cost(X, wb_plus, y) - nn.cost(X, wb_minus, y)) / (2 * eps)

        # Verify that relative error is lower than tolerance
        denom = partial_i if partial_i != 0 else 1
        assert abs((partial_i - grad[i][0]) / denom) < tol 
