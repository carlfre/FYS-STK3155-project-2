# Test function gradient descent using unit tests

import numpy as np
import pytest
import gradient_descent as gd
import regression_tools as rt
import regression_cost_funcs as rcf
from sklearn.linear_model import LogisticRegression

np.random.seed(1)

def test_gd_ols_convergence():
    """Test gradient descent using different modes."""
    # Create data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1]
    true_weights = rt.ols_regression(X, y)

    y = y.reshape(-1, 1)

    # Initialize weights
    w = np.zeros((2)).reshape(-1, 1)

    
    descent = gd.GradientDescent()
    w = descent.train(X, w, y, rcf.OLSCost(), 0.01, 1000)
    assert(w.flatten() == pytest.approx(true_weights, 0.5))


@pytest.mark.parametrize("batch_size", [20])
@pytest.mark.parametrize("momentum_param", [0.5, 0.9])
@pytest.mark.parametrize("mode", ["normal", "adagrad", "rmsprop", "adam"])
def test_sdg_ols_convergence(batch_size, mode, momentum_param):
    """Test gradient descent using different modes."""
    # Create data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1]
    true_weights = rt.ols_regression(X, y)

    y = y.reshape(-1, 1)

    # Initialize weights
    w = np.zeros((2)).reshape(-1, 1)

    descent = gd.GradientDescent(momentum_param=momentum_param, batch_size=batch_size, mode=mode)
    w = descent.train(X, w, y, rcf.OLSCost(), 0.01, 1000)
    assert(w.flatten() == pytest.approx(true_weights, 0.5))


@pytest.mark.parametrize("batch_size", [20])
@pytest.mark.parametrize("momentum_param", [0.5, 0.9])
@pytest.mark.parametrize("mode", ["normal", "adagrad", "rmsprop", "adam"])
@pytest.mark.parametrize("lambd", [0.00001, 0.01])
def test_sdg_ridge_convergence(batch_size, mode, momentum_param, lambd):
    """Test gradient descent using different modes."""
    # Create data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1]
    true_weights = rt.ridge_regression(X, y, lambd)

    y = y.reshape(-1, 1)

    # Initialize weights
    w = np.zeros((2)).reshape(-1, 1)

    descent = gd.GradientDescent(momentum_param=momentum_param, batch_size=batch_size, mode=mode)
    w = descent.train(X, w, y, rcf.RidgeCost(lambd), 0.01, 1000)
    assert(w.flatten() == pytest.approx(true_weights, 0.5))


@pytest.mark.parametrize("batch_size", [None, 10])
@pytest.mark.parametrize("momentum_param", [0.9])
@pytest.mark.parametrize("mode", ["normal", "adagrad", "rmsprop", "adam"])
def test_gradient_descent_logistic(batch_size, mode, momentum_param):
    """Test gradient descent using different modes."""
    # Create data
    N = 100
    X = np.random.randn(N, 2)
    y = 405 * X[:, 0] + 32 * X[:, 1]

    # Initialize weights
    w = np.zeros((2)).reshape(-1, 1)
    descent = gd.GradientDescent(momentum_param=momentum_param, batch_size=batch_size, mode=mode)
    w = descent.train(X, w, y, rcf.LogisticCost(), 0.01, 1000)

    logmod = rcf.LogisticCost()

    y = (y > 0).astype(int)
    y_pred = logmod.predict_class(X, w)

    assert((np.mean(y == y_pred)) == pytest.approx(1, 0.25))

