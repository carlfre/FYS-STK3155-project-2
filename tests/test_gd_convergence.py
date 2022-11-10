# Test function gradient descent using unit tests

import numpy as np
import pytest
import gradient_descent as gd
import regression_tools as rt
import regression_cost_funcs as rcf
from sklearn.linear_model import LogisticRegression
from data_generation import generate_data_binary

np.random.seed(2)

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
    w = descent.train(X, w, y, rcf.OLSCost(), 0.001, 1000)
    
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


@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("momentum_param", [0.5, 0.9])
@pytest.mark.parametrize("mode", ["normal", "adagrad", "rmsprop", "adam"])
def test_sdg_logistic_convergence(batch_size, mode, momentum_param):
    """Test gradient descent using different modes."""
    # Create data
    N = 1000
    seed = 6789
    #X = np.random.randn(N, 2)
    #y = 4.5 * X[:, 0] + 3.2 * X[:, 1]
    X, y = generate_data_binary(N, seed, add_col_ones=True)

    # Initialize weights
    w = np.zeros((3)).reshape(-1, 1)
    descent = gd.GradientDescent(momentum_param=momentum_param, batch_size=batch_size, mode=mode)
    w = descent.train(X, w, y, rcf.LogisticCost(), 0.03, 200)

    logmod = rcf.LogisticCost()

    y = (y > 0).astype(int)
    y_pred = logmod.predict_class(X, w)

    print(y)
    print(y_pred)
    assert((np.mean(y == y_pred)) == pytest.approx(1, 0.1)) # sklearn gives slightly above 90% accuracy

