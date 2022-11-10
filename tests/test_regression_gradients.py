import pytest
import numpy as np
from regression_cost_funcs import OLSCost, RidgeCost, LogisticCost
from data_generation import generate_data_binary

@pytest.mark.parametrize("w", 
[
    np.array([1, 2]), 
    np.array([-3, 5]),
    np.array([0, 0]),
    np.array([-3.4, 5.6]),
])
def test_ols_gradient(w):
    """Numerically verify gradient of OLS cost function."""
    np.random.seed(67)
    ols = OLSCost()
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1]

    y = y.reshape(-1, 1)

    # Compute gradient
    grad = ols.gradient(X, w, y)

    # Compute numerical gradient
    eps = 1e-4
    
    dw = np.array([eps, 0])
    dCdw1 = (ols.cost(X, w + dw, y) - ols.cost(X, w - dw, y)) / (2 * eps)
    dw = np.array([0, eps])
    dCdw2 = (ols.cost(X, w + dw, y) - ols.cost(X, w - dw, y)) / (2 * eps)
    
    numerical_grad = np.array([dCdw1, dCdw2]).reshape(-1, 1)

    assert np.allclose(grad, numerical_grad, rtol=1e-3, atol=1e-3)
    
@pytest.mark.parametrize("w", 
[
    np.array([1, 2]), 
    np.array([-3, 5]),
    np.array([0, 0]),
    np.array([-3.4, 5.6]),
])
@pytest.mark.parametrize("regularization", [0, 0.001, 0.1, 0.5])
def test_ridge_gradient(w, regularization):
    """Numerically verify gradient of OLS cost function."""
    np.random.seed(67)

    
    ridge = RidgeCost(regularization)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1]

    y = y.reshape(-1, 1)

    # Compute gradient
    grad = ridge.gradient(X, w, y)

    # Compute numerical gradient
    eps = 1e-4
    
    dw = np.array([eps, 0])
    dCdw1 = (ridge.cost(X, w + dw, y) - ridge.cost(X, w - dw, y)) / (2 * eps)
    dw = np.array([0, eps])
    dCdw2 = (ridge.cost(X, w + dw, y) - ridge.cost(X, w - dw, y)) / (2 * eps)
    
    numerical_grad = np.array([dCdw1, dCdw2]).reshape(-1, 1)

    assert np.allclose(grad, numerical_grad, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("w", 
[
    np.array([1, 2]), 
    np.array([-3, 5]),
    np.array([0, 0]),
    np.array([-3.4, 5.6]),
])
@pytest.mark.parametrize("regularization", [0, 0.001, 0.1, 0.5])
def test_logistic_gradient(w, regularization):
    """Numerically verify gradient of OLS cost function."""
    seed = 89

    
    logistic = LogisticCost(regularization)
    X, y = generate_data_binary(100, seed)

    # Compute gradient
    grad = logistic.gradient(X, w, y)

    # Compute numerical gradient
    eps = 1e-4
    
    dw = np.array([eps, 0])
    dCdw1 = (logistic.cost(X, w + dw, y) - logistic.cost(X, w - dw, y)) / (2 * eps)
    dw = np.array([0, eps])
    dCdw2 = (logistic.cost(X, w + dw, y) - logistic.cost(X, w - dw, y)) / (2 * eps)
    
    numerical_grad = np.array([dCdw1, dCdw2]).reshape(-1, 1)

    print(grad)
    print(numerical_grad)
    assert np.allclose(grad, numerical_grad, rtol=1e-3, atol=1e-3)