#%%
import numpy as np
from regression_cost_funcs import OLSCost, RidgeCost

def gradient_descent(
    X, 
    y, 
    w, 
    model, 
    n_iter, 
    learning_rate,
    compute_extra=True,
):
    """Gradient descent algorithm.

    Parameters
    ----------
    X: np.ndarray
        Design matrix
    y: np.ndarray
        Response vector
    w: np.ndarray
        Initial guess for weights
    model: ModelCost
        Cost function with gradient method
    n_iter: int
        Number of iterations
    learning_rate: float
        Learning rate
    compute_extra: bool
        Whether to return cost function values and weight path
    
    Returns
    -------
    gamma: np.ndarray
        Estimated gamma
    cost: np.ndarray (if compute_extra=True)
        Cost function values after each iteration
    W: np.ndarray (if compute_extra=True)
        Weight path
    """
    if compute_extra:
        cost = np.zeros((n_iter + 1))
        cost[0] = model.cost(X, w, y)
        W = np.zeros((n_iter + 1, w.shape[0]))
        W[0] = w.flatten()
    
    for i in range(1, n_iter + 1):
        # Compute the gradient
        grad = model.gradient(X, w, y)
        # Update w
        w = w - learning_rate * grad
        # Compute the cost function if specified
        if compute_extra:
            cost[i] = model.cost(X, w, y)
            W[i] = w.flatten()

    if compute_extra:
        return w, cost, W
    else:
        return w


def momentum_gradient_descent(
    X, 
    y, 
    w, 
    model, 
    n_iter, 
    learning_rate, 
    gamma=0.9,
    compute_extra=False,
):
    """Gradient descent algorithm with momentum.

    Parameters
    ----------
    X: np.ndarray
        Design matrix
    y: np.ndarray
        Response vector
    w: np.ndarray
        Initial guess for weights
    model: ModelCost
        Cost function with gradient method
    n_iter: int
        Number of iterations
    learning_rate: float
        Learning rate
    gamma: float
        Momentum parameter
    compute_extra: bool
        Whether to return cost function values and weight path
    
    Returns
    -------
    w: np.ndarray
        Estimated weights
    cost: np.ndarray (if compute_extra=True)
        Cost function values after each iteration
    W: np.ndarray (if compute_extra=True)
        Weight path
    """
    if compute_extra:
        cost = np.zeros(n_iter + 1)
        cost[0] = model.cost(X, w, y)
        W = np.zeros((n_iter + 1, w.shape[0]))
        W[0] = w.flatten()
    
    v = np.zeros_like(w)
    for i in range(1, n_iter + 1):
        # Compute the gradient
        grad = model.gradient(X, w, y)
        # Update v
        v = gamma * v + learning_rate * grad
        # Update w
        w = w - v
        # Compute the cost function if specified
        if compute_extra:
            cost[i] = model.cost(X, w, y)
            W[i] = w.flatten()

    if compute_extra:
        return w, cost, W
    else:
        return w


def stochastic_gradient_descent(
    X, 
    y, 
    w, 
    model, 
    n_iter, 
    batch_size, 
    learning_rate, 
    compute_extra=False,
):
    """Stochastic gradient descent algorithm.

    Parameters
    ----------
    X: np.ndarray
        Design matrix
    y: np.ndarray
        Response vector
    w: np.ndarray
        Initial guess for weights
    model: ModelCost
        Cost function with gradient method
    n_iter: int
        Number of iterations
    batch_size: int
        Number of samples to use in each iteration
    learning_rate: float
        Learning rate
    compute_extra: bool
        Whether to return cost function values and weight path

    Returns
    -------
    gamma: np.ndarray
        Estimated gamma
    cost: np.ndarray (if compute_extra=True)
        Cost function values after each iteration
    W: np.ndarray (if compute_extra=True)
        Weight path
    """
    if compute_extra:
        cost = np.zeros((n_iter + 1))
        cost[0] = model.cost(X, w, y)
        W = np.zeros((n_iter + 1, w.shape[0]))
        W[0] = w.flatten()
    
    for i in range(1, n_iter + 1):
        # Randomly sample batch_size indices from [0, N-1]
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        # Compute the gradient
        grad = model.gradient(X[indices], w, y[indices])
        # Update w
        w = w - learning_rate * grad

        if compute_extra:
            cost[i] = model.cost(X, w, y)
            W[i] = w.flatten()
    
    if compute_extra:
        return w, cost, W
    else:
        return w

def stochastic_gradient_descent_with_momentum(
    X,
    y,
    w,
    model,
    n_iter,
    batch_size,
    learning_rate,
    gamma=0.9,
    compute_extra=False,
):
    """Stochastic gradient descent algorithm with momentum.

    Parameters
    ----------
    X: np.ndarray
        Design matrix
    y: np.ndarray
        Response vector
    w: np.ndarray
        Initial guess for weights
    model: ModelCost
        Cost function with gradient method
    n_iter: int
        Number of iterations
    batch_size: int
        Number of samples to use in each iteration
    learning_rate: float
        Learning rate
    gamma: float
        Momentum parameter
    compute_extra: bool
        Whether to return cost function values and weight path

    Returns
    -------
    w: np.ndarray
        Estimated weights
    cost: np.ndarray (if compute_extra=True)
        Cost function values after each iteration
    W: np.ndarray (if compute_extra=True)   
        Weight path
    """
    if compute_extra:
        cost = np.zeros((n_iter + 1))
        cost[0] = model.cost(X, w, y)
        W = np.zeros((n_iter + 1, w.shape[0]))
        W[0] = w.flatten()
    
    v = np.zeros_like(w)
    for i in range(1, n_iter + 1):
        # Randomly sample batch_size indices from [0, N-1]
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        # Compute the gradient
        grad = model.gradient(X[indices], w, y[indices])
        # Update v
        v = gamma * v + learning_rate * grad
        # Update w
        w = w - v

        if compute_extra:
            cost[i] = model.cost(X, w, y)
            W[i] = w.flatten()
    
    if compute_extra:
        return w, cost, W
    else:
        return w


def gradient_descent_example():
    import matplotlib.pyplot as plt

    # Generate data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(N)

    # Initialize weights
    w = np.zeros(2)

    # Run gradient descent
    w, cost, W = gradient_descent(X, y, w, OLSCost(), 200, 0.01, compute_extra=True)
    print("True weights: [2, 3]")
    print(f"Final weights: {w}")
    print(f"Final cost: {cost[-1]}")

    # Plot cost function
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Plot weight path
    plt.plot(W[:, 0], label="w0")
    plt.plot(W[:, 1], label="w1")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()

def stochastic_gradient_descent_example():
    import matplotlib.pyplot as plt

    # Generate data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(N)

    # Initialize weights
    w = np.zeros(2)

    # Run gradient descent
    w, cost, W = stochastic_gradient_descent(X, y, w, OLSCost(), 200, 10, 0.01, compute_extra=True)
    print("True weights: [2, 3]")
    print(f"Final weights: {w}")
    print(f"Final cost: {cost[-1]}")

    # Plot cost function
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Plot weight path
    plt.plot(W[:, 0], label="w0")
    plt.plot(W[:, 1], label="w1")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()

def momentum_gradient_descent_example():
    import matplotlib.pyplot as plt

    # Generate data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(N)

    # Initialize weights
    w = np.zeros(2)

    # Run gradient descent
    w, cost, W = momentum_gradient_descent(X, y, w, OLSCost(), 200, 0.01, compute_extra=True)
    print("True weights: [2, 3]")
    print(f"Final weights: {w}")
    print(f"Final cost: {cost[-1]}")

    # Plot cost function
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Plot weight path
    plt.plot(W[:, 0], label="w0")
    plt.plot(W[:, 1], label="w1")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()

def stochastic_gradient_descent_with_momentum_example():
    import matplotlib.pyplot as plt

    # Generate data
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(N)

    # Initialize weights
    w = np.zeros(2)

    # Run gradient descent
    w, cost, W = stochastic_gradient_descent_with_momentum(X, y, w, OLSCost(), 200, 10, 0.01, compute_extra=True)
    print("True weights: [2, 3]")
    print(f"Final weights: {w}")
    print(f"Final cost: {cost[-1]}")

    # Plot cost function
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Plot weight path
    plt.plot(W[:, 0], label="w0")
    plt.plot(W[:, 1], label="w1")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    stochastic_gradient_descent_with_momentum_example()
