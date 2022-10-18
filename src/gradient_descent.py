#%%
import math

import numpy as np
from regression_cost_funcs import OLSCost, RidgeCost
from abc import ABC, abstractmethod


class GradientDescent:

    def __init__(
        self,
        batch_size = None,
        momentum_param = 0,
        adagrad = False,
        adam = False,
        rmsprop = False,
        store_extra=False
    ):  
        # Batch size of None corresponds to gradient descent
        self.batch_size = batch_size

        # Momentum discount factor
        self.momentum_param = momentum_param
        self.momentum = 0

        self.use_adagrad = adagrad
        self.use_adam = adam
        self.use_rmsprop = rmsprop
        self.store_extra = store_extra

        # TODO: make adjustable
        self.rms_param = 0.9


    def train(self, X, w, y, model, learning_rate, n_epochs):
        self.X = X
        self.w = w.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.model = model
        self.eta = learning_rate

        if self.batch_size is None:
            batch_size = len(y)
        else:
            batch_size = self.batch_size

        if self.store_extra:
            # Store cost function and weight values for each epoch
            self.costs = np.zeros(n_epochs + 1)
            self.weights = np.zeros((n_epochs + 1, len(w)))
            # Store cost and weight values before training
            self.costs[0] = model.cost(X, self.w, self.y)
            self.weights[0] = w.flatten()
        
        for epoch in range(1, n_epochs + 1):
            # Shuffle data
            if self.batch_size:
                idx = np.random.permutation(len(self.y))
                self.X = self.X[idx]
                self.y = self.y[idx]
            # For each batchm, update weights
            for batch in range(math.ceil(len(y) / self.batch_size)):
                self.X_batch = self.X[batch * batch_size: (batch + 1) * batch_size]
                self.y_batch = self.y[batch * batch_size: (batch + 1) * batch_size]
                self.w = self.update()
            # Store cost and weight values after each epoch
            if self.store_extra:
                    self.costs[epoch] = model.cost(X, self.w, y)
                    self.weights[epoch] = self.w.flatten()
        
        return self.w

    def delta_w(self, X, w, y, model, eta):
        return self.momentum * self.momentum_param - eta * model.gradient(X, w, y)

    def adagrad(self):
        if not hasattr(self, "G"):
            self.G = np.zeros(( len(self.w), 1))
        #print(self.G.shape, (self.model.gradient(self.X_batch, self.w, self.y_batch)**2).shape)
        #print((self.G + self.model.gradient(self.X_batch, self.w, self.y_batch)**2).shape)
        self.G += self.model.gradient(self.X_batch, self.w, self.y_batch)**2
        return self.eta / np.sqrt(self.G)
    
    def rmsprop(self):
        if not hasattr(self, "v"):
            self.v = np.zeros((len(self.w), 1))
        self.v = self.rms_param * self.v + (1 - self.rms_param) * self.model.gradient(self.X_batch, self.w, self.y_batch)**2
        return self.eta / np.sqrt(self.v)

    #def sampler(self):
    #    if not hasattr(self, "y") or not hasattr(self, "X"):
    #        raise ValueError("X and y have not been set.")
    #        
    #    if self.batch_size == None:
    #        return (self.X, self.y)
    #
    #    elif 0 < self.batch_size < 1:
    #        n_samples = self.batch_size
    #        idx = np.random.choice(len(self.y), n_samples, replace=False)
    #        return (self.X[idx], self.y[idx])
    #
    #    else:
    #        raise ValueError("Batch size must be in interval (0, 1]")
    
    def update(self):
        X = self.X_batch
        w = self.w
        y = self.y_batch
        model = self.model
        eta = self.eta

        if self.use_adagrad:
            eta_star = self.adagrad()
        elif self.use_rmsprop:
            eta_star = self.rmsprop()
        else:
            eta_star = eta

        delta_w = self.delta_w(X, w, y, model, eta_star)
        self.momentum = delta_w
        return w + delta_w
        # TODO: Handle adam
        



def gradient_descent(
    X, 
    y, 
    w, 
    model, 
    n_iter, 
    eta,
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
    eta: float
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
        w = w - eta * grad
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
    eta, 
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
    eta: float
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
        v = gamma * v + eta * grad
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
    eta, 
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
    eta: float
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
        w = w - eta * grad

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
    eta,
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
    eta: float
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
        v = gamma * v + eta * grad
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

def gradient_descent_class_example():
    import matplotlib.pyplot as plt
    N = 100
    X = np.random.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(N)

    y = y.reshape(-1, 1)

    # Initialize weights
    w = np.zeros((2)).reshape(-1, 1)

    gd = GradientDescent(momentum_param=0.5, batch_size=20, store_extra=True, rmsprop=True)
    w = gd.train(X, w, y, OLSCost(), 0.5, 5)
    print("True weights: [2, 3]")
    print(f"Final weights: {w}")
    print(f"Final cost: {gd.costs[-1]}")
    print(gd.costs)

    model = OLSCost()
    print(w)
    print(y.shape)
    print(model.cost(X, w, y))

    # Plot cost function
    plt.plot(gd.costs)
    plt.show()

    # Plot weight path
    plt.plot(gd.weights[:, 0], label="w0")
    plt.plot(gd.weights[:, 1], label="w1")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    gradient_descent_class_example()

