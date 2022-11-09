#%%
from abc import ABC, abstractmethod

import numpy as np


class ModelCost(ABC):
    """Blueprint for cost function implementations.
    Includes a gradient method."""

    @abstractmethod
    def __init__(self):
        """Initialize the cost function. If needed, specify parameters"""
        pass

    @abstractmethod
    def cost(self, X, w, y):
        """Evaluate the cost function.
        Parameters
        ----------
        X: np.ndarray
            Design matrix
        y: np.ndarray
            Target vector (column vector)
        w: np.ndarray
            Weight vector (column vector)
        
        Returns
        -------
        cost: float
            Value of the cost function
        """
        pass

    @abstractmethod
    def gradient(self, X, w, y):
        """Evaluate the gradient of the cost function.
        
        Parameters
        ----------
        X: np.ndarray
            Design matrix
        y: np.ndarray
            Target vector (column vector)
        w: np.ndarray
            Weight vector (column vector)
        
        Returns
        -------
        gradient: np.ndarray
            Gradient of the cost function"""
        pass

    @abstractmethod
    def predict(self, X, w):
        """Predict target values from input data.
        Parameters
        ----------
        X: np.ndarray
            Design matrix
        w: np.ndarray
            Weight vector (column vector)
        
        Returns
        -------
        y: np.ndarray
            Predicted target values
        """
        pass

    def preprocess(self, w, y=None):
        """Preprocess the weight and target vectors by convering
        them to column vectors if they are not already.

        Parameters
        ----------
        w: np.ndarray
            Weight vector (column vector)
        y: np.ndarray
            Target vector (column vector)
        
        Returns
        -------
        w: np.ndarray
            Preprocessed weight vector (column vector)
        y: np.ndarray
            Preprocessed target vector (column vector)
        """
        if y is not None:
            return w.reshape(-1, 1), y.reshape(-1, 1)
        else:
            return w.reshape(-1, 1)

    def n_params(self, X):
        """Get the number of parameters in the model.
        Parameters
        ----------
        X: np.ndarray
            Design matrix
        
        Returns
        -------
        n_params: int
            Number of parameters in the model
        """
        return X.shape[1]

class OLSCost(ModelCost):
    """Ordinary least squares cost function."""

    def __init__(self):
        pass

    def cost(self, X, w, y):
        """Evaluate the cost function."""
        w, y = self.preprocess(w, y)
        return np.mean((X @ w - y) ** 2)

    def gradient(self, X, w, y):
        """Evaluate the gradient of the cost function."""
        w, y = self.preprocess(w, y)
        return 2 / y.size * X.T @ (X @ w - y) 

    def predict(self, X, w):
        """Predict target values from input data."""
        w = self.preprocess(w)
        return X @ w

class RidgeCost(ModelCost):
    """Ridge regression cost function."""

    def __init__(self, regularization):
        self.regularization = regularization

    def cost(self, X, w, y):
        """Evaluate the cost function."""
        w, y = self.preprocess(w, y)
        return np.mean((X @ w - y) ** 2) + self.regularization * np.sum(w ** 2)

    def gradient(self, X, w, y):
        """Evaluate the gradient of the cost function."""
        w, y = self.preprocess(w, y)
        return 2 / y.size * X.T @ (X @ w - y) + 2 * self.regularization * w
    
    def predict(self, X, w):
        """Predict target values from input data."""
        w = self.preprocess(w)
        return X @ w


class LogisticCost(ModelCost):
    """Logistic regression cost function."""
    
    def __init__(self, regularization=0):
        """Initialize, set L2 regularization parameter."""
        self.regularization = regularization
    
    def cost(self, X, w, y):
        """Evaluate cross entropy cost."""
        w, y = self.preprocess(w, y)
        p = self.predict(X, w)
        return - np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + self.regularization * np.sum(w[1:] ** 2)
    
    def gradient(self, X, w, y):
        """Evaluate the gradient of the cost function."""
        w, y = self.preprocess(w, y)
        p = self.predict(X, w)
        reg = 2 * self.regularization * w
        reg[0] = 0
        return - X.T @ (y - p) + reg
        
    def predict(self, X, w):
        """Compute the probability of a positive class."""
        w = self.preprocess(w)
        return 1 / (1 + np.exp(-X @ w))
    
    def predict_class(self, X, w):
        """Predict target values from input data."""
        w = self.preprocess(w)
        return (1 / (1 + np.exp(-X @ w)) > 0.5).astype(int)

def cost_ols_example():
    """Example of how to use the cost function wrapper."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    
    # Create some data
    X = np.random.normal(size=(100, 2))
    y = 1*X[:, 0] + 1 *X[:, 1] #+ np.random.normal(size=100)
    y = y.reshape(-1, 1)

    
    # Create a grid of beta values
    beta_0 = np.linspace(-2, 2, 100)
    beta_1 = np.linspace(-2, 2, 100)
    beta_0, beta_1 = np.meshgrid(beta_0, beta_1)

    # Initialize cost function wrapper
    model = OLSCost()

    
    # Compute the cost function for each beta value
    cost = np.zeros(beta_0.shape)
    for i in range(beta_0.shape[0]):
        for j in range(beta_0.shape[1]):
            beta = np.array([beta_0[i, j], beta_1[i, j]]).reshape(-1, 1)
            cost[i, j] = model.cost(X, beta, y)

    
    # Plot the cost function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        beta_0, beta_1, cost, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel(r"$\beta_1$")
    ax.set_zlabel("Cost")
    ax.set_title("OLS cost function")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    
    # Plot the gradient
    model = OLSCost()
    beta = np.array([2, 1]).reshape(-1, 1)
    grad = model.gradient(X, beta, y)

    beta_0 = np.linspace(-2, 2, 20)
    beta_1 = np.linspace(-2, 2, 20)
    beta_0, beta_1 = np.meshgrid(beta_0, beta_1)

    # evaluate gradient at each point (beta0, beta1)
    grad_0 = np.zeros(beta_0.shape)
    grad_1 = np.zeros(beta_1.shape)
    for i in range(beta_0.shape[0]):
        for j in range(beta_0.shape[1]):
            beta = np.array([beta_0[i, j], beta_1[i, j]])
            grad = model.gradient(X, beta, y)
            grad_0[i, j] = grad[0]
            grad_1[i, j] = grad[1]


    plt.quiver(beta_0, beta_1, grad_0, grad_1)
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel(r"$\beta_1$")
    ax.set_zlabel("Gradient")
    ax.set_title("OLS gradient")
    plt.show()

def main():
    cost_ols_example()

if __name__ == "__main__":
    main()
