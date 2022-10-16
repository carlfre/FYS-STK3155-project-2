#%%
import numpy as np

class CostFunction:
    """Wrapper for storing the regularization parameter."""

    def __init__(self, function_type="ols", regularization=None):
        """Initialize the cost function.

        Parameters
        ----------
        function_type: str
            Type of cost function. Either "ols" or "ridge".
        regularization: float
            Regularization parameter.
        """
        self.function_type = function_type
        self.regularization = regularization

    def __call__(self, y, y_pred, beta=None):
        """Evaluate the cost function.

        Parameters
        ----------
        y: np.ndarray
            True values.
        y_pred: np.ndarray
            Predicted values.
        beta: np.ndarray
            Regression coefficients.

        Returns
        -------
        cost: float
            Cost value.
        """
        if self.function_type == "ols":
            cost = np.mean((y - y_pred) ** 2)
        elif self.function_type == "ridge":
            cost = np.mean((y - y_pred) ** 2) + self.regularization * np.mean(beta ** 2)
        else:
            raise ValueError("Unknown cost function type.")
        return cost


class Gradient:
    """Wrapper for certain gradients."""

    def __init__(self, function_type="ols", regularization=None):
        if function_type == "ols":
            self.function = grad_ols
        elif function_type == "ridge":
            self.function = grad_ridge
        else:
            raise ValueError("function_type must be 'ols' or 'ridge'")
        
        self.regularization = regularization

    def __call__(self, X, y, beta):
        if len(beta.shape) == 1:
            beta = beta.reshape(-1, 1)

        if self.regularization == None:
            return self.function(X, y, beta)
        else:
            return self.function(X, y, beta, self.regularization)
        
def cost_ols(X, y, beta):
    """Compute the OLS cost function."""
    return np.mean((X @ beta - y) ** 2)


def cost_ridge(X, y, beta, lmbda):
    """Compute the ridge regression cost function."""
    return np.mean((X @ beta - y) ** 2) + lmbda * np.mean(beta ** 2)


def grad_ols(X, beta, y):
    """Compute the gradient of the OLS cost function."""
    return 2 * X.T @ ( X @ beta - y)


def grad_ridge(X, beta, y, lmbda):
    """Compute the gradient of the ridge regression cost function."""
    return 2 * X.T @ ( X @ beta - y) + 2 * lmbda * beta


def gradient_descent(
    X, 
    y, 
    w, 
    diff, 
    cost_fn, 
    n_iter, 
    learning_rate, 
    lmbd=0
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
    diff: function
        Function that computes the gradient of the cost function
    cost_fn: function
        Function that computes loss
    n_iter: int
        Number of iterations
    learning_rate: float
        Learning rate

    Returns
    -------
    beta: np.ndarray
        Estimated beta
    cost: np.ndarray
        Cost function values after each iteration
    """
    cost = np.zeros(n_iter)
    for i in range(n_iter):
        # Compute the gradient
        grad = diff(X, w, y)
        # Update beta
        w = w - learning_rate * grad
        # Compute the cost function
        cost[i] = cost_fn(X, y, w)
    return w, cost


def stochastic_gradient_descent(
    X, 
    y, 
    W, 
    diff, 
    cost_fn, 
    n_iter, 
    batch_size, 
    learning_rate, 
    lmbd=0
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
    diff: function
        Function that computes the gradient of the cost function
    cost_fn: function
        Function that computes loss
    n_iter: int
        Number of iterations
    batch_size: int
        Number of samples to use in each iteration
    learning_rate: float
        Learning rate

    Returns
    -------
    beta: np.ndarray
        Estimated beta
    cost: np.ndarray
        Cost function values after each iteration
    """
    cost = np.zeros(n_iter)
    for i in range(n_iter):
        # Randomly sample batch_size indices from [0, N-1]
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        # Compute the gradient
        grad = diff(X[indices], w, y[indices])
        # Update beta
        w = w - learning_rate * grad
        # Compute the cost function
        cost[i] = cost_fn(X, y, w)
    return w, cost



#  NOTE: Experimental / example code below
#
#


def cost_ols_example():
    """Example of how to use the cost function wrapper."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    
    # Create some data
    X = np.random.normal(size=(100, 2))
    y = X[:, 0] + X[:, 1] + np.random.normal(size=100)

    
    # Create a grid of beta values
    beta_0 = np.linspace(-2, 2, 100)
    beta_1 = np.linspace(-2, 2, 100)
    beta_0, beta_1 = np.meshgrid(beta_0, beta_1)

    # Initialize cost function wrapper
    cost_fn = CostFunction(function_type="ols")

    
    # Compute the cost function for each beta value
    cost = np.zeros(beta_0.shape)
    for i in range(beta_0.shape[0]):
        for j in range(beta_0.shape[1]):
            beta = np.array([beta_0[i, j], beta_1[i, j]])
            cost[i, j] = cost_fn(X @ beta, y)

    
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

    """
    # Plot the gradient
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    grad = grad_ols(X, np.array([beta_0, beta_1]), y)
    surf = ax.plot_surface(
        beta_0,
        beta_1,
        grad[0],
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.5,
    )
    """


def gradient_descent_example():
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load the data
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add a column of ones to the design matrix
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # Initialize beta
    beta = np.zeros(X_train.shape[1])

    # initialize gradient and cost function
    grad_fn = Gradient(function_type="ols")
    cost_fn = CostFunction(function_type="ols")

    # Run gradient descent
    beta, cost = gradient_descent(
        X_train, y_train, beta, grad_fn, cost_fn, 1000, 0.1
    )

    # Plot the cost function
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Compute the predictions
    y_pred = X_test @ beta

    # Compute the mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"MSE: {mse:.3f}")

if __name__ == "__main__":
    cost_ols_example()

# %%
