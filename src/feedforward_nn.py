#%%
import numpy as np

class NeuralNetwork:

    def __init__(self, layers, activation_function="sigmoid"):
        """Initialize a neural network with the given layers and activation function.
        Parameters
        ----------
        layers: list
            List of integers, where each integer represents the number of nodes in a layer.
    
        """

        self.layers = layers
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        if activation_function == "sigmoid":
            self.activation_fn = self._sigmoid
            self.activation_fn_derivative = self._sigmoid_derivative
        elif activation_function == "relu":
            self.activation_fn = self._relu
            self.activation_fn_derivative = self._relu_derivative
        elif activation_function == "leaky_relu":
            self.activation_fn = self._leaky_relu
            self.activation_fn_derivative = self._leaky_relu_derivative
        else:
            raise ValueError("Activation function not recognized.")

        # TODO: set final activation function and derivative
    
    def _initialize_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            weights.append(np.random.normal(0, 1, (self.layers[i], self.layers[i+1])))
        return weights

    def _initialize_biases(self):
        biases = []
        for i in range(len(self.layers) - 1):
            biases.append(np.random.normal(0, 1, (1, self.layers[i+1])))
        return biases

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _relu(self, z):
        return np.maximum(z, 0)

    def _leaky_relu(self, z):
        return np.where(z > 0, z, z * 0.01)

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))
    
    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _leaky_relu_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    def _forward_propagation(self):
        W = self.weights
        b = self.biases
        activations = [self.X]
        for i in range(len(self.weights)):
            activation = self.activation_fn(activations[i] @ W[i] + b[i])
            activations.append(activation)
        
        return activations


    def _back_propagation(self):
        """Perform back propagation to compute the cost gradient.
        Returns
        -------
        dW: list
            List of weight matrices.
        db: list
            List of bias vectors.
        """
        y = self.y
        W = self.weights
        activations = self._forward_propagation()
        dW = [np.zeros(w.shape) for w in W]
        db = [np.zeros(b.shape) for b in self.biases]

        # TODO: implement back propagation
        raise NotImplementedError()

        return dW, db
        
    def flattened_weights_and_biases(self):
        """Flattens the weights and biases into a single vector."""
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

    def unflatten_weights_and_biases(self, Wb):
        """Unflattens the weights and biases from a single vector.
        Parameters
        ----------
        w_and_b: np.ndarray
            Vector of weights and biases.
        Returns
        -------
        weights: list
            List of weight matrices.
        biases: list
            List of bias vectors.
        """
        weights = []
        biases = []
        start = 0
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i] * self.layers[i+1]
            weights.append(Wb[start:end].reshape(self.layers[i], self.layers[i+1]))
            start = end
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i+1]
            biases.append(Wb[start:end].reshape(1, self.layers[i+1]))
            start = end
        return weights, biases

    def cost(self, X, Wb, y):
        weights, biases = self.unflatten_weights_and_biases(Wb)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        activations = self._forward_propagation()
        return np.mean((activations[-1] - y) ** 2)

    def gradient(self, X, Wb, y):
        weights, biases = self.unflatten_weights_and_biases(Wb)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        dW, db = self._back_propagation()
        return np.concatenate([dw.flatten() for dw in dW] + [db_.flatten() for db_ in db])

    def predict(self, X, Wb):
        weights, biases = self.unflatten_weights_and_biases(Wb)

        self.X = X
        self.weights = weights
        self.biases = biases

        activations = self._forward_propagation()
        return activations[-1]
    
def nn_example():
    from data_generation import generate_data_linear

    x, y, z, _ = generate_data_linear(3, 0, 22)
    X = np.vstack([x, y]).T

    nn = NeuralNetwork([2, 2, 2, 1], "relu")
    
    Wprev = nn.weights
    bprev = nn.biases

    Wb = nn.flattened_weights_and_biases()
    print("Initial cost: ", nn.cost(X, Wb, z))
    print("Initial gradient: ", nn.gradient(X, Wb, z))

def main():
    nn_example()

if __name__ == "__main__":
    main()