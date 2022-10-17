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

        # TODO: implement rest of initialization
    
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

    def forward_propagation(self):
        pass

    def back_propagation(self):
        pass

    def cost(self, X, w, y):
        pass

    def gradient(self, X, w, y):
        pass

    def predict(self, X, w):
        pass
    
