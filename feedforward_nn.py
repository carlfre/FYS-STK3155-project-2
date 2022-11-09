from regression_cost_funcs import ModelCost

import numpy as np

class NeuralNetwork(ModelCost):

    def __init__(self, layers, activation="sigmoid", output_activation="sigmoid", regularization=0):
        """Initialize a neural network with the given layers and activation function.
        Parameters
        ----------
        layers: list
            List of integers, where each integer represents the number of nodes in a layer.
        activation: str
            Activation function to use for all layers except the input and final layers.
            Options are "sigmoid", "relu", "leaky_relu", and "linear".
        output_activation: str
            Activation function to use for the final layer.
            Options are "sigmoid", "relu", "leaky_relu", and "linear".
        regularization: float
            L2 regularization parameter.
        """
        self.layers = layers
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        # Activation functions
        activation_dict = {
            "sigmoid": self._sigmoid,
            "relu": self._relu,
            "leaky_relu": self._leaky_relu,
            "linear": self._linear
        }
        activation_derivative_dict = {
            "sigmoid": self._sigmoid_derivative,
            "relu": self._relu_derivative,
            "leaky_relu": self._leaky_relu_derivative,
            "linear": self._linear_derivative
        }
        # Set input layer activation function to be linear.
        self.input_activation_fn = activation_dict["linear"]
        self.input_activation_fn_derivative = activation_derivative_dict["linear"]

        # Set hidden layer activation function
        self.activation_fn = activation_dict[activation]
        self.activation_fn_derivative = activation_derivative_dict[activation]

        # Set final layer activation function
        self.final_activation_fn = activation_dict[output_activation]
        self.final_activation_fn_derivative = activation_derivative_dict[output_activation]

        self.regularization = regularization
    
    def _initialize_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            weights.append(np.random.normal(0, 1, (self.layers[i], self.layers[i+1])))
        return weights

    def _initialize_biases(self):
        biases = []
        for i in range(len(self.layers) - 1):
            #TODO: make zero?
            biases.append(np.zeros((1, self.layers[i+1])))
        return biases

    def _linear(self, z):
        return z

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _relu(self, z):
        return np.maximum(z, 0)

    def _leaky_relu(self, z):
        return np.where(z > 0, z, z * 0.01)

    def _linear_derivative(self, z):
        return np.ones(z.shape)

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))
    
    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _leaky_relu_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    def get_activation_fn(self, layer):
        if layer == 0:
            return self.input_activation_fn
        elif layer == len(self.layers) - 1:
            return self.final_activation_fn
        else:
            return self.activation_fn
        

    def _forward_propagation(self):
        """Perform forward propagation to compute the activations and
        activation inputs."""
        W = self.weights
        b = self.biases
        
        Z = [] # activation inputs
        A = [] # activation values

        z = self.X
        for layer in range(len(self.layers)):
            activation = self.get_activation_fn(layer)
            a = activation(z)
            
            A.append(a)
            Z.append(z)

            if layer != len(self.layers) - 1:
                z = a @ W[layer] + b[layer]    
        
        return A, Z


    def _back_propagation(self):
        """Perform back propagation to compute the cost gradient.
        Returns
        -------
        dW: list
            List of weight matrices.
        db: list
            List of bias vectors.
        """
        y = self.y.reshape(-1, 1)
        W = self.weights
        activations, inputs = self._forward_propagation()
        dW = [np.zeros(w.shape) for w in W]
        db = [np.zeros(b.shape) for b in self.biases]

        # initial delta (ie. dC/d(z_L))
        delta = (
            2 
            * (activations[-1] - y) 
            * self.final_activation_fn_derivative(inputs[-1])
            )

        for layer in range(len(self.layers) - 1, 0, -1):
            dW[layer - 1] = activations[layer - 1].T @ delta
            db[layer - 1] = np.sum(delta, axis=0, keepdims=True)
            delta = delta @ W[layer - 1].T * self.activation_fn_derivative(inputs[layer - 1])

        return dW, db
        
    def concatenated_weights_and_biases(self):
        """reshapes the weights and biases into a single column vector."""
        wb = np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        return wb.reshape(-1, 1)

    def wb(self):
        return self.concatenated_weights_and_biases()

    def unflatten_weights_and_biases(self, wb):
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
        wb = wb.flatten()
        weights = []
        biases = []
        start = 0
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i] * self.layers[i+1]
            weights.append(wb[start:end].reshape(self.layers[i], self.layers[i+1]))
            start = end
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i+1]
            biases.append(wb[start:end].reshape(1, self.layers[i+1]))
            start = end
        return weights, biases

    def n_params(self, X=None):
        """Returns the number of parameters in the network."""
        return self.concatenated_weights_and_biases().size

    def cost(self, X, wb, y):
        weights, biases = self.unflatten_weights_and_biases(wb)
        y = y.reshape(-1, 1)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        activations, _ = self._forward_propagation()
        cost = np.sum((activations[-1] - y) ** 2)

        # Add L2 regularization
        if self.regularization > 0:
            reg = self.regularization * np.sum(np.concatenate([w.flatten() ** 2 for w in weights]))
            cost += reg

        return cost

    def gradient(self, X, wb, y):
        weights, biases = self.unflatten_weights_and_biases(wb)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        dW, db = self._back_propagation()
        grad = np.concatenate([dw.flatten() for dw in dW] + [db_.flatten() for db_ in db])

        # Apply L2 regularization to weights (not biases)
        if self.regularization > 0:
            reg = self.regularization * 2 * np.concatenate([w.flatten() for w in weights])
            grad[:len(reg)] += reg

        return grad.reshape(-1, 1)

    def predict(self, X, wb):
        weights, biases = self.unflatten_weights_and_biases(wb)

        self.X = X
        self.weights = weights
        self.biases = biases

        activations, _ = self._forward_propagation()
        return activations[-1]

    def predict_binary(self, X, wb):
        return self.predict(X, wb) > 0.5

def nn_example():

    from data_generation import generate_data_binary
    X, z = generate_data_binary(500, 787)
    
    nn = NeuralNetwork([2, 4, 4, 4, 1], "relu", output_activation="sigmoid", regularization=0.000)

    from gradient_descent import GradientDescent
    wb = nn.wb()

    preds = nn.predict_binary(X, wb)

    cnt = 0

    for i in range(len(preds)):
        if preds[i] == z[i]:
            cnt += 1
    
    print(cnt / len(preds))


    gd = GradientDescent(batch_size=5, store_extra=True)
    wb = gd.train(X, wb, z, nn, 0.1, 100)

    ## Plot the cost function
    import matplotlib.pyplot as plt
    plt.plot(gd.costs)
    plt.show()


    # 3d plot of prediction

    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    Z = np.hstack([X, Y])

    preds = nn.predict(Z, wb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, preds)
    plt.show()

def main():
    nn_example()

if __name__ == "__main__":
    main()