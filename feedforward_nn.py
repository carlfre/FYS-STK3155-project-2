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

        self.input_activation_fn = self._linear
        self.input_activation_fn_derivative = self._linear_derivative
        self.final_activation_fn = self._sigmoid
        self.final_activation_fn_derivative = self._sigmoid_derivative
        # TODO: set final activation function and derivative
    
    def _initialize_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            weights.append(np.random.normal(0, 1, (self.layers[i], self.layers[i+1])))
        return weights

    def _initialize_biases(self):
        biases = []
        for i in range(len(self.layers) - 1):
            #TODO: make zero?
            biases.append(0*np.random.normal(0, 1, (1, self.layers[i+1])))
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
            2 / y.size 
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
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

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

    def cost(self, X, wb, y):
        weights, biases = self.unflatten_weights_and_biases(wb)
        y = y.reshape(-1, 1)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        activations, _ = self._forward_propagation()
        return np.mean((activations[-1] - y) ** 2)

    def gradient(self, X, wb, y):
        weights, biases = self.unflatten_weights_and_biases(wb)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        dW, db = self._back_propagation()
        grad = np.concatenate([dw.flatten() for dw in dW] + [db_.flatten() for db_ in db])
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
    pass

from data_generation import generate_data_linear

x, y, z, _ = generate_data_linear(3, 0, 22)
X = np.vstack([x, y]).T

nn = NeuralNetwork([2, 2, 2, 1], "relu")

from data_generation import generate_data_binary
x, y, z = generate_data_binary(100, 75767)
X = np.vstack([x, y]).T

from gradient_descent import GradientDescent
wb = nn.wb()

preds = nn.predict_binary(X, wb)

cnt = 0
for i in range(len(preds)):
    if preds[i] == z[i]:
        cnt += 1
print(cnt / len(preds))

print(np.count_nonzero(preds))

#print(nn.predict(X, wb))
gd = GradientDescent(batch_size=5, store_extra=True)
wb = gd.train(X, wb, y, nn, 0.01, 1000)

# Plot the cost function
import matplotlib.pyplot as plt
plt.plot(gd.costs)
plt.show()

#%%
preds = nn.predict_binary(X, wb)

cnt = 0
for i in range(len(preds)):
    if preds[i] == z[i]:
        cnt += 1
print(cnt / len(preds))

print(np.count_nonzero(preds))



#%%
# Test prediction
X = np.array([[0, 1],
              [1, 0],
              [1, 1],
              [0, 0]])

print(nn.predict(X, wb))




#%% 
#plot x, y, z
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()





def main():
    nn_example()

if __name__ == "__main__":
    main()