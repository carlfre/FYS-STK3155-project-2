import gradient_descent as gd
import regression_cost_funcs as rcf
import numpy as np

# Create data



np.random.seed(0)

N = 50
X = np.random.randn(N, 2)
y = 30 * X[:, 0] + 420 * X[:, 1]
w = np.zeros((2)).reshape(-1, 1)

descent = gd.GradientDescent(mode="adam", store_extra=True)
w = descent.train(X, w, y, rcf.OLSCost(), 0.1, 10_000)
w_evo = descent.weights

# Plot the evolution of the weights

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(w_evo[:, 0], "--o", label="w0")
ax[1].plot(w_evo[:, 1], "--o", label="w1")
ax[0].legend()
ax[0].set_title("Weight evolution")
plt.show()


