from sklearn.preprocessing import StandardScaler
import gradient_descent as gd
import regression_cost_funcs as rcf
import regression_tools as rt
import numpy as np
import matplotlib.pyplot as plt

 # Create data
N = 100
X = np.random.randn(N, 2)
y = 2 * X[:, 0] + 3 * X[:, 1]



# Initialize weights
w = np.zeros((2)).reshape(-1, 1)

descent = gd.GradientDescent(momentum_param=0, batch_size=None, mode="normal", store_extra=False)
w = descent.train(X, w, y, rcf.LogisticCost(), 0.01, 1000)
logmod = rcf.LogisticCost()

y_pred = logmod.predict(X, w)
print(f"Weights: {w}")
print(f"Accuracy: {rt.MSE(y, y_pred)}")

y = (y > 0).astype(int)
y_pred = logmod.predict_class(X, w)

print(f"Accuracy_class: {np.mean(y == y_pred)}")