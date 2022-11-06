import numpy as np
import matplotlib.pyplot as plt
import regression_tools as rt
import regression_cost_funcs as rcf
import gradient_descent as gd
import time

# Create data
def Franke_function(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_data(N, sigma=0.01):
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    z = Franke_function(x, y)
    if sigma is not None:
        z += np.random.normal(0, sigma, z.shape)
    return x, y, z

x, y, z = create_data(1000, 0.1)
polydeg = 2
X = rt.create_X_polynomial(x, y, polydeg)
y = z.reshape(-1, 1)


def test_gradient_descent(X, z, momentum, batch_size, mode, cost_func, lambd=None):
    descent = gd.GradientDescent(momentum_param=momentum, batch_size=batch_size, mode=mode)
    w = np.zeros((X.shape[1])).reshape(-1, 1)
    # Record time to train
    start = time.time()
    w = descent.train(X, w, z, cost_func, 0.01, 10_000)
    end = time.time()
    return rt.MSE(y, X @ w), end - start

def test_OLS_gradient_descent(X, z, momentum, batch_size, mode):
    # Test different parameters
    mom = [0, 0.5, 0.9]
    batch = [None, 10, 100]
    mode = ["normal", "rmsprop", "adagrad", "adam"]

    # Print results

    print(f"OLSCost for degree {polydeg}, sigma=0.1, N=1000, iterations=10_000, Franke function")
    print("Mode, momentum, batch size, MSE, time")
    for m in mode:
        for mo in mom:
            for b in batch:
                mse, t = test_gradient_descent(X, y, mo, b, m, rcf.OLSCost())
                print(f"{m}, {mo}, {b}, {mse}, {t}")
            




def main():
    pass

if __name__ == "__main__":
    main()

    