# FYS-STK3155 Project 2
In this project, we explore gradient descent methods. In particular, we use linear regression as a test case for benchmarking our gradient descent implementation. We also implement logistic regression and a feedforward neural network.

## Analysis
Our analysis can be found in the jupyter notebooks found in the notebooks-folder. It includes
* Benchmarking gradient descent using linear regression.
* Comparing the performance of linear regression and neural networks when approximating functions
* Comparing the performance of logistic regression and neural networks for classification tasks

## Usage
All tests can be run with
```BASH
pytest
```
The cost, gradient and prediction functions are wrapped into a single class. Eg. OLSCost implements cost, gradient, and predict methods for OLS, and NeuralNetwork implements the same methods for a FFNN. The gradient descent training takes one such object with these 3 methods as input.

More example code can be found in the individual scripts. Eg. feedforward_nn.py showcases an example of usage for the neural network.
