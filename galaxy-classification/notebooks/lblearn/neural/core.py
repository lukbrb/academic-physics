import numpy as np


class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None 

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass 


class Activation(Layer):
    def __init__(self, activation, activation_derivative) -> None:
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative
                           (self.input))


class Tanh(Activation):
    def __init__(self) -> None:
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(np.tanh, tanh_prime)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)
