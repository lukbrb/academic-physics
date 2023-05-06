import numpy as np
from neural import Dense, Tanh, mse, mse_prime
from torch import nn
import torch
# Test on the XOR problem
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 2), 
    Tanh(),
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh(),
]

epochs = 1
lr = 1e-2

def training(epochs, network):
    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            output = x 
            for layer in network:
                output = layer.forward(output)
            error += mse(y, output)

            grad = mse_prime(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad, lr)
        
    error /= len(X)
    print(f"{e+1}/{epochs}, Erreur = {error}")
    return network

def read_network(network):
    formule = "x"
    for layer in network:
        if isinstance(layer, Dense):
            formule = f"w{layer.input_size}." + formule + f" + b{layer.output_size}"
        else:
            formule = "tanh(" + formule + ")"
    return formule

def get_formula(layer):
    if isinstance(layer, Dense):
        input_size, output_size = layer.input_size, layer.output_size
        return f"w{output_size} dot {input_size} + b{output_size}"
    elif isinstance(layer, Tanh):
        return "tanh"
    else:
        raise ValueError("Unknown layer type")


def detail_network(network):
    pass


print(read_network(network))

class FCN(nn.Module):
    def __init__(self, _layers, act=nn.Tanh()):
        super().__init__()
        self.layers = _layers
        self.activation = act
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1])
                                      for i in range(len(self.layers) - 1)])
        self.iter = 0  # For the Optimizer

        for i in range(len(self.layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, _x):
        if not torch.is_tensor(_x):
            _x = torch.from_numpy(_x)
        a = _x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

def read_pytorch_model(model):
    # Parcourir toutes les sous-couches du modèle et obtenir leur représentation
    layers = []
    for name, module in model.named_modules():
        if name != "":
            layer_str = f"{name}({module.__class__.__name__})"
            layers.append(layer_str)

    # Combiner les représentations des sous-couches en une seule chaîne de caractères
    network_str = " -> ".join(layers)

    return network_str

def get_model_formula(model):
    formula = "tanh("
    with torch.no_grad():
        for i, module in enumerate(model.modules()):
            if isinstance(module, nn.Linear):
                w = module.weight.flatten().tolist()
                b = module.bias.flatten().tolist()
                if i == 1:
                    formula += f"{w[0]:.4f}*tanh({w[1]:.4f}*tanh({w[2]:.4f}*x + {b[2]:.4f}) + {b[1]:.4f}) + {b[0]:.4f}"
                elif i == 3:
                    formula += f"*tanh({w[0]:.4f}*tanh({w[1]:.4f}*tanh({w[2]:.4f}*x + {b[2]:.4f}) + {b[1]:.4f}) + {b[0]:.4f})"
    return formula


from torchsummary import summary
# net = FCN([2, 32, 32, 1])
net = torch.load('/Users/lucas/Documents/memoire/GPINN/models/dehnen.pt')
print(get_model_formula(net))