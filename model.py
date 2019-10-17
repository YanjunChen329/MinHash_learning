import torch
from torch import nn

class FCN(nn.Module):
    def __init__(self, dimension, num_layers=3, num_class=2):
        super(FCN, self).__init__()

        self.first_layer = nn.Linear(dimension, 1000)
        mid_layers = []
        for i in range(num_layers - 2):
            mid_layers.append(nn.Linear(1000, 1000))
            mid_layers.append(nn.ReLU())
        self.mid_layers = nn.Sequential(*mid_layers)

        self.last_layer = nn.Linear(1000, num_class)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.last_layer(x)
        x = self.softmax(x)
        return x

