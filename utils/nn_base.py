from torch import nn


class NNBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def n_layers(self) -> int:
        return len(self.layers)

    def add_layer(self, layer):
        self.layers.append(layer)
        return layer

    def add_layers(self, layers):
        self.layers += layers
        return layers
