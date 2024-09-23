"""
    Very simple MLP. Fully connected NN with ReLU activations.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int,int,int], num_classes: int, hidden_sizes=(10, 10), bias=False, save=False) -> None:
        super().__init__()
        ch, w, h = input_shape
        self.input_size = ch*w*h
        self.layers_size = list(hidden_sizes)
        self.layers_size.insert(0, self.input_size)
        self.layers_size.append(num_classes)
        self.num_layers = len(self.layers_size) + 2
        self.bias = bias
        self.save = save
        self.layers = nn.ModuleList()
        self.matrix_input_dim = self.input_size + 1 if bias else self.input_size
        
        for idx in range(len(self.layers_size) - 1):
            self.layers.append(nn.Linear(self.layers_size[idx],
                                         self.layers_size[idx + 1],
                                         bias=bias))
            if idx == len(self.layers_size) - 2:
                break
            self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor, rep=False) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        if not rep:
            for layer in self.layers:
                x = layer(x)
            return x

        self.pre_acts: list[torch.Tensor] = []
        self.acts: list[torch.Tensor] = []

        x = self.layers[0](x) # Linear
        if self.save:
            self.pre_acts.append(x.detach().clone())
        x = self.layers[1](x) # RELU
        if self.save:
            self.acts.append(x.detach().clone())

        for i in range(2, len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
        return x

    def get_weights(self) -> list[torch.Tensor]:
        w: list[torch.Tensor] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w.append(m.weight.data)
        return w

    def get_biases(self) -> list[torch.Tensor]:
        b: list[torch.Tensor] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                b.append(m.bias.data)
        return b

    def init(self) -> None:
        def init_func(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
        self.apply(init_func)
