import torch
from torch import nn

from detection_method.neural_nets.model import MLP


class MlpRepresentation:
    def __init__(self, model: MLP, device:str="cpu") -> None:
        self.model = model
        self.device = device
        self.mlp_weights: list[torch.Tensor] = []
        self.mlp_biases: list[torch.Tensor] = []
        self.input_size: int = model.input_size
        self.act_fn = nn.ReLU()

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                self.mlp_weights.append(layer.weight.data)
                if self.model.bias:
                    self.mlp_biases.append(layer.bias.data)
                else:
                    self.mlp_biases.append(torch.zeros(layer.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = torch.flatten(x).to(device=self.device)
        self.model.save = True
        _ = self.model(flat_x, rep=True)

        A = self.mlp_weights[0].to(self.device) * flat_x
        a = self.mlp_biases[0]

        for i in range(1, len(self.mlp_weights)):
            layeri = self.mlp_weights[i].to(self.device)
            pre_act = self.model.pre_acts[i-1]
            post_act = self.model.acts[i-1]
            vertices = post_act / pre_act
            vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0

            B = layeri * vertices
            A = torch.matmul(B, A)

            if self.model.bias:
                b = self.mlp_biases[i]
                a = torch.matmul(B, a) + b

        if self.model.bias:
            return torch.cat([A, a.unsqueeze(1)], dim=1)
        else:
            return A
