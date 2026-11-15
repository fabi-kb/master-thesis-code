import torch
import torch.nn as nn
from utils.initializer import initialize_weights

class MLP(nn.Module):
    def __init__(
            self, 
            config: dict,
            input_dim : int | tuple,
            output_dim : int
    ):
        super().__init__()

        self.config = config
        
        # Network parameters
        self.inp_dim = input_dim
        self.hidden_dim = config["hidden_dim"]
        self.hidden_layers = config["hidden_layers"]
        self.bias = config.get("bias", True)
        self.out_dim = output_dim

        # prevent unexpected names in config file
        if config["activation"] not in ["ReLU", "Sigmoid", "Tanh", "Softplus", "SiLU"]:
            raise ValueError(f"Unexpected activation function: {config['activation']}")

        if config["layer_norm"] not in ["LayerNorm", "BatchNorm1d","RMSNorm", "None"]:
            raise ValueError(f"Unexpected layer_norm type: {config['layer_norm']}")

        self.activation = getattr(nn, config["activation"]) 

        self.layer_norm = getattr(nn, config["layer_norm"]) if config["layer_norm"] != "None" else None

        # Build MLP

        layers = []

        for i in range(self.hidden_layers): 
            if i == 0:
                layers.append(nn.Linear(self.inp_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias))

            if config["layer_norm"] != "None":
                layers.append(self.layer_norm(self.hidden_dim, eps=1e-03))

            layers.append(self.activation())

        layers.append(nn.Linear(self.hidden_dim, self.out_dim, bias=self.bias))

        self.mlp = nn.Sequential(*layers)

        initialize_weights(self.mlp, config["weight_dist"], config["output_weight_dist"], config["fan_mode"])


    def forward(self, x):
        return self.mlp(x)
    


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, norm=True, activation=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_Linear", nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_LayerNorm", nn.LayerNorm(3 * hidden_size, eps=1e-03))
        
    @property
    def state_size(self):
        return self._hidden_size
    
    def forward(self, input, state):
        parts = self.layers(torch.cat([input, state], dim=-1))
        r, z, h = torch.split(parts, self._hidden_size, dim=-1)
        r = torch.sigmoid(r)
        z = self._activation(z * r)
        h = torch.sigmoid(h + self._update_bias) 
        out = h * z + (1 - h) * state
        return out, out