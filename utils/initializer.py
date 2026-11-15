import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

def target_scale(
    input_dimension: int,
    output_dimension: int,
    fan_mode: str
) -> float:
    
    fan_mode = fan_mode.lower()

    if fan_mode in {"fan_in", "in"}:
        return float(input_dimension)
    
    if fan_mode in {"fan_out", "out"}:
        return float(output_dimension)
    
    if fan_mode in {"fan_avg", "avg", "average"}:
        return 0.5 * (input_dimension + output_dimension)
    
    raise ValueError(f"Unknown fan_mode: {fan_mode}") 

@torch.no_grad()
def initialize_weights(
    modules: nn.Module, 
    weight_dist="truncated_normal", 
    output_weight_dist="truncated_normal",
    fan_mode="fan_in"
) -> None:

    weight_dist = weight_dist.lower()
    output_weight_dist = output_weight_dist.lower()
    fan_mode = fan_mode.lower()

    linear_modules = [m for m in modules.modules() if isinstance(m, nn.Linear)]
    
    for i, module in enumerate(linear_modules):
        is_last = (i == len(linear_modules) - 1)
        dist = output_weight_dist if (is_last and output_weight_dist != "none") else weight_dist

        input_dimension, output_dimension = module.in_features, module.out_features
        denominator = target_scale(input_dimension, output_dimension, fan_mode)

        if dist == "truncated_normal":
            scale = np.sqrt(1.0 / denominator)
            std = scale / 0.87962566103423978
            init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        elif dist == "uniform":
            bound = np.sqrt(1.0 / denominator)
            init.uniform_(module.weight, -bound, bound)

        elif dist == "normal":
            std = np.sqrt(1.0 / denominator)
            init.normal_(module.weight, mean=0.0, std=std)

        elif dist == "zeros":
            init.zeros_(module.weight)

        else:
            raise ValueError(f"Unknown initializer distribution: {dist}")
        
        if module.bias is not None:
            init.zeros_(module.bias)


        