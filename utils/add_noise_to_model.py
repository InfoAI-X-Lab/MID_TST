import torch
import torch.nn as nn
import copy
def add_gaussian_noise_to_model(model, noise_std_ratio, inplace=False, seed=None):

    if not inplace:
        model = copy.deepcopy(model)

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # === Weight noise ===
                if layer.weight is not None:
                    weight_std = layer.weight.std().item()
                    weight_noise = torch.randn_like(layer.weight) * (weight_std * noise_std_ratio)
                    layer.weight.data += weight_noise

                # === Bias noise ===
                if layer.bias is not None:
                    bias_std = layer.bias.std().item()
                    bias_noise = torch.randn_like(layer.bias) * (bias_std * noise_std_ratio)
                    layer.bias.data += bias_noise

    return model
