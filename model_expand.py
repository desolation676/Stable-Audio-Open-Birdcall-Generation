import torch
from safetensors.torch import load_file, save_file


target_layer_name = "model.model.to_global_embed.0.weight"

tensors = load_file("checkpoints/model.safetensors")
new_tensors = {}
patched_count = 0


for key, tensor in tensors.items():
    if key == target_layer_name:
        print(f"Found target layer: {key} | Original Shape: {tensor.shape}")

        new_tensor = torch.zeros((tensor.shape[0], 2304), dtype=tensor.dtype)

        new_tensor[:, :1536] = tensor
        new_tensors[key] = new_tensor
        patched_count += 1
    else:
        new_tensors[key] = tensor

if patched_count > 0:
    print(f"\nSuccessfully patched {patched_count} layers.")

    save_file(new_tensors, "checkpoints/model_2304.safetensors")
