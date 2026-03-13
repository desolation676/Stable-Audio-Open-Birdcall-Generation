from huggingface_hub import hf_hub_download
import os


os.makedirs("./checkpoints", exist_ok=True)

config_path = hf_hub_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    filename="model_config.json",
    local_dir="./checkpoints"
)
print(f"Success! Config saved to: {config_path}")

model_path = hf_hub_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    filename="model.safetensors",
    local_dir="./checkpoints"
)
print(f"Success! Model weights saved to: {model_path}")