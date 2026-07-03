import os
import json
import torch


def get_custom_metadata(info, audio):
    """
    custom function to pass jsons to model
    """
    audio_path = info["path"]
    json_path = os.path.splitext(audio_path)[0] + ".json"

    with open(json_path, "r") as f:
        metadata = json.load(f)
        # int cast needed by training skript
        metadata["seconds_total"] = torch.tensor(int(metadata["seconds_total"]), dtype=torch.long)

    return metadata