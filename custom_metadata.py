import os
import json


def get_custom_metadata(info, audio):
    """
    custom function to pass jsons to model
    """
    audio_path = info["path"]
    json_path = os.path.splitext(audio_path)[0] + ".json"

    with open(json_path, "r") as f:
        metadata = json.load(f)

    return metadata