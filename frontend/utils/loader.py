import pickle
from pathlib import Path

def load_model(model_path: str):
    path = Path(model_path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
