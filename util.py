import random

import numpy as np
import torch


def get_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else device
    return device


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
