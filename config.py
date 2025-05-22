import torch
import random
import numpy as np

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
PUB_PATH = "pub.pt"
PRIV_PATH = "priv_out.pt"
SHADOW_MODEL_PATH = "shadow_model.pt"
ATTACK_MODEL_PATH = "attack_model.pkl"
SUBMISSION_CSV = "test.csv"

# Token (replace with yours)
TEAM_TOKEN = "01354973"

# Training
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
SPLIT_RATIO = 0.7
N_SHADOW_MODELS = 3

# Seed
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
