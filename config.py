import torch
import random
import numpy as np

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
PUB_PATH = "pub.pt"
PRIV_PATH = "priv_out.pt"
# SHADOW_MODEL_PATH = "/content/drive/MyDrive/TML25_A1_1/shadow_model.pt"
# ATTACK_MODEL_PATH = "/content/drive/MyDrive/TML25_A1_1/attack_model.pkl"
# MEMBER_CONFIG_PATH = "/content/drive/MyDrive/TML25_A1_1/member_conf.npy"
# NON_MEMBER_CONFIG_PATH = "/content/drive/MyDrive/TML25_A1_1/nonmember_conf.npy"
SHADOW_MODEL_PATH = "shadow_model.pt"
ATTACK_MODEL_PATH = "attack_model.pkl"
MEMBER_CONFIG_PATH = "member_conf.npy"
NON_MEMBER_CONFIG_PATH = "nonmember_conf.npy"
SUBMISSION_CSV = "test.csv"

# Token (replace with yours)
TEAM_TOKEN = "01354973"

# Training
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SPLIT_RATIO = 0.7
N_SHADOW_MODELS = 7

# Seed
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
