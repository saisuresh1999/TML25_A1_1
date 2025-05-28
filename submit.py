import torch
import numpy as np
import pandas as pd
import requests
from torch.serialization import add_safe_globals
from data.dataset import MembershipDataset
add_safe_globals([MembershipDataset])

from data.dataset import load_priv_dataset
from attacks.feature_extractor import extract_features
from models.shadow_model import ShadowResNet18
from config import (
    DEVICE, PRIV_PATH, SHADOW_MODEL_PATH, SUBMISSION_CSV, TEAM_TOKEN, MEMBER_CONFIG_PATH, NON_MEMBER_CONFIG_PATH
)

def submit():
    print("Loading private dataset...")
    priv_data = load_priv_dataset(PRIV_PATH)

    print("Loading trained shadow model...")
    model = ShadowResNet18()
    model.load_state_dict(torch.load(SHADOW_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Extracting features from private data...")
    priv_feat = extract_features(model, priv_data, device=DEVICE)

    print("Loading saved shadow confidences for LiRA...")
    member_conf = np.load(MEMBER_CONFIG_PATH)
    nonmember_conf = np.load(NON_MEMBER_CONFIG_PATH)

    print("Running LiRA scoring...")
    priv_probs = priv_feat["features"]
    priv_labels = priv_feat["labels"]
    correct_conf = priv_probs[np.arange(len(priv_labels)), priv_labels]

    # Fit Gaussians
    mu_m, std_m = member_conf.mean(), member_conf.std() + 1e-6
    mu_nm, std_nm = nonmember_conf.mean(), nonmember_conf.std() + 1e-6

    from scipy.stats import norm
    p_m = norm.pdf(correct_conf, loc=mu_m, scale=std_m)
    p_nm = norm.pdf(correct_conf, loc=mu_nm, scale=std_nm)

    scores = p_m / (p_m + p_nm)

    print("Saving submission file...")
    df = pd.DataFrame({"ids": priv_feat["ids"], "score": scores})
    df.to_csv(SUBMISSION_CSV, index=False)

    print("Submitting to server...")
    response = requests.post(
        "http://34.122.51.94:9090/mia",
        files={"file": open(SUBMISSION_CSV, "rb")},
        headers={"token": TEAM_TOKEN}
    )
    print("Server Response:", response.json())

if __name__ == "__main__":
    submit()
