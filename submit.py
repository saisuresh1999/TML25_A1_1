import torch
import pandas as pd
import requests
import joblib
from torch.serialization import add_safe_globals
from data.dataset import MembershipDataset
add_safe_globals([MembershipDataset])
from data.dataset import load_priv_dataset
from attacks.feature_extractor import extract_features
from attacks.attack_model import AttackModel
from models.shadow_model import ShadowResNet18

from config import (
    DEVICE, PRIV_PATH, SHADOW_MODEL_PATH, ATTACK_MODEL_PATH,
    SUBMISSION_CSV, TEAM_TOKEN
)

def submit():
    print("🔹 Loading private dataset...")
    priv_data = load_priv_dataset(PRIV_PATH)

    print("🔹 Loading trained shadow model...")
    model = ShadowResNet18()
    model.load_state_dict(torch.load(SHADOW_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("🔹 Extracting features from private data...")
    priv_feat = extract_features(model, priv_data, device=DEVICE)

    print("🔹 Loading attack model...")
    attacker = AttackModel()
    attacker.clf = joblib.load(ATTACK_MODEL_PATH)

    print("🔹 Predicting membership scores...")
    scores = attacker.predict_proba(priv_feat["features"])

    print("🔹 Saving submission file...")
    df = pd.DataFrame({"ids": priv_feat["ids"], "score": scores})
    df.to_csv(SUBMISSION_CSV, index=False)

    print("🔹 Submitting to server...")
    response = requests.post(
        "http://34.122.51.94:9090/mia",
        files={"file": open(SUBMISSION_CSV, "rb")},
        headers={"token": TEAM_TOKEN}
    )
    print("✅ Server Response:", response.json())

if __name__ == "__main__":
    submit()
