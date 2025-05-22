from torch.serialization import add_safe_globals
from data.dataset import MembershipDataset
add_safe_globals([MembershipDataset])

import torch
import numpy as np
from data.dataset import generate_shadow_splits
from models.shadow_model import train_shadow_model
from attacks.feature_extractor import extract_features
from attacks.attack_model import AttackModel
from config import (
    DEVICE, PUB_PATH, SHADOW_MODEL_PATH, ATTACK_MODEL_PATH,
    BATCH_SIZE, EPOCHS, SPLIT_RATIO
)
from config import set_seed


set_seed()

def main():
    k = 3  # number of shadow models
    print(f"🔹 Generating {k} shadow splits...")
    shadow_splits = generate_shadow_splits(PUB_PATH, k=k, split_ratio=SPLIT_RATIO)

    all_features = []
    all_labels = []

    for i, (train_ds, test_ds) in enumerate(shadow_splits):
        print(f"🔹 Training shadow model {i+1}/{k}...")
        model = train_shadow_model(train_ds, test_ds, device=DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE)

        print(f"🔹 Extracting features from shadow model {i+1}...")
        feat = extract_features(model, test_ds, device=DEVICE, batch_size=BATCH_SIZE)

        all_features.append(feat["features"])
        all_labels.append(feat["membership"])

    # Merge all shadow outputs into one training set
    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_labels)

    print("🔹 Training attack model on aggregated shadow outputs...")
    attacker = AttackModel()
    attacker.train(X_train, y_train)

    # Save the last shadow model (just for inference structure)
    torch.save(model.state_dict(), SHADOW_MODEL_PATH)
    import joblib
    joblib.dump(attacker.clf, ATTACK_MODEL_PATH)

    print("✅ Finished training attack model on shadow ensemble.")

if __name__ == "__main__":
    main()
