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
    BATCH_SIZE, EPOCHS, SPLIT_RATIO, N_SHADOW_MODELS, MEMBER_CONFIG_PATH, NON_MEMBER_CONFIG_PATH
)
from config import set_seed
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

set_seed()

def main():
    k = N_SHADOW_MODELS
    print(f"Generating {k} shadow splits...")
    shadow_splits = generate_shadow_splits(PUB_PATH, k=k, split_ratio=SPLIT_RATIO)

    all_features, all_labels = [], []
    member_confidences, nonmember_confidences = [], []

    for i, (train_ds, test_ds) in enumerate(shadow_splits):
        print(f"🔹 Training shadow model {i+1}/{k}...")
        model = train_shadow_model(train_ds, test_ds, device=DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE)

        print(f"🔹 Extracting features from shadow model {i+1}...")
        train_feat = extract_features(model, train_ds, device=DEVICE, batch_size=BATCH_SIZE)
        test_feat = extract_features(model, test_ds, device=DEVICE, batch_size=BATCH_SIZE)

        def get_correct_confidences(feat):
            probs = feat["features"]
            labels = feat["labels"]
            return probs[np.arange(len(labels)), labels]

        member_confidences.extend(get_correct_confidences(train_feat))
        nonmember_confidences.extend(get_correct_confidences(test_feat))

        all_features.append(test_feat["features"])
        all_labels.append(test_feat["membership"])

    np.save(MEMBER_CONFIG_PATH, np.array(member_confidences))
    np.save(NON_MEMBER_CONFIG_PATH, np.array(nonmember_confidences))
    print("Saved member and non-member confidences for LiRA.")

    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_labels)

    print("🔹 Training attack model on aggregated shadow outputs...")
    attacker = AttackModel()
    attacker.train(X_train, y_train)

    torch.save(model.state_dict(), SHADOW_MODEL_PATH)
    joblib.dump(attacker.clf, ATTACK_MODEL_PATH)

    print("Finished training attack model on shadow ensemble.")

    def evaluate(scores, labels):
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        tpr_at_005 = max(tpr[fpr <= 0.05]) if any(fpr <= 0.05) else 0.0
        print(f"Local Evaluation → AUC: {auc:.4f}, TPR@FPR=0.05: {tpr_at_005:.4f}")

    scores = attacker.predict_proba(X_train)
    evaluate(scores, y_train)

if __name__ == "__main__":
    main()
