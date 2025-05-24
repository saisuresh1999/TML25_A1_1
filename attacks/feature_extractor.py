import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def compute_softmax_features(probs):
    # Confidence (max probability)
    confidence = np.max(probs, axis=1).reshape(-1, 1)

    # Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1).reshape(-1, 1)

    # Margin = top1 - top2
    sorted_probs = np.sort(probs, axis=1)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)

    return confidence, entropy, margin

def extract_features(model, dataset, device='cuda', batch_size=64, include_stats=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    features, labels, ids, memberships = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                id_, x, y, m = batch
                memberships.extend(m.numpy())
            else:
                id_, x, y = batch

            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            if include_stats:
                conf, ent, marg = compute_softmax_features(probs)
                base_feats = np.hstack([probs, conf, ent, marg])  # [44 + 3]
            else:
                base_feats = probs

            features.append(base_feats)
            labels.extend(y.numpy())
            ids.extend(id_)

    features = np.concatenate(features, axis=0)
    result = {
        "ids": np.array(ids),
        "features": features,
        "labels": np.array(labels)
    }

    if memberships:
        result["membership"] = np.array(memberships)

    return result
