import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def extract_features(model, dataset, device='cuda', batch_size=64):
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

            features.append(probs)         # 💡 Now using full 44-dim softmax
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
