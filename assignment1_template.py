import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import requests
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from torch.serialization import add_safe_globals
from torchvision.models import ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalize
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]
transform = transforms.Normalize(mean=mean, std=std)


# Model
model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("01_MIA.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

# Dataset
class MembershipDataset(Dataset):
    def __init__(self, ids, imgs, labels, memberships, transform=None):
        self.ids = ids
        self.imgs = imgs
        self.labels = labels
        self.memberships = memberships
        self.transform = transform

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        return self.ids[idx], img, self.labels[idx], self.memberships[idx]

    def __len__(self):
        return len(self.ids)

class PrivateDataset(Dataset):
    def __init__(self, ids, imgs, labels, transform=None):
        self.ids = ids
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        return self.ids[idx], img, self.labels[idx]

    def __len__(self):
        return len(self.ids)

# Load with safe globals
add_safe_globals([MembershipDataset])
pub_raw = torch.load("pub.pt", weights_only=False)
priv_raw = torch.load("priv_out.pt", weights_only=False)

# Inject transform
pub_raw.transform = transform
priv_raw.transform = transform

# Rewrap in clean datasets
pub = MembershipDataset(pub_raw.ids, pub_raw.imgs, pub_raw.labels, pub_raw.membership, transform)
priv = PrivateDataset(priv_raw.ids, priv_raw.imgs, priv_raw.labels, transform)

# Feature extractor
def extract_features(model, dataset):
    loader = DataLoader(dataset, batch_size=64)
    features, ids = [], []
    memberships = []
    for batch in loader:
        if len(batch) == 4:
            id_, x, y, m = batch
            memberships.extend(m.numpy())
        else:
            id_, x, y = batch
        x = x.to(device)
        with torch.no_grad():
            logits = model(x).cpu()
        probs = F.softmax(logits, dim=1)
        conf = torch.max(probs, dim=1).values
        features.extend(conf.numpy())
        ids.extend(id_)
    return np.array(ids), np.array(features), np.array(memberships) if memberships else None

# Train
pub_ids, pub_feat, pub_m = extract_features(model, pub)
clf = LogisticRegression().fit(pub_feat.reshape(-1, 1), pub_m)

# Predict
priv_ids, priv_feat, _ = extract_features(model, priv)
priv_scores = clf.predict_proba(priv_feat.reshape(-1, 1))[:, 1]

# Submit
df = pd.DataFrame({"ids": priv_ids, "score": priv_scores})
df.to_csv("test.csv", index=None)
response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "01354973"})
print(response.json())
