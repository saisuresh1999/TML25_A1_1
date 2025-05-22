import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

# Define constants
MEAN = [0.2980, 0.2962, 0.2987]
STD = [0.2886, 0.2875, 0.2889]

# Define Dataset first (needed before registering!)
class MembershipDataset(Dataset):
    def __init__(self, ids, imgs, labels, memberships=None, transform=None):
        self.ids = ids
        self.imgs = imgs
        self.labels = labels
        self.memberships = memberships
        self.transform = transform

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        if self.memberships is not None:
            return self.ids[idx], img, self.labels[idx], self.memberships[idx]
        else:
            return self.ids[idx], img, self.labels[idx]

    def __len__(self):
        return len(self.ids)

# Now safe to register!
from torch.serialization import add_safe_globals
add_safe_globals([MembershipDataset])

def get_transform():
    return transforms.Normalize(mean=MEAN, std=STD)

def load_pub_dataset(path: str, split_ratio=0.5, seed=42):
    raw = torch.load(path, weights_only=False)
    transform = get_transform()
    full = MembershipDataset(raw.ids, raw.imgs, raw.labels, raw.membership, transform=transform)

    n_total = len(full)
    n_train = int(split_ratio * n_total)
    n_val = n_total - n_train

    torch.manual_seed(seed)
    shadow_train, shadow_test = random_split(full, [n_train, n_val])
    return shadow_train, shadow_test

def load_priv_dataset(path: str):
    raw = torch.load(path, weights_only=False)
    transform = get_transform()
    return MembershipDataset(raw.ids, raw.imgs, raw.labels, memberships=None, transform=transform)

def generate_shadow_splits(path: str, k: int = 3, split_ratio=0.5, seed=42):
    from torch.serialization import add_safe_globals
    add_safe_globals([MembershipDataset])
    raw = torch.load(path, weights_only=False)
    transform = get_transform()

    full = MembershipDataset(raw.ids, raw.imgs, raw.labels, raw.membership, transform=transform)
    n_total = len(full)
    n_samples = int(2 * split_ratio * n_total)

    torch.manual_seed(seed)
    splits = []

    for _ in range(k):
        if n_samples <= n_total:
            indices = torch.randperm(n_total)[:n_samples].tolist()
        else:
            indices = torch.randint(0, n_total, size=(n_samples,)).tolist()  # ✅ allow overlapping

        subset = torch.utils.data.Subset(full, indices)
        train_size = n_samples // 2
        test_size = n_samples - train_size
        train, test = random_split(subset, [train_size, test_size])
        splits.append((train, test))

    return splits
