import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm

class ShadowResNet18(nn.Module):
    def __init__(self, num_classes=44):
        super().__init__()
        self.model = resnet18(weights=None)
        # Turn off dropout (if exists in residual blocks)
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_shadow_model(train_dataset, val_dataset=None, device='cuda', epochs=10, batch_size=64, lr=0.001):
    model = ShadowResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")
        for _, x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=total_loss/total, acc=correct/total)

        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for _, x, y, _ in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    correct += (outputs.argmax(1) == y).sum().item()
                    total += x.size(0)
            print(f"[Epoch {epoch+1}] Validation Accuracy: {correct/total:.4f}")

    return model
