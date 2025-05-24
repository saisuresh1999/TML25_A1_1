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
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_shadow_model(train_dataset, val_dataset=None, device='cuda', epochs=200, batch_size=64, lr=0.001, weight_decay=1e-4, patience=5):
    model = ShadowResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    best_val_acc = 0
    patience_counter = 0

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
            val_acc = correct / total
            print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    return model
