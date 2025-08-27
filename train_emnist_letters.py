import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

def get_device():
    try:
        if torch.backends.mps.is_available():  # Apple Silicon GPU
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

class RotateCCW90:
    def __call__(self, img):
        # EMNIST oryantasyonu düzeltmek için: -90 derece (CCW 90)
        return TF.rotate(img, -90)

class HFlip:
    def __call__(self, img):
        return TF.hflip(img)

def upright_transform():
    # MNIST istatistikleri ile normalize
    return transforms.Compose([
        RotateCCW90(),
        HFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

class SmallCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_one_epoch(model, loader, crit, opt):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / total

def main():
    DATA_ROOT = "./data"
    OUT_PATH = "emnist_letters_cnn.pt"

    BATCH_SIZE = 128
    EPOCHS = 3
    LR = 1e-3
    NUM_WORKERS = 0  # macOS/Python 3.12 için güvenli

    print(f"Device: {DEVICE}")

    transform = upright_transform()
    print("EMNIST (letters) indiriliyor/yükleniyor...")
    train_ds = datasets.EMNIST(root=DATA_ROOT, split="letters", train=True, download=True, transform=transform)
    test_ds  = datasets.EMNIST(root=DATA_ROOT, split="letters", train=False, download=True, transform=transform)

    # EMNIST letters: etiketler 1..26 -> 0..25'e kaydır
    train_ds.targets = train_ds.targets - 1
    test_ds.targets  = test_ds.targets  - 1

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=256,       shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    model = SmallCNN(num_classes=26).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, crit, opt)
        val_acc = evaluate(model, test_loader)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | {dt:.1f}s")

    torch.save({"state_dict": model.state_dict()}, OUT_PATH)
    print(f"Model kaydedildi: {OUT_PATH}")

if __name__ == "__main__":
    main()