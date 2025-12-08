# FINE TUNING ResNet50

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS_DIR = "./datasets"

# --- Transformacije ---
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Dataset & DataLoader ---
train_ds = datasets.ImageFolder(root=f"{DATASETS_DIR}/imagenette2/train", transform=train_tf)
val_ds   = datasets.ImageFolder(root=f"{DATASETS_DIR}/imagenette2/val",   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# --- Model ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)   # Imagenette = 10 klasa
model = model.to(DEVICE)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# --- Trening ---
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

# --- Evaluacija ---
@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    for imgs, targets in val_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    acc = 100.0 * correct / total
    return acc

# --- Glavni loop ---
EPOCHS = 5
for epoch in range(1, EPOCHS+1):
    train_one_epoch(epoch)
    val_acc = evaluate()
    print(f"Validation Accuracy after epoch {epoch}: {val_acc:.2f}%")
