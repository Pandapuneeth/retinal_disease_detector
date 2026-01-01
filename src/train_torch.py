import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler


# ===================== CONFIG ===================== #

DATA_ROOT = os.path.join("data")   # Adjust if needed (your root data folder)

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2  # if Windows complains, set to 0

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===================== DATA ===================== #

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = val_transforms


def build_dataloaders():
    print("Loading datasets...")

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    print("Classes:", train_ds.classes)

    return train_loader, val_loader, test_loader, train_ds.classes


# ===================== MODEL ===================== #

def build_model(num_classes: int):
    print("Building MobileNetV2 model...")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model.to(device)


# ===================== TRAIN ONE EPOCH ===================== #

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # FP16 TURBO MODE 🚀
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time

    return epoch_loss, epoch_acc, elapsed


# ===================== EVALUATE (NOW IN FP16 ALSO) ===================== #

@torch.no_grad()
def evaluate(model, loader, criterion, split_name="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # FP16 evaluation as well 🚀
    with autocast(enabled=(device.type == "cuda")):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"{split_name} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc


# ===================== MAIN LOOP ===================== #

def main():
    train_loader, val_loader, test_loader, class_names = build_dataloaders()

    num_classes = len(class_names)
    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_model_path = os.path.join(
        MODEL_DIR,
        f"mobilenetv2_retinal_oct_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )

    print("\n===== START TRAINING =====")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Time: {train_time:.1f}s")

        val_loss, val_acc = evaluate(model, val_loader, criterion, split_name="Val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, best_model_path)
            print(f"✅ New best model saved to: {best_model_path}")

    print("\n===== TRAINING DONE =====")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%")

    print("\nRunning final test evaluation on best model...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate(model, test_loader, criterion, split_name="Test")


if __name__ == "__main__":
    main()
