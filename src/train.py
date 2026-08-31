import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm
import random
import torchvision.models as models
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = os.path.join("config", "config.json")
MODEL_DIR = "models"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


# ------------------------------
# Label smoothing loss
# ------------------------------
class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.smoothing) + (self.smoothing / 2)
        return self.bce(logits, smooth_targets)


# ------------------------------
# Model
# ------------------------------
class SliceClassifier(nn.Module):
    def __init__(self, dropout_rate=0.4, context=1):
        super().__init__()
        in_channels = 2 * context + 1
        #at context=2 we are effectively taking in a 5 channel input
        backbone = models.resnet18(weights='IMAGENET1K_V1')

        #repeating the initial weights for the extra channels since resnet is 3 channel by default
        orig_weight = backbone.conv1.weight.data
        new_weight  = orig_weight.repeat(1, in_channels, 1, 1)
        new_weight  = new_weight[:, :in_channels, :, :] / (in_channels / 3.0)

        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.conv1.weight.data = new_weight

        self.encoder     = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout     = nn.Dropout(dropout_rate)
        self.classifier  = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.classifier(self.dropout(features))


# ------------------------------
# Dataset
# ------------------------------
class SliceDataset(Dataset):
    def __init__(self, pt_files, transform=None, context=1):
        self.pt_files = pt_files
        self.transform = transform
        self.context = context

        self.cache = {}

        self.samples = []

        for f in pt_files:
            data = torch.load(f, weights_only=False)
            tensors = data["tensors"]
            labels  = data["slice_labels"]
            self.cache[f] = (tensors, labels)

            D = tensors.shape[0]
            for i in range(D):
                self.samples.append((f, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx]
        tensors, labels = self.cache[file_path]

        D = tensors.shape[0]
        indices = []
        for offset in range(-self.context, self.context + 1):
            idx_clamped = max(0, min(D - 1, slice_idx + offset))
            indices.append(idx_clamped)

        
        stack = torch.stack([tensors[i] for i in indices], dim=0).float()
        label = labels[slice_idx].float()

        if self.transform:
            stack = self.transform(stack)

        return stack, label



class RandomGamma: #simulates variations like T1 and T2 in MR based imaging
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-6:
            return img
        img_unit = (img - mn) / (mx - mn)
        gamma = np.random.uniform(*self.gamma_range)
        img_gamma = img_unit ** gamma
        
        return img_gamma * (mx - mn) + mn


class RandomBiasField: #helps model generalize across different MRI scanners and variations
    def __init__(self, strength=0.4, order=3):
        self.strength = strength
        self.order = order

    def __call__(self, img):
        
        C, H, W = img.shape
        
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        field = torch.ones(H, W)
        for i in range(self.order):
            for j in range(self.order - i):
                coeff = np.random.uniform(-self.strength, self.strength)
                field = field + coeff * (yy ** i) * (xx ** j)


        return img * field.unsqueeze(0)


class RandomGaussianNoise:
    def __init__(self, std_range=(0.0, 0.15)):
        self.std_range = std_range

    def __call__(self, img):
        std = np.random.uniform(*self.std_range) * img.std()
        return img + torch.randn_like(img) * std


class RandomIntensityScale:
    def __init__(self, scale_range=(0.75, 1.25)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = np.random.uniform(*self.scale_range)
        return (img * scale).clamp(0, 1)


class RandomIntensityShift:
    def __init__(self, shift_range=(-0.15, 0.15)):  # absolute in [0,1] space
        self.shift_range = shift_range

    def __call__(self, img):
        shift = np.random.uniform(*self.shift_range)
        return (img + shift).clamp(0, 1)


# ------------------------------
# Augmentations
# ------------------------------
train_transform = transforms.Compose([
    # Spatial
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.85, 1.15)),

    # Intensity based augmentations (inspired by SynthStrip)
    transforms.RandomApply([RandomGamma(gamma_range=(0.5, 2.0))],     p=0.5),
    transforms.RandomApply([RandomBiasField(strength=0.3, order=3)],  p=0.5),
    transforms.RandomApply([RandomGaussianNoise(std_range=(0.0, 0.1))], p=0.4),
    transforms.RandomApply([RandomIntensityScale(scale_range=(0.75, 1.25))], p=0.5),
    transforms.RandomApply([RandomIntensityShift(shift_range=(-0.25, 0.25))], p=0.5),
])


def find_best_threshold(model, dataloader, min_recall=0.85):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            probs = torch.sigmoid(model(imgs.to(device)).squeeze()).cpu().numpy()
            all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1, best_t = 0.0, 0.5

    # Scan thresholds
    for t in np.arange(0.10, 0.70, 0.01):
        preds = (all_probs > t).astype(float)
        rec = recall_score(all_labels, preds, zero_division=0)
        if rec < min_recall:         
            continue
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Fallback: if no threshold satisfies min_recall, just maximise F1
    if best_f1 == 0.0:
        for t in np.arange(0.10, 0.70, 0.01):
            preds = (all_probs > t).astype(float)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

    return best_t, best_f1

def compute_metrics(model, dataloader, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs).squeeze()).cpu().numpy()
            
            if probs.ndim == 0: #exception for batch = 1
                probs = [float(probs)]
            preds = (np.array(probs) > threshold).astype(float)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    return acc, rec, prec, f1


def save_config(config_data):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=4)


# ------------------------------
# Main training
# ------------------------------
if __name__ == "__main__":
    pt_files = glob.glob("I:/Segmentation_preprocessed/*.pt")
    print(f"Found {len(pt_files)} volume files")

    train_vols, val_vols = train_test_split(pt_files, test_size=0.2, random_state=42)
    print(f"Training volumes: {len(train_vols)}, Validation volumes: {len(val_vols)}")
    CONTEXT=2
    train_dataset = SliceDataset(train_vols, transform=train_transform,context=CONTEXT)
    val_dataset   = SliceDataset(val_vols,   transform=None,context=CONTEXT)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=48, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # Class weighing
    pos_count = sum(1 for _, l in train_dataset if l == 1)
    neg_count = sum(1 for _, l in train_dataset if l == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)]).to(device)
    print(f"Positive: {pos_count}, Negative: {neg_count}, pos_weight: {pos_weight.item():.3f}")




    model     = SliceClassifier(dropout_rate=0.4,context=CONTEXT).to(device)
    criterion = SmoothBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=0.1)

    best_model_name = "best_slice_classifier.pt"
    os.makedirs(MODEL_DIR, exist_ok=True)

    num_epochs = 40
    backbone_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": 3e-5},
        {"params": classifier_params, "lr": 3e-4},
    ], weight_decay=1e-4)

   
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)


    patience = 3
    best_val_f1 = -1.0
    best_threshold = 0.5
    best_val_acc = 0.0
    best_val_rec = 0.0
    best_val_prec = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    runtime_config = {
        "model_name": best_model_name,
        "context_slices": CONTEXT,
        "resize_width": 256,
        "resize_height": 256,
        "spacing": "1mmx1mmx1mm",
        "normalization": "p99",
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_weight": pos_weight.item(),
        "epochs": num_epochs,
        "batch_size": 48,
        "backbone_lr": 3e-5,
        "classifier_lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealing",
        "dropout": 0.4,
        "label_smoothing": 0.1,
        "threshold": best_threshold,
        "best_val_accuracy": best_val_acc,
        "best_val_recall": best_val_rec,
        "best_val_precision": best_val_prec,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
    }
    save_config(runtime_config)

    for epoch in range(num_epochs):
        model.train()
        train_correct = train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        epoch_loss = 0
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.3f}", acc=f"{train_correct/train_total:.3f}")
            epoch_loss += loss.item()

        scheduler.step()
        train_acc = train_correct / train_total
        current_lr = scheduler.get_last_lr()[0]
        epoch_loss /= len(train_loader)

        opt_threshold, _ = find_best_threshold(model, val_loader, min_recall=0.85)
        val_acc, val_rec, val_prec, val_f1 = compute_metrics(model, val_loader, threshold=opt_threshold)

        print(
            f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}  Recall: {val_rec:.4f}  "
            f"Prec: {val_prec:.4f}  F1: {val_f1:.4f} | "
            f"Threshold: {opt_threshold:.2f}  LR: {current_lr:.2e}"
        )

        runtime_config.update({
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_recall": val_rec,
            "val_precision": val_prec,
            "val_f1": val_f1,
            "val_threshold": opt_threshold,
            "lr": current_lr,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = opt_threshold
            best_val_acc = val_acc
            best_val_rec = val_rec
            best_val_prec = val_prec
            best_epoch = epoch + 1
            runtime_config.update({
                "threshold": best_threshold,
                "best_val_accuracy": best_val_acc,
                "best_val_recall": best_val_rec,
                "best_val_precision": best_val_prec,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
            })
            torch.save({
                "model_state": model.state_dict(),
                "threshold": best_threshold,
                "epoch": epoch + 1,
                "val_f1": best_val_f1,
                "val_recall": best_val_rec,
                "val_prec": best_val_prec,
            }, os.path.join(MODEL_DIR, best_model_name))
            epochs_no_improve = 0
            print(f"  -> Saved (F1={best_val_f1:.4f}, threshold={best_threshold:.2f}, recall={best_val_rec:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                save_config(runtime_config)
                break

        save_config(runtime_config)

    checkpoint = torch.load(os.path.join(MODEL_DIR, best_model_name), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nDone. Best val F1: {checkpoint['val_f1']:.4f} at epoch {checkpoint['epoch']}")
    print(f"Use threshold={checkpoint['threshold']:.2f} at inference")
    print(f"Recall={checkpoint['val_recall']:.4f}, Precision={checkpoint['val_prec']:.4f}")
    
