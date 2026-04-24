#!/usr/bin/env python3
"""Fine-tune MobileNetV3-small as a binary crib-occupancy classifier.

Expects a dataset directory like:
    dataset/
        empty/*.jpg
        occupied/*.jpg

Outputs:
    presence_model.pt    - torch state_dict
    presence_model.json  - metadata (architecture, transforms, val metrics)

Run on Mac (MPS or CPU); copy the .pt + .json to the VM for deployment.
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm

# Index 0 = empty, index 1 = occupied. Order matters: the detector reads
# probability at index 1 as "probability of occupied".
CLASSES = ["empty", "occupied"]
INPUT_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class CribDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            for p in sorted((root / cls).glob("*.jpg")):
                self.samples.append((p, idx))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> nn.Module:
    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASSES))
    return model


def stratified_indices(samples, val_fraction: float, seed: int):
    """Split indices so each class is represented in val at ~val_fraction."""
    by_class: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(samples):
        by_class.setdefault(label, []).append(i)
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_fraction)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def evaluate(model, loader, device) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = 0
    total = 0
    confusion = [[0, 0], [0, 0]]  # confusion[true][pred]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t][p] += 1
    return correct / max(total, 1), confusion


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--output", type=Path, default=Path("presence_model.pt"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print(f"device: {device}")

    normalize = T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    train_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        T.ToTensor(),
        normalize,
    ])
    eval_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        normalize,
    ])

    # Build two datasets with the same samples but different transforms,
    # then Subset with stratified indices.
    train_full = CribDataset(args.dataset, train_tf)
    eval_full = CribDataset(args.dataset, eval_tf)
    if len(train_full) == 0:
        print(f"no samples in {args.dataset}")
        return 1

    class_counts = Counter(label for _, label in train_full.samples)
    print(f"dataset: {len(train_full)} samples "
          f"({class_counts[0]} empty, {class_counts[1]} occupied)")

    train_idx, val_idx = stratified_indices(
        train_full.samples, args.val_fraction, args.seed,
    )
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    print(f"split:   {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_acc = 0.0
    best_state = None
    best_confusion = [[0, 0], [0, 0]]

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)
            pbar.set_postfix(loss=f"{running_loss / seen:.4f}")
        pbar.close()
        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_acc, confusion = evaluate(model, val_loader, device)
        dur = time.monotonic() - t0
        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({dur:.1f}s)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_confusion = confusion

    print()
    print(f"best val accuracy: {best_acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(f"            pred_empty  pred_occupied")
    print(f"empty       {best_confusion[0][0]:10d}  {best_confusion[0][1]:13d}")
    print(f"occupied    {best_confusion[1][0]:10d}  {best_confusion[1][1]:13d}")

    assert best_state is not None
    torch.save(best_state, args.output)
    meta = {
        "architecture": "mobilenet_v3_small",
        "classes": CLASSES,
        "input_size": INPUT_SIZE,
        "normalize_mean": NORM_MEAN,
        "normalize_std": NORM_STD,
        "val_accuracy": best_acc,
        "val_confusion": best_confusion,
        "class_counts": {CLASSES[k]: v for k, v in class_counts.items()},
        "epochs": args.epochs,
        "seed": args.seed,
    }
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved: {args.output}")
    print(f"saved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
