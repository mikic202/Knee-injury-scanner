"""ResNet3D training and inference script for 3D medical images."""

import pickle
from pathlib import Path
import sys
from typing import Callable, Optional, Tuple
from collections import Counter

# Ensure project root is on sys.path for "src" imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import os
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_architecture.resnet3d.resnet import get_resnet3d, FocalLoss
from src.model_training.training_helpers.loggers import WandbLogger


class PCK3DDataset(Dataset):
    """
    Ładuje pliki .pck zawierające numpy array lub dict z kluczem 'image'/'volume'/'img'/'data'.
    Zwraca tensor (C=1, D, H, W) float32 z zakresu [0,1] przeskalowany do target_shape.
    root: katalog z plikami .pck (może zawierać podkatalogi).
    label_fn: funkcja Path -> int (domyślnie z metadata.csv po volumeFilename i label_column).
    target_shape: (D, H, W)
    """
    def __init__(self, root: str, extensions=(".pck",), target_shape: Tuple[int,int,int]=(32,128,128),
                 label_fn: Optional[Callable[[Path], int]] = None,
                 labels_dict: Optional[dict] = None):
        self.root = Path(root)
        self.paths = []
        for ext in extensions:
            self.paths.extend(sorted(self.root.rglob(f"*{ext}")))
        if not self.paths:
            raise FileNotFoundError(f"No files with {extensions} under {root}")
        self.target_shape = target_shape
        # labels_dict: {volumeFilename: label}
        if labels_dict is not None:
            self.label_fn = lambda p: labels_dict.get(p.name, -1)
        else:
            self.label_fn = label_fn or self._infer_label_from_parent

    def _infer_label_from_parent(self, path: Path) -> int:
        parent = path.parent.name.lower()
        if "healthy" in parent or "normal" in parent:
            return 0
        return 1

    def __len__(self) -> int:
        return len(self.paths)

    def _extract_array_from_pickle(self, obj) -> np.ndarray:
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            for key in ("image", "volume", "img", "data", "array"):
                if key in obj:
                    return np.asarray(obj[key])
        # fallback: search for first ndarray in attributes
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                if isinstance(v, np.ndarray):
                    return v
        raise ValueError("No numpy array found in pickle")

    def _load_pickle(self, p: Path) -> np.ndarray:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        arr = self._extract_array_from_pickle(obj)
        arr = np.asarray(arr)
        # normalize intensities to 0-1
        if arr.dtype.kind == "u" or arr.dtype.kind == "i":
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return arr

    def _to_tensor_resized(self, arr: np.ndarray):
        # ensure arr is (D, H, W) or (H,W) -> make (D,H,W)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim == 3:
            # assume (D,H,W) or (H,W,C) ; heuristics: if last dim <=4 treat as channels -> take first channel
            if arr.shape[-1] <= 4 and arr.shape[0] > 4:
                pass
            elif arr.shape[0] <= 4 and arr.shape[-1] > 4:
                arr = arr[0:1]
        else:
            raise ValueError(f"Unsupported ndarray ndim={arr.ndim}")
        D, H, W = arr.shape
        td, th, tw = self.target_shape

        # ensure contiguous float32 numpy array (worker-safe) and convert to tensor
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1,1,D,H,W)

        # resize to target shape
        t = F.interpolate(t, size=(td, th, tw), mode="trilinear", align_corners=False)
        t = t.squeeze(0)  # shape (1,td,th,tw)
        return t  # (C=1, D, H, W)

    def __getitem__(self, idx):
        p = self.paths[idx]
        arr = self._load_pickle(p)
        t = self._to_tensor_resized(arr)  # (1, D, H, W)
        label = int(self.label_fn(p))
        return t, label


def choose_device(preferred: Optional[str] = None) -> torch.device:
    """
    Select device taking into account that torch.cuda.is_available() can be True
    even when the GPU/driver is incompatible at runtime. Attempts a tiny CUDA
    operation and synchronizes; on any error falls back to CPU.
    """
    # if user explicitly requested a device string, try to use it (may raise)
    if preferred:
        # allow values like "cpu", "cuda", "cuda:0"
        try:
            dev = torch.device(preferred)
            if dev.type == "cpu":
                return dev
            # test small CUDA op to ensure runtime compatibility
            t = torch.tensor([1.0], device=dev)
            _ = t * 2.0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return dev
        except Exception as e:
            print(f"Warning: requested device '{preferred}' not usable ({e}), falling back to auto selection.")

    # auto selection: try CUDA if available and verify with a tiny op
    if torch.cuda.is_available():
        try:
            t = torch.tensor([1.0], device="cuda")
            _ = t * 2.0
            torch.cuda.synchronize()
            return torch.device("cuda")
        except Exception as e:
            # common case: incompatible GPU / wrong CUDA build -> fall back to CPU
            print(f"Warning: CUDA detected but not usable ({e}). Falling back to CPU.")
            return torch.device("cpu")
    # otherwise use CPU
    return torch.device("cpu")


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, argmax = preds.max(1)
        all_preds.extend(argmax.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        correct += (argmax == labels).sum().item()
        total += imgs.size(0)
    print(f"  [train] label dist: {Counter(all_labels)}, pred dist: {Counter(all_preds)}, unique preds: {set(all_preds)}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, criterion, device, return_predictions: bool = False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        total_loss += loss.item() * imgs.size(0)
        _, argmax = preds.max(1)
        all_preds.extend(argmax.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        correct += (argmax == labels).sum().item()
        total += imgs.size(0)
    print(f"  [val] label dist: {Counter(all_labels)}, pred dist: {Counter(all_preds)}, unique preds: {set(all_preds)}")
    if return_predictions:
        return total_loss / total, correct / total, all_labels, all_preds
    return total_loss / total, correct / total


def load_model_from_checkpoint(ckpt_path: str, device: torch.device, num_classes: int = 3, in_channels: int = 1):
    """
    Wczytuje weights do modelu i ustawia eval().
    """
    model = get_resnet3d(num_classes=num_classes, in_channels=in_channels, device=device)
    sd = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def _bbox_from_mask(mask: np.ndarray):
    # mask: (D,H,W) boolean or 0-1
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        # nic nie wykryto -> zwróć cały zakres
        D, H, W = mask.shape
        return (0, D-1), (0, H-1), (0, W-1)
    d0, d1 = int(nz[0].min()), int(nz[0].max())
    h0, h1 = int(nz[1].min()), int(nz[1].max())
    w0, w1 = int(nz[2].min()), int(nz[2].max())
    return (d0, d1), (h0, h1), (w0, w1)


def _smooth_cam_3d(cam: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur na CAM dla gładszej wizualizacji."""
    return gaussian_filter(cam, sigma=sigma)


def _cam_metrics(cam: np.ndarray, threshold: float = 0.5) -> dict:
    """Oblicz metryki CAM: średnia intensywność, % voxeli powyżej thresholdu, etc."""
    above_thresh = (cam >= threshold).sum()
    total_voxels = cam.size
    pct_active = 100.0 * above_thresh / total_voxels if total_voxels > 0 else 0.0
    return {
        "mean_intensity": float(cam.mean()),
        "max_intensity": float(cam.max()),
        "std_intensity": float(cam.std()),
        "pct_active_voxels": pct_active,
        "num_active_voxels": int(above_thresh),
    }


def compute_gradcam_3d(model: nn.Module, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                       target_layer: str = "layer4", upsample_size: Tuple[int,int,int] = None, 
                       device: Optional[torch.device] = None, smooth_sigma: float = 1.0):
    """
    Ulepszona 3D-GradCAM z Gaussian smooth i metykami.
    Zwraca: probs (1D np), pred_class (int), cam_np (D,H,W float32), cam_metrics (dict)
    """
    if device is None:
        device = input_tensor.device
    named_modules = dict(model.named_modules())
    module = named_modules.get(target_layer)
    if module is None:
        raise ValueError(f"Module '{target_layer}' not found in model")
    activations = None
    gradients = None

    def forward_hook(mod, inp, out):
        nonlocal activations
        activations = out.clone()

    def backward_hook(mod, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].clone()

    fh = module.register_forward_hook(forward_hook)
    if hasattr(module, "register_full_backward_hook"):
        bh = module.register_full_backward_hook(lambda mod, gi, go: backward_hook(mod, gi, go))
    else:
        bh = module.register_backward_hook(backward_hook)

    model.zero_grad()
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probs_t = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]
    if target_class is None:
        pred = int(probs_t.argmax())
    else:
        pred = int(target_class)

    score = logits[0, pred]
    score.backward(retain_graph=False)

    if activations is None or gradients is None:
        fh.remove(); bh.remove()
        raise RuntimeError("Grad-CAM hooks didn't capture activations/gradients")

    weights = torch.mean(gradients, dim=(2,3,4), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    if upsample_size is None:
        upsample_size = input_tensor.shape[2:]
    cam_up = F.interpolate(cam, size=upsample_size, mode="trilinear", align_corners=False)
    cam_up = cam_up.squeeze(0).squeeze(0)
    cam_np = cam_up.cpu().detach().numpy()
    
    # normalize 0-1
    mn = float(cam_np.min()); mx = float(cam_np.max())
    if mx > mn:
        cam_np = (cam_np - mn) / (mx - mn)
    else:
        cam_np = cam_np * 0.0

    # Smooth CAM z Gaussian blur
    cam_smooth = _smooth_cam_3d(cam_np, sigma=smooth_sigma)
    
    # Oblicz metryki
    metrics = _cam_metrics(cam_smooth, threshold=0.5)

    fh.remove(); bh.remove()
    return probs_t, pred, cam_smooth, metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = "confusion_matrix.png"):
    """Rysuje i zapisuje confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def print_metrics_summary(y_true, y_pred, class_names):
    """Wypisuje szczegółowe metryki: precision, recall, f1-score per klasa."""
    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"\nWeighted Avg Precision: {precision:.4f}")
    print(f"Weighted Avg Recall:    {recall:.4f}")
    print(f"Weighted Avg F1-Score:  {f1:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train / eval / infer ResNet3D on .pck dataset")
    parser.add_argument("--data", help="root folder with .pck files", required=False)
    parser.add_argument("--metadata-csv", type=str, required=True, help="Path to metadata.csv with volumeFilename and label columns")
    parser.add_argument("--label-column", type=str, default="aclDiagnosis", help="Column in metadata.csv to use as label (default: aclDiagnosis)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in FC layer")
    parser.add_argument("--patience", type=int, default=7, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--weighted-loss", action="store_true", default=True, 
                       help="Use weighted CrossEntropyLoss to handle imbalanced classes (default: True)")
    parser.add_argument("--no-weighted-loss", dest="weighted_loss", action="store_false",
                       help="Disable weighted loss")
    parser.add_argument("--focal-loss", action="store_true", default=False,
                       help="Use Focal Loss instead of weighted CrossEntropyLoss (for harder examples)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Gamma parameter for Focal Loss (higher = more focus on hard examples)")
    parser.add_argument("--target-shape", nargs=3, type=int, default=(96,128,128), help="D H W")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not set)")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--infer", type=str, help="path to single .pck file to run inference and exit", default=None)
    parser.add_argument("--no-save", action="store_true", help="don't save checkpoints")
    parser.add_argument("--cam", action="store_true", help="compute Grad-CAM and report lesion bounding box (used with --infer)")
    parser.add_argument("--wandb-project", type=str, default=None, help="Enable Weights & Biases logging with given project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--class-weights", nargs="+", type=float, default=None,
                        help="Optional fixed class weights e.g. --class-weights 0.25 0.93 1.5; if not set, weights are computed from train distribution")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Wczytaj etykiety z metadata.csv ----
    df = pd.read_csv(args.metadata_csv)
    if args.label_column not in df.columns or "volumeFilename" not in df.columns:
        raise ValueError(f"metadata.csv musi mieć kolumny 'volumeFilename' i '{args.label_column}'")
    labels_dict = dict(zip(df["volumeFilename"], df[args.label_column]))
    print(f"[DEBUG] Loaded {len(labels_dict)} labels from {args.metadata_csv} (label column: {args.label_column})")
    # ----

    # Optional W&B logger
    wandb_logger = None
    if args.wandb_project:
        wandb_config = {
            "model": "resnet3d",
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "patience": args.patience,
            "focal_loss": args.focal_loss,
            "focal_gamma": args.focal_gamma,
            "weighted_loss": args.weighted_loss,
            "target_shape": args.target_shape,
        }
        wandb_logger = WandbLogger(project_name=args.wandb_project, run_name=args.wandb_run_name, config=wandb_config)

    if args.infer:
        p = Path(args.infer)
        ds_tmp = PCK3DDataset(str(p.parent), target_shape=tuple(args.target_shape), labels_dict=labels_dict)
        arr = ds_tmp._load_pickle(p)
        t = ds_tmp._to_tensor_resized(arr).unsqueeze(0).to(device)
        ckpt = sorted(Path(args.save_dir).glob("resnet3d_*.pt"))
        model = get_resnet3d(num_classes=3, in_channels=1, device=device)
        if ckpt:
            model = load_model_from_checkpoint(str(ckpt[-1]), device, num_classes=3, in_channels=1)
        model.eval()
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
        print(f"Inference result for {p}: class={pred}, probs={probs.tolist()}")
        if args.cam:
            probs_t, pred_idx, cam_np, metrics = compute_gradcam_3d(model, t, target_class=None, target_layer="layer4",
                                                          upsample_size=tuple(args.target_shape), device=device, smooth_sigma=1.0)
            mask = (cam_np >= 0.5).astype(np.uint8)
            bbox = _bbox_from_mask(mask)
            print(f"Grad-CAM pred={pred_idx}, probs={probs_t.tolist()}")
            print(f"CAM Metrics: mean_intensity={metrics['mean_intensity']:.4f}, max={metrics['max_intensity']:.4f}, "
                  f"std={metrics['std_intensity']:.4f}, pct_active={metrics['pct_active_voxels']:.2f}%, "
                  f"num_active_voxels={metrics['num_active_voxels']}")
            print(f"Lesion bounding box (D_range, H_range, W_range): {bbox}")
            try:
                outp = Path(args.save_dir) / (p.stem + "_cam.npy")
                np.save(str(outp), cam_np)
                print(f"Saved CAM to {outp}")
            except Exception as e:
                print(f"Could not save CAM: {e}")
        if wandb_logger:
            wandb_logger.finish()
        return

    if not args.data:
        raise SystemExit("Provide --data when training or evaluating")

    dataset = PCK3DDataset(args.data, target_shape=tuple(args.target_shape), labels_dict=labels_dict)
    n = len(dataset)
    all_labels = [dataset.label_fn(dataset.paths[i]) for i in range(n)]
    print(f"[DEBUG] Total samples: {n}")
    print(f"[DEBUG] Label distribution (full dataset): {Counter(all_labels)}")
    print(f"[DEBUG] Sample paths (first 5):")
    for i in range(min(5, n)):
        print(f"  {dataset.paths[i]} -> label {all_labels[i]}")

    n_train = int(n * 0.8)
    if n_train < 1 or n - n_train < 1:
        raise SystemExit("Too few samples for train/val split; need >=2")
    
    torch.manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])
    train_indices = train_ds.indices
    val_indices = val_ds.indices
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    print(f"[DEBUG] Train split ({len(train_labels)} samples): {Counter(train_labels)}")
    print(f"[DEBUG] Val split ({len(val_labels)} samples): {Counter(val_labels)}")

    # ---- WEIGHTED LOSS ----
    label_counts = Counter(train_labels)
    total_train = len(train_labels)
    class_weights = None

    if args.class_weights is not None:
        if len(args.class_weights) != 3:
            raise SystemExit("--class-weights must have exactly 3 values (for classes 0,1,2)")
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
        print(f"[DEBUG] Using fixed class weights from CLI: {class_weights.cpu().tolist()}")
    else:
        class_weights_dict = {}
        for class_id in range(3):  # 0, 1, 2
            if class_id in label_counts:
                weight = total_train / (3 * label_counts[class_id])
            else:
                weight = total_train / 3
            class_weights_dict[class_id] = weight
        class_weights = torch.tensor([class_weights_dict[i] for i in range(3)], dtype=torch.float32).to(device)
        print(f"[DEBUG] Class weights for CrossEntropyLoss (computed): {class_weights.cpu().tolist()}")
    print(f"[DEBUG] Train label distribution: {dict(sorted(label_counts.items()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch), shuffle=False, num_workers=max(0, args.num_workers//2), pin_memory=True)

    model = get_resnet3d(num_classes=3, in_channels=1, device=device, dropout_rate=args.dropout)
    
    print("[DEBUG] Checking first 3 batches from train_loader:")
    for batch_idx, (imgs_dbg, labels_dbg) in enumerate(train_loader):
        print(f"  Batch {batch_idx}: imgs shape {imgs_dbg.shape}, labels {labels_dbg.tolist()}, unique {set(labels_dbg.tolist())}")
        if batch_idx >= 2:
            break

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"[INFO] Using Focal Loss (gamma={args.focal_gamma}) with class weights")
    elif args.weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[INFO] Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"[INFO] Using standard CrossEntropyLoss (no class weights)")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if wandb_logger:
            wandb_logger.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": optimizer.param_groups[0]['lr'],
            })
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if not args.no_save:
                best_model_path = Path(args.save_dir) / "resnet3d_best.pt"
                torch.save(model.state_dict(), str(best_model_path))
                print(f"  → Saved best model (val_loss={val_loss:.4f}) to {best_model_path}")
        
        if not args.no_save:
            fname = Path(args.save_dir) / f"resnet3d_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), str(fname))

    # ---- FINAL EVALUATION WITH METRICS ----
    print(f"\n{'='*60}\nBest model from epoch {best_epoch} (val_loss={best_val_loss:.4f})\n{'='*60}")
    
    best_model_path = Path(args.save_dir) / "resnet3d_best.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(str(best_model_path), map_location=device))
        print(f"Loaded best model from {best_model_path}")
    
    val_loss_final, val_acc_final, y_true, y_pred = evaluate(model, val_loader, criterion, device, return_predictions=True)
    print(f"\nFinal Validation Accuracy: {val_acc_final:.4f}")
    
    class_names = [f"Class_{i}" for i in sorted(set(y_true))]
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=str(Path(args.save_dir) / "confusion_matrix.png"))
    print_metrics_summary(y_true, y_pred, class_names)
    
    # ---- GRAD-CAM VISUALIZATION ON SAMPLE IMAGES ----
    print(f"\n{'='*60}\nGenerating Grad-CAM for sample images\n{'='*60}")
    
    samples_to_visualize = []
    for class_id in range(3):
        class_samples = [(i, all_labels[val_indices[i]]) for i in range(len(val_indices)) if all_labels[val_indices[i]] == class_id]
        if class_samples:
            idx = class_samples[0][0]
            sample_path = dataset.paths[val_indices[idx]]
            samples_to_visualize.append((sample_path, class_id))
            print(f"Selected sample for class {class_id}: {sample_path.name}")
    
    if len(samples_to_visualize) < 3:
        print(f"[WARNING] Could not find samples for all classes, taking first 3 from validation set")
        samples_to_visualize = []
        for i in range(min(3, len(val_indices))):
            idx = val_indices[i]
            sample_path = dataset.paths[idx]
            label = all_labels[idx]
            samples_to_visualize.append((sample_path, label))
    
    for sample_idx, (sample_path, true_label) in enumerate(samples_to_visualize):
        print(f"\n[Sample {sample_idx + 1}/3] Processing: {sample_path.name} (true_label={true_label})")
        
        arr = dataset._load_pickle(sample_path)
        t = dataset._to_tensor_resized(arr).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
        
        print(f"  Prediction: class={pred}, probs={[f'{p:.4f}' for p in probs]}")
        print(f"  True label: class={true_label}")
        
        try:
            probs_t, pred_idx, cam_np, metrics = compute_gradcam_3d(
                model, t, target_class=None, target_layer="layer4",
                upsample_size=tuple(args.target_shape), device=device, smooth_sigma=1.0
            )
            
            mask = (cam_np >= 0.5).astype(np.uint8)
            bbox = _bbox_from_mask(mask)
            
            print(f"  CAM Metrics: mean={metrics['mean_intensity']:.4f}, max={metrics['max_intensity']:.4f}, "
                  f"std={metrics['std_intensity']:.4f}, active_voxels={metrics['pct_active_voxels']:.2f}%")
            print(f"  Lesion bbox (D, H, W): {bbox}")
            
            cam_output_path = Path(args.save_dir) / f"{sample_path.stem}_cam.npy"
            np.save(str(cam_output_path), cam_np)
            print(f"  ✓ Saved CAM to {cam_output_path}")
            
        except Exception as e:
            print(f"  ✗ Error generating Grad-CAM: {e}")
    
    if wandb_logger:
        wandb_logger.log({
            "best/epoch": best_epoch,
            "best/val_loss": best_val_loss,
            "final/val_acc": val_acc_final,
            "final/val_loss": val_loss_final,
        })
        wandb_logger.finish()

    print(f"\n{'='*60}\nTraining and evaluation complete!\n{'='*60}")


if __name__ == "__main__":
    main()
