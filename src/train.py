"""
train.py — Loop de entrenamiento completo.

Incluye:
  - Entrenamiento y validación por época
  - ReduceLROnPlateau scheduler (reduce LR cuando val Dice no mejora)
  - Checkpointing del mejor modelo (según mean_dice_cardiac en val)
  - Guardado en Google Drive si se detecta Colab
  - Early stopping para evitar overfitting
  - Logging claro del progreso
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Agregar src al path si se ejecuta desde la raíz del repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import prepare_dataset
from models  import get_model
from losses  import CombinedLoss
from metrics import compute_metrics, MetricTracker


# ── Configuración por defecto ─────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_root":    "datos_corazon",
    "model":        "unet",
    "encoder":      "resnet34",
    "epochs":       50,
    "batch_size":   8,
    "lr":           1e-4,
    "image_size":   256,
    "alpha":        0.5,       # peso Dice en la loss combinada
    "patience":     10,        # early stopping patience
    "num_workers":  2,
    "save_dir":     "checkpoints",
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """Para el entrenamiento si la métrica no mejora en `patience` épocas."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter    = 0
        return self.should_stop


# ── Una época de entrenamiento ────────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    torch.device,
) -> dict:
    """Entrena el modelo por una época y devuelve métricas."""
    model.train()
    loss_tracker   = 0.0
    metric_tracker = MetricTracker()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        loss_tracker += loss.item()

        # Calcular métricas en CPU sin gradientes
        metrics = compute_metrics(logits.detach(), masks)
        metric_tracker.update(metrics, n=images.size(0))

    avg_loss    = loss_tracker / len(loader)
    avg_metrics = metric_tracker.get_averages()
    avg_metrics["loss"] = avg_loss

    return avg_metrics


# ── Una época de validación ───────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device:  torch.device,
) -> dict:
    """Evalúa el modelo en validación."""
    model.eval()
    loss_tracker   = 0.0
    metric_tracker = MetricTracker()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        loss   = loss_fn(logits, masks)

        loss_tracker += loss.item()
        metrics = compute_metrics(logits, masks)
        metric_tracker.update(metrics, n=images.size(0))

    avg_loss    = loss_tracker / len(loader)
    avg_metrics = metric_tracker.get_averages()
    avg_metrics["loss"] = avg_loss

    return avg_metrics


# ── Loop principal ────────────────────────────────────────────────────────────
def train(config: dict) -> dict:
    """
    Entrena el modelo con la configuración dada.

    Args:
        config: Diccionario de configuración (ver DEFAULT_CONFIG)

    Returns:
        Diccionario con historial de métricas de train y val
    """
    print("=" * 60)
    print(f"Modelo:   {config['model']} + {config['encoder']}")
    print(f"Device:   {config['device']}")
    print(f"Épocas:   {config['epochs']}")
    print(f"Batch:    {config['batch_size']}")
    print(f"LR:       {config['lr']}")
    print("=" * 60)

    device = torch.device(config["device"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nCargando dataset...")
    loaders = prepare_dataset(
        data_root=config["data_root"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("\nCreando modelo...")
    model = get_model(
        config["model"],
        encoder_name=config["encoder"],
    ).to(device)

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    loss_fn   = CombinedLoss(alpha=config["alpha"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",        # maximizar Dice
        factor=0.5,        # reducir LR a la mitad
        patience=5,        # si no mejora en 5 épocas
    )
    early_stopping = EarlyStopping(patience=config["patience"])

    # ── Directorio de checkpoints ─────────────────────────────────────────────
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    model_tag     = f"{config['model']}_{config['encoder']}"
    best_ckpt     = os.path.join(save_dir, f"best_{model_tag}.pth")
    last_ckpt     = os.path.join(save_dir, f"last_{model_tag}.pth")

    # ── Historial ─────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_dice_lv": [], "val_dice_lv": [],
        "train_dice_myo": [], "val_dice_myo": [],
        "train_dice_la": [], "val_dice_la": [],
        "train_iou": [], "val_iou": [],
    }
    best_val_dice = 0.0

    print(f"\nEntrenando por {config['epochs']} épocas...\n")

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, loss_fn, device
        )
        val_metrics = validate_one_epoch(
            model, loaders["val"], loss_fn, device
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # Guardar historial
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_metrics["mean_dice_cardiac"])
        history["val_dice"].append(val_metrics["mean_dice_cardiac"])
        history["train_dice_lv"].append(train_metrics["dice_lv"])
        history["val_dice_lv"].append(val_metrics["dice_lv"])
        history["train_dice_myo"].append(train_metrics["dice_myo"])
        history["val_dice_myo"].append(val_metrics["dice_myo"])
        history["train_dice_la"].append(train_metrics["dice_la"])
        history["val_dice_la"].append(val_metrics["dice_la"])
        history["train_iou"].append(train_metrics["mean_iou_cardiac"])
        history["val_iou"].append(val_metrics["mean_iou_cardiac"])

        # Log de la época
        print(
            f"Época [{epoch:3d}/{config['epochs']}] ({elapsed:.0f}s) "
            f"lr={lr_now:.2e}\n"
            f"  Train  loss={train_metrics['loss']:.4f}  "
            f"dice={train_metrics['mean_dice_cardiac']:.4f}  "
            f"[LV={train_metrics['dice_lv']:.3f} "
            f"MYO={train_metrics['dice_myo']:.3f} "
            f"LA={train_metrics['dice_la']:.3f}]\n"
            f"  Val    loss={val_metrics['loss']:.4f}  "
            f"dice={val_metrics['mean_dice_cardiac']:.4f}  "
            f"[LV={val_metrics['dice_lv']:.3f} "
            f"MYO={val_metrics['dice_myo']:.3f} "
            f"LA={val_metrics['dice_la']:.3f}]"
        )

        # Guardar mejor modelo
        val_dice = val_metrics["mean_dice_cardiac"]
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_dice":     best_val_dice,
                "config":       config,
            }, best_ckpt)
            print(f"  ✓ Mejor modelo guardado (val_dice={best_val_dice:.4f})")

        # Checkpoint del último epoch
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "history":      history,
            "config":       config,
        }, last_ckpt)

        # Scheduler y early stopping
        scheduler.step(val_dice)
        if early_stopping(val_dice):
            print(f"\nEarly stopping en época {epoch}.")
            break

        print()

    print("=" * 60)
    print(f"Entrenamiento finalizado.")
    print(f"Mejor val Dice: {best_val_dice:.4f}")
    print(f"Checkpoint guardado en: {best_ckpt}")
    print("=" * 60)

    history["best_val_dice"] = best_val_dice
    history["best_ckpt"]     = best_ckpt

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo de segmentación CAMUS")
    parser.add_argument("--data_root",   default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--model",       default=DEFAULT_CONFIG["model"],
                        choices=["unet", "attention-unet"])
    parser.add_argument("--encoder",     default=DEFAULT_CONFIG["encoder"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--image_size",  type=int,   default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--alpha",       type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--num_workers", type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--save_dir",    default=DEFAULT_CONFIG["save_dir"])
    parser.add_argument("--device",      default=DEFAULT_CONFIG["device"])

    args   = parser.parse_args()
    config = vars(args)

    history = train(config)
