"""
evaluate.py — Evaluación en el conjunto de test y visualización de resultados.

Genera:
  1. Tabla de métricas por clase (Dice e IoU)
  2. Grilla de visualizaciones: imagen | máscara real | predicción
  3. Comparativa entre modelos (si se proveen múltiples checkpoints)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import prepare_dataset, CLASS_NAMES
from models  import get_model
from losses  import CombinedLoss
from metrics import compute_metrics, MetricTracker


# ── Colores para visualización ────────────────────────────────────────────────
# Colores anatómicamente convencionales en cardiología
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0: Fondo — negro
    [255, 0,   0  ],   # 1: LV (ventrículo izquierdo) — rojo
    [0,   255, 0  ],   # 2: MYO (miocaridio) — verde
    [0,   0,   255],   # 3: LA (aurícula izquierda) — azul
], dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convierte máscara de clases (H, W) a imagen RGB (H, W, 3)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(CLASS_COLORS):
        rgb[mask == class_idx] = color
    return rgb


# ── Evaluación en test set ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:       torch.nn.Module,
    loader:      torch.utils.data.DataLoader,
    loss_fn:     torch.nn.Module,
    device:      torch.device,
    model_name:  str = "",
) -> dict:
    """
    Evalúa el modelo en el conjunto de test.

    Returns:
        Diccionario con métricas promedio sobre todo el test set
    """
    model.eval()
    loss_tracker   = 0.0
    metric_tracker = MetricTracker()

    for images, masks in tqdm(loader, desc=f"Evaluando {model_name}", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        loss   = loss_fn(logits, masks)

        loss_tracker += loss.item()
        metric_tracker.update(compute_metrics(logits, masks), n=images.size(0))

    avg_metrics = metric_tracker.get_averages()
    avg_metrics["loss"] = loss_tracker / len(loader)

    return avg_metrics


def print_metrics_table(metrics: dict, model_name: str = "Modelo"):
    """Imprime tabla de métricas formateada."""
    print(f"\n{'=' * 55}")
    print(f"  Resultados en Test Set — {model_name}")
    print(f"{'=' * 55}")
    print(f"  {'Métrica':<25} {'Valor':>10}")
    print(f"  {'-' * 35}")
    print(f"  {'Loss':<25} {metrics['loss']:>10.4f}")
    print(f"  {'-' * 35}")
    print(f"  {'Mean Dice (cardíaco)':<25} {metrics['mean_dice_cardiac']:>10.4f}")
    print(f"  {'Dice LV':<25} {metrics['dice_lv']:>10.4f}")
    print(f"  {'Dice MYO':<25} {metrics['dice_myo']:>10.4f}")
    print(f"  {'Dice LA':<25} {metrics['dice_la']:>10.4f}")
    print(f"  {'-' * 35}")
    print(f"  {'Mean IoU (cardíaco)':<25} {metrics['mean_iou_cardiac']:>10.4f}")
    print(f"  {'IoU LV':<25} {metrics['iou_lv']:>10.4f}")
    print(f"  {'IoU MYO':<25} {metrics['iou_myo']:>10.4f}")
    print(f"  {'IoU LA':<25} {metrics['iou_la']:>10.4f}")
    print(f"{'=' * 55}\n")


# ── Visualización ─────────────────────────────────────────────────────────────
@torch.no_grad()
def visualize_predictions(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    n_samples:  int = 6,
    save_path:  str = None,
    model_name: str = "Modelo",
):
    """
    Genera grilla de visualizaciones:
    [imagen ecográfica | máscara ground truth | predicción del modelo]

    Args:
        model:      Modelo evaluado
        loader:     DataLoader de test
        device:     Dispositivo
        n_samples:  Número de ejemplos a visualizar
        save_path:  Ruta para guardar la figura (None = mostrar)
        model_name: Nombre para el título
    """
    model.eval()
    images_list, masks_list, preds_list = [], [], []

    for images, masks in loader:
        if len(images_list) * images.size(0) >= n_samples:
            break
        logits = model(images.to(device))
        preds  = torch.argmax(logits, dim=1).cpu()
        images_list.append(images.cpu())
        masks_list.append(masks.cpu())
        preds_list.append(preds)

    images_all = torch.cat(images_list)[:n_samples]
    masks_all  = torch.cat(masks_list)[:n_samples]
    preds_all  = torch.cat(preds_list)[:n_samples]

    n = len(images_all)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    fig.suptitle(f"Predicciones — {model_name}", fontsize=14, fontweight="bold", y=1.02)


    col_titles = ["Imagen Ecográfica", "Ground Truth", "Predicción"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=8)

    for i in range(n):
        img  = images_all[i, 0].numpy()          # (H, W) — grayscale
        img_display = (img - img.min()) / (img.max() - img.min() + 1e-8)
        mask = masks_all[i].numpy()               # (H, W)
        pred = preds_all[i].numpy()               # (H, W)

        # Calcular Dice para este ejemplo
        dice_lv  = float((2 * ((pred == 1) & (mask == 1)).sum()) /
                         ((pred == 1).sum() + (mask == 1).sum() + 1e-6))
        dice_myo = float((2 * ((pred == 2) & (mask == 2)).sum()) /
                         ((pred == 2).sum() + (mask == 2).sum() + 1e-6))
        dice_la  = float((2 * ((pred == 3) & (mask == 3)).sum()) /
                         ((pred == 3).sum() + (mask == 3).sum() + 1e-6))

        # Columna 0: imagen ecográfica
        axes[i, 0].imshow(img_display, cmap="gray")
        axes[i, 0].axis("off")

        # Columna 1: ground truth
        axes[i, 1].imshow(img_display, cmap="gray")
        axes[i, 1].imshow(mask_to_rgb(mask), alpha=0.5)
        axes[i, 1].axis("off")

        # Columna 2: predicción + métricas
        axes[i, 2].imshow(img_display, cmap="gray")
        axes[i, 2].imshow(mask_to_rgb(pred), alpha=0.5)
        axes[i, 2].set_xlabel(
            f"LV={dice_lv:.2f}  MYO={dice_myo:.2f}  LA={dice_la:.2f}",
            fontsize=8
        )
        axes[i, 2].axis("off")

    # Leyenda de colores
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[1] / 255, label="LV"),
        mpatches.Patch(color=CLASS_COLORS[2] / 255, label="MYO"),
        mpatches.Patch(color=CLASS_COLORS[3] / 255, label="LA"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualización guardada en: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(
    history:    dict,
    model_name: str = "Modelo",
    save_path:  str = None,
):
    """Grafica las curvas de loss y Dice durante el entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Historial de Entrenamiento — {model_name}", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], "b-",  label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r--", label="Val Loss")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("Loss (Dice + CE)")
    axes[0].set_title("Curva de Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Dice por clase ────────────────────────────────────────────────────────
    axes[1].plot(epochs, history["val_dice_lv"],  "r-",  label="Val LV",  linewidth=2)
    axes[1].plot(epochs, history["val_dice_myo"], "g-",  label="Val MYO", linewidth=2)
    axes[1].plot(epochs, history["val_dice_la"],  "b-",  label="Val LA",  linewidth=2)
    axes[1].plot(epochs, history["val_dice"],     "k--", label="Val Mean", linewidth=2)
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].set_title("Dice por Clase (Validación)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Historial guardado en: {save_path}")
    else:
        plt.show()
    plt.close()


def compare_models(results: dict, save_path: str = None):
    """
    Grafica comparativa de Dice entre modelos.

    Args:
        results: {nombre_modelo: metrics_dict}
        save_path: Ruta para guardar
    """
    model_names = list(results.keys())
    classes     = ["LV", "MYO", "LA", "Mean"]
    dice_keys   = ["dice_lv", "dice_myo", "dice_la", "mean_dice_cardiac"]

    x     = np.arange(len(classes))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, (name, metrics) in enumerate(results.items()):
        values = [metrics[k] for k in dice_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width * 0.9,
                        label=name, color=colors[i % len(colors)], alpha=0.85)

        # Mostrar valores sobre las barras
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    # Línea de referencia (paper CAMUS original)
    reference = [0.94, 0.85, 0.89, 0.89]
    ax.plot(x, reference, "k--o", label="Referencia (Leclerc 2019)",
            linewidth=1.5, markersize=5, zorder=5)

    ax.set_xlabel("Estructura")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("Comparativa de Modelos — Dice por Clase (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparativa guardada en: {save_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo en test set")
    parser.add_argument("--checkpoint", required=True,
                        help="Ruta al checkpoint .pth del mejor modelo")
    parser.add_argument("--data_root",  default="datos_corazon")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_samples",  type=int, default=6,
                        help="Número de ejemplos a visualizar")
    parser.add_argument("--save_dir",   default="resultados")

    args = parser.parse_args()

    # Cargar checkpoint
    print(f"Cargando checkpoint: {args.checkpoint}")
    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = get_model(
        config["model"],
        encoder_name=config["encoder"],
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    model_name = f"{config['model']} + {config['encoder']}"
    print(f"Modelo: {model_name}  (epoch {ckpt['epoch']})")

    # Cargar dataset
    loaders = prepare_dataset(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    loss_fn = CombinedLoss()

    # Evaluar
    metrics = evaluate(model, loaders["test"], loss_fn, device, model_name)
    print_metrics_table(metrics, model_name)

    # Guardar resultados
    os.makedirs(args.save_dir, exist_ok=True)
    tag = config["model"].replace("-", "_") + "_" + config["encoder"]

    # Visualizaciones
    visualize_predictions(
        model, loaders["test"], device,
        n_samples=args.n_samples,
        save_path=os.path.join(args.save_dir, f"predictions_{tag}.png"),
        model_name=model_name,
    )
