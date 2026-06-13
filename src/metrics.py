"""
metrics.py — Métricas de evaluación: Dice e IoU por clase.

Las métricas estándar para segmentación semántica médica son:
  - Dice Similarity Coefficient (DSC): mide el solapamiento entre
    la predicción y el ground truth. Rango [0, 1], 1 = perfecto.
  - Intersection over Union (IoU / Jaccard Index): similar al Dice
    pero más estricto. Relación: IoU = Dice / (2 - Dice).

Se reportan por clase (LV, MYO, LA) — no incluimos el fondo
en el promedio porque no es una estructura clínica de interés.

Referencia: Leclerc et al. (2019) usa exactamente estas métricas
para el benchmark de CAMUS. Los scores de referencia son:
  LV:  Dice ~0.94
  MYO: Dice ~0.85
  LA:  Dice ~0.89
"""

import torch
import numpy as np
from typing import Dict, List


CLASS_NAMES = ["Fondo", "LV", "MYO", "LA"]
CARDIAC_CLASSES = [1, 2, 3]   # excluimos fondo del promedio


def dice_coefficient(
    pred:    torch.Tensor,
    target:  torch.Tensor,
    class_idx: int,
    smooth:  float = 1e-6,
) -> float:
    """
    Dice Coefficient para una clase específica.

    DSC = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred:      Predicciones (B, H, W) — clase por píxel (argmax aplicado)
        target:    Ground truth (B, H, W) — clase por píxel
        class_idx: Índice de la clase a evaluar
        smooth:    Factor para evitar división por cero

    Returns:
        Dice score como float en [0, 1]
    """
    pred_bin   = (pred == class_idx).float()
    target_bin = (target == class_idx).float()

    intersection = (pred_bin * target_bin).sum()
    union        = pred_bin.sum() + target_bin.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    class_idx: int,
    smooth:    float = 1e-6,
) -> float:
    """
    IoU (Intersection over Union / Jaccard Index) para una clase.

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred:      Predicciones (B, H, W)
        target:    Ground truth (B, H, W)
        class_idx: Índice de la clase
        smooth:    Factor para evitar división por cero

    Returns:
        IoU score como float en [0, 1]
    """
    pred_bin   = (pred == class_idx).float()
    target_bin = (target == class_idx).float()

    intersection = (pred_bin * target_bin).sum()
    union        = pred_bin.sum() + target_bin.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_metrics(
    logits:      torch.Tensor,
    masks:       torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Calcula Dice e IoU para todas las clases y el promedio cardíaco.

    Args:
        logits: Salida del modelo (B, C, H, W) — sin softmax
        masks:  Ground truth (B, H, W) — clases enteras
        num_classes: Número de clases

    Returns:
        Diccionario con métricas:
          - dice_lv, dice_myo, dice_la, mean_dice_cardiac
          - iou_lv,  iou_myo,  iou_la,  mean_iou_cardiac
    """
    with torch.no_grad():
        # Convertir logits a predicciones (argmax sobre canales)
        pred = torch.argmax(logits, dim=1)   # (B, H, W)

        # Mover a CPU para cálculo
        pred   = pred.cpu()
        target = masks.cpu()

        metrics = {}

        # Métricas por clase cardíaca (excluimos fondo = clase 0)
        class_map = {1: "lv", 2: "myo", 3: "la"}

        dice_cardiac = []
        iou_cardiac  = []

        for class_idx, class_name in class_map.items():
            dice = dice_coefficient(pred, target, class_idx)
            iou  = iou_score(pred, target, class_idx)

            metrics[f"dice_{class_name}"] = dice
            metrics[f"iou_{class_name}"]  = iou

            dice_cardiac.append(dice)
            iou_cardiac.append(iou)

        # Promedios cardíacos (LV + MYO + LA)
        metrics["mean_dice_cardiac"] = float(np.mean(dice_cardiac))
        metrics["mean_iou_cardiac"]  = float(np.mean(iou_cardiac))

    return metrics


class MetricTracker:
    """
    Acumula métricas a lo largo de una época y calcula promedios.

    Uso:
        tracker = MetricTracker()
        for batch in loader:
            logits, masks = ...
            metrics = compute_metrics(logits, masks)
            tracker.update(metrics)

        epoch_metrics = tracker.get_averages()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Agrega las métricas de un batch.

        Args:
            metrics: Diccionario de métricas del batch
            n:       Tamaño del batch (para promedio ponderado)
        """
        for key, value in metrics.items():
            if key not in self._sums:
                self._sums[key]   = 0.0
                self._counts[key] = 0
            self._sums[key]   += value * n
            self._counts[key] += n

    def get_averages(self) -> Dict[str, float]:
        """Devuelve el promedio de cada métrica acumulada."""
        return {
            key: self._sums[key] / self._counts[key]
            for key in self._sums
            if self._counts[key] > 0
        }

    def summary_string(self, prefix: str = "") -> str:
        """Genera un string legible con las métricas."""
        avgs = self.get_averages()
        parts = []
        for key in ["mean_dice_cardiac", "dice_lv", "dice_myo", "dice_la",
                    "mean_iou_cardiac"]:
            if key in avgs:
                parts.append(f"{key}: {avgs[key]:.4f}")
        return (prefix + " | " if prefix else "") + " | ".join(parts)


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_CLASSES = 4
    B, H, W = 4, 256, 256

    # Simular predicción perfecta (logits que generan el ground truth)
    target = torch.randint(0, NUM_CLASSES, (B, H, W))
    # Logits perfectos: clase correcta tiene score muy alto
    logits = torch.zeros(B, NUM_CLASSES, H, W)
    for c in range(NUM_CLASSES):
        logits[:, c, :, :] = (target == c).float() * 10.0

    metrics = compute_metrics(logits, target)
    print("Métricas con predicción perfecta (esperado ~1.0):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Simular predicción aleatoria
    print("\nMétricas con predicción aleatoria:")
    logits_random = torch.randn(B, NUM_CLASSES, H, W)
    metrics_random = compute_metrics(logits_random, target)
    for k, v in metrics_random.items():
        print(f"  {k}: {v:.4f}")
