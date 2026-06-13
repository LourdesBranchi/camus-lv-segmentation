"""
losses.py — Loss function combinada Dice + CrossEntropy.

Motivación:
  El dataset CAMUS tiene un desbalance de clases severo:
    - Fondo:     77.07% de los píxeles
    - LV:         8.54%
    - Miocaridio: 9.29%
    - Aurícula:   5.10%

  Con CrossEntropy sola, el modelo puede lograr una loss baja
  simplemente prediciendo todo como fondo, sin aprender a segmentar
  ninguna estructura cardíaca.

  Solución: combinar Dice Loss + CrossEntropy.

  - Dice Loss: mide el solapamiento entre predicción y ground truth,
    independientemente del ratio foreground/background. No le importa
    cuántos píxeles son fondo.
  - CrossEntropy: asegura que el entrenamiento sea estable y que
    el modelo aprenda la distribución completa de clases.

  Referencias:
    - Taghanaki et al. (2019), "Combo Loss: Handling Input and Output
      Imbalance in Multi-Organ Segmentation", CMIG. Validado
      específicamente en ecocardiografía con segmentación de LV.
    - Milletari et al. (2016), "V-Net: Fully Convolutional Neural
      Networks for Volumetric Medical Image Segmentation". Introduce
      Dice Loss para segmentación médica.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Loss combinada: alpha * DiceLoss + (1 - alpha) * CrossEntropyLoss

    SMP ya incluye implementaciones optimizadas de ambas losses,
    así que las usamos directamente.

    Args:
        alpha:       Peso de Dice Loss (default: 0.5 — igual peso)
        smooth:      Factor de suavizado para Dice (evita división por cero)
        ignore_index: Clase a ignorar (-1 = ninguna)
    """

    def __init__(
        self,
        alpha:        float = 0.5,
        smooth:       float = 1e-6,
        ignore_index: int   = -1,
    ):
        super().__init__()
        self.alpha = alpha

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            smooth=smooth,
            ignore_index=ignore_index if ignore_index >= 0 else None,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,  # (B, C, H, W) — salida del modelo sin softmax
        masks:  torch.Tensor,  # (B, H, W)    — clases enteras
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicciones del modelo (sin softmax)
            masks:  Máscaras ground truth con valores 0..NUM_CLASSES-1

        Returns:
            Loss escalar combinada
        """
        dice = self.dice_loss(logits, masks)
        ce   = self.ce_loss(logits, masks)

        combined = self.alpha * dice + (1 - self.alpha) * ce
        return combined

    def __repr__(self) -> str:
        return (
            f"CombinedLoss(alpha={self.alpha}, "
            f"dice={self.alpha:.2f} * DiceLoss + "
            f"{1-self.alpha:.2f} * CrossEntropyLoss)"
        )


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_CLASSES = 4
    B, H, W = 4, 256, 256

    logits = torch.randn(B, NUM_CLASSES, H, W)
    masks  = torch.randint(0, NUM_CLASSES, (B, H, W))

    loss_fn = CombinedLoss(alpha=0.5)
    print(loss_fn)

    loss = loss_fn(logits, masks)
    print(f"\nLoss de prueba: {loss.item():.4f}")
    print("✓ Loss calculada correctamente")
