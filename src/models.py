"""
models.py — Modelos de segmentación con encoders preentrenados.

Decisiones de diseño:
- Se usa segmentation-models-pytorch (SMP) en lugar de implementar
  desde cero. Esto nos da acceso a +800 encoders preentrenados y
  arquitecturas validadas en producción.
- Transfer learning desde ImageNet: aunque los ecos son distintos
  a fotos naturales, los features de bajo nivel (bordes, texturas,
  gradientes) son igualmente útiles para detectar bordes del ventrículo.
  Referencia: revisado en Raghu et al. (2019), "Transfusion: Understanding
  Transfer Learning for Medical Imaging", NeurIPS.

Modelo 1 — U-Net + ResNet34:
  El estándar de la industria. Ronneberger et al. (2015) propuso U-Net
  específicamente para imágenes médicas. ResNet34 como encoder ofrece
  buen balance velocidad/accuracy.

Modelo 2 — Attention U-Net + EfficientNet-B0:
  Oktay et al. (2018) demostró que los Attention Gates en el decoder
  mejoran la segmentación de estructuras con bordes difusos, exactamente
  el caso del miocaridio en ecocardiogramas. EfficientNet-B0 es más
  eficiente que ResNet34 con resultados similares o mejores.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ── Constantes ────────────────────────────────────────────────────────────────
NUM_CLASSES = 4   # fondo, LV, MYO, LA
IN_CHANNELS = 1   # escala de grises


def get_unet(
    encoder_name:    str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes:     int = NUM_CLASSES,
    in_channels:     int = IN_CHANNELS,
) -> smp.Unet:
    """
    U-Net con encoder preentrenado.

    SMP adapta automáticamente el primer layer del encoder para aceptar
    `in_channels=1` (grayscale), preservando los pesos ImageNet en las
    capas siguientes.

    Args:
        encoder_name:    Backbone del encoder (ver smp.encoders.get_encoder_names())
        encoder_weights: Pesos preentrenados ('imagenet' u otro)
        num_classes:     Número de clases de salida (4 para CAMUS)
        in_channels:     Canales de entrada (1 para escala de grises)

    Returns:
        Modelo U-Net como nn.Module
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,          # CrossEntropy espera logits, no softmax
        decoder_use_batchnorm=True,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"U-Net ({encoder_name}, ImageNet) — "
        f"{n_params:.1f}M parámetros"
    )
    return model


def get_attention_unet(
    encoder_name:    str = "efficientnet-b0",
    encoder_weights: str = "imagenet",
    num_classes:     int = NUM_CLASSES,
    in_channels:     int = IN_CHANNELS,
) -> smp.Unet:
    """
    Attention U-Net con encoder preentrenado.

    Implementa Attention Gates (Oktay et al., 2018) mediante el parámetro
    decoder_attention_type="scse" de SMP, que aplica atención espacial y
    de canal (Squeeze-and-Excitation) en las skip connections del decoder.

    Los Attention Gates aprenden a suprimir regiones irrelevantes
    (eg. tejido fuera del corazón) y enfocarse en las estructuras
    de interés, mejorando especialmente la segmentación del miocaridio
    (clase más difícil por sus bordes difusos en ultrasonido).

    Args:
        encoder_name:    Backbone del encoder
        encoder_weights: Pesos preentrenados
        num_classes:     Número de clases
        in_channels:     Canales de entrada

    Returns:
        Modelo Attention U-Net como nn.Module
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        decoder_use_batchnorm=True,
        decoder_attention_type="scse",    # <-- activa los attention gates
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"Attention U-Net ({encoder_name}, ImageNet) — "
        f"{n_params:.1f}M parámetros"
    )
    return model


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function — devuelve el modelo por nombre.

    Args:
        model_name: 'unet' o 'attention-unet'
        **kwargs:   Argumentos adicionales para el constructor del modelo

    Returns:
        Modelo como nn.Module
    """
    models = {
        "unet":          get_unet,
        "attention-unet": get_attention_unet,
    }

    if model_name not in models:
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. "
            f"Opciones: {list(models.keys())}"
        )

    return models[model_name](**kwargs)


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Batch de prueba: 4 imágenes grayscale 256x256
    x = torch.randn(4, 1, 256, 256).to(device)

    print("── Modelo 1: U-Net + ResNet34 ──────────────────────")
    m1 = get_unet().to(device)
    out1 = m1(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out1.shape}   (esperado: [4, 4, 256, 256])")

    print("\n── Modelo 2: Attention U-Net + EfficientNet-B0 ─────")
    m2 = get_attention_unet().to(device)
    out2 = m2(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out2.shape}   (esperado: [4, 4, 256, 256])")
