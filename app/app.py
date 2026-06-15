"""
app.py — Interfaz web de demostración con Streamlit.

Permite cargar un ecocardiograma y visualizar la segmentación
automática de las estructuras cardíacas.

Uso:
    cd app
    streamlit run app.py

O desde la raíz del repo:
    streamlit run app/app.py
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import get_model

# ── Constantes ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Fondo", 1: "Ventrículo Izq. (LV)",
               2: "Miocaridio (MYO)", 3: "Aurícula Izq. (LA)"}

CLASS_COLORS = np.array([
    [0,   0,   0  ],
    [255, 0,   0  ],
    [0,   255, 0  ],
    [0,   0,   255],
], dtype=np.uint8)

IMAGE_SIZE = 256

# ── Página ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación Cardíaca — CAMUS",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Segmentación Automática de Estructuras Cardíacas")
st.markdown(
    "Trabajo Práctico Final — Visión por Computadora II · CEIA FIUBA  \n"
    "Detecta automáticamente el **ventrículo izquierdo**, el **miocaridio** "
    "y la **aurícula izquierda** en ecocardiogramas 2D."
)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    model_choice = st.selectbox(
        "Modelo",
        ["U-Net + ResNet34", "Attention U-Net + EfficientNet-B0"],
        help="Seleccioná el modelo a usar para la predicción"
    )

    default_ckpt = (
        "checkpoints/best_unet_resnet34.pth"
        if model_choice == "U-Net + ResNet34"
        else "checkpoints/best_attention-unet_efficientnet-b0.pth"
    )

    ckpt_path = st.text_input(
        "Ruta al checkpoint (.pth)",
        value=default_ckpt,
        help="Ruta al archivo de pesos guardado durante el entrenamiento"
    )

    st.divider()
    st.markdown(
        "**Leyenda de colores:**  \n"
        "🔴 Ventrículo Izquierdo (LV)  \n"
        "🟢 Miocaridio (MYO)  \n"
        "🔵 Aurícula Izquierda (LA)"
    )

    st.divider()
    st.markdown(
        "**Referencias:**  \n"
        "- Leclerc et al. (2019) — CAMUS dataset  \n"
        "- Ronneberger et al. (2015) — U-Net  \n"
        "- Oktay et al. (2018) — Attention U-Net"
    )


# ── Funciones ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_choice: str, ckpt_path: str):
    """Carga el modelo desde un checkpoint. Se cachea entre interacciones."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_choice == "U-Net + ResNet34":
        model_name, encoder = "unet", "resnet34"
    else:
        model_name, encoder = "attention-unet", "efficientnet-b0"

    model = get_model(model_name, encoder_name=encoder)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        epoch   = ckpt.get("epoch", "?")
        val_dice = ckpt.get("val_dice", "?")
        status  = f"✅ Checkpoint cargado (época {epoch}, val Dice={val_dice:.4f})"
    else:
        status = "⚠️ Checkpoint no encontrado — usando pesos sin entrenar (demo visual)"

    model = model.to(device).eval()
    return model, device, status


def preprocess_image(image: Image.Image) -> torch.Tensor:
    img_gray = image.convert("L")
    img_resized = img_gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min:
        img_array = (img_array - img_min) / (img_max - img_min)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return tensor


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convierte máscara de clases a imagen RGB."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(CLASS_COLORS):
        rgb[mask == class_idx] = color
    return rgb


@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Ejecuta la inferencia y devuelve la máscara de clases."""
    tensor = tensor.to(device)
    logits = model(tensor)
    pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred


def plot_result(img_array: np.ndarray, pred_mask: np.ndarray) -> plt.Figure:
    """Genera la figura de resultado con imagen + overlay de segmentación."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Imagen Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Segmentación pura
    axes[1].imshow(mask_to_rgb(pred_mask))
    axes[1].set_title("Segmentación", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img_array, cmap="gray")
    overlay = mask_to_rgb(pred_mask)
    overlay_masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[2].imshow(mask_to_rgb(pred_mask), alpha=0.45)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Leyenda
    patches = [
        mpatches.Patch(color=CLASS_COLORS[1]/255, label="LV"),
        mpatches.Patch(color=CLASS_COLORS[2]/255, label="MYO"),
        mpatches.Patch(color=CLASS_COLORS[3]/255, label="LA"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    return fig


def compute_pixel_stats(pred_mask: np.ndarray) -> dict:
    """Calcula porcentaje de píxeles por clase."""
    total = pred_mask.size
    stats = {}
    for idx, name in CLASS_NAMES.items():
        count = (pred_mask == idx).sum()
        stats[name] = (count / total) * 100
    return stats


# ── UI principal ──────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 2])

with col_upload:
    st.subheader("📂 Cargar imagen")
    uploaded_file = st.file_uploader(
        "Subí un ecocardiograma",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Formatos soportados: PNG, JPG, BMP, TIFF"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        st.caption(f"Tamaño original: {image.size[0]} × {image.size[1]} px")

    run_button = st.button(
        "🔍 Segmentar",
        disabled=uploaded_file is None,
        use_container_width=True,
        type="primary",
    )

with col_result:
    st.subheader("📊 Resultado")

    if uploaded_file is None:
        st.info("Cargá una imagen para comenzar.")

    elif run_button or "last_result" in st.session_state:

        if run_button:
            # Cargar modelo
            with st.spinner("Cargando modelo..."):
                model, device, status = load_model(model_choice, ckpt_path)
            st.caption(status)

            # Preprocesar e inferir
            with st.spinner("Segmentando..."):
                image    = Image.open(uploaded_file)
                tensor   = preprocess_image(image)
                img_arr  = np.array(tensor.squeeze())  # (H, W)
                pred_mask = predict(model, tensor, device)

            st.session_state["last_result"] = (img_arr, pred_mask)

        img_arr, pred_mask = st.session_state["last_result"]

        # Figura principal
        fig = plot_result(img_arr, pred_mask)
        st.pyplot(fig, use_container_width=True)

        # Estadísticas de píxeles
        st.subheader("📈 Distribución de estructuras")
        stats = compute_pixel_stats(pred_mask)
        cols  = st.columns(4)
        icons = ["⬛", "🔴", "🟢", "🔵"]
        for col, (name, pct), icon in zip(cols, stats.items(), icons):
            col.metric(f"{icon} {name.split('(')[0].strip()}", f"{pct:.1f}%")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "CEIA FIUBA · Visión por Computadora II · 2025 · "
    "Dataset: CAMUS (Leclerc et al., 2019)"
    "</div>",
    unsafe_allow_html=True,
)
