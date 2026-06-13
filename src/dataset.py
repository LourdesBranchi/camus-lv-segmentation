"""
dataset.py — CamusDataset con data augmentation médicamente correcto.

Decisiones de diseño:
- NO se usa flip horizontal: invertiría la anatomía cardíaca
  (LV derecho ↔ LV izquierdo), lo que es clínicamente incorrecto.
  Referencia: Shorten & Khoshgoftaar (2019), "A survey on image
  data augmentation for deep learning".
- Rotaciones limitadas a ±10°: el corazón tiene orientación
  anatómica consistente entre pacientes.
- Augmentation solo en train, nunca en val/test.
"""

import os
import glob
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ── Clases del dataset ────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Fondo",
    1: "Ventrículo Izquierdo (LV)",
    2: "Miocaridio (MYO)",
    3: "Aurícula Izquierda (LA)",
}
NUM_CLASSES = 4
IMAGE_SIZE  = 256  # px — tamaño estándar para todos los modelos


# ── Transforms ────────────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """
    Augmentation conservador para imágenes médicas cardíacas.

    Técnicas INCLUIDAS (preservan semántica anatómica):
      - Rotación leve (±10°)
      - Zoom leve (scale 0.9–1.1)
      - Ajuste de brillo/contraste
      - Ruido gaussiano (simula speckle noise del ultrasonido)

    Técnicas EXCLUIDAS:
      - Flip horizontal: invertiría posición anatómica del corazón
      - Flip vertical: ídem
      - Deformaciones elásticas grandes: alteran forma del ventrículo
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Rotate(limit=10, p=0.5),
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, p=1.0),
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_limit=(3.16, 7.07), p=0.3),
        A.Normalize(mean=[0.0], std=[1.0]),   # ya normalizamos en __getitem__
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Sin augmentation — solo resize y conversión a tensor."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.0], std=[1.0]),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class CamusDataset(Dataset):
    """
    Dataset para el CAMUS (Cardiac Acquisitions for Multi-structure
    Ultrasound Segmentation).

    Referencia: Leclerc et al. (2019), "Deep Learning for Segmentation
    using an Open Large-Scale Dataset in 2D Echocardiography", IEEE TMI.

    Args:
        image_paths: Lista de rutas a imágenes .nii.gz
        mask_paths:  Lista de rutas a máscaras .nii.gz (misma longitud)
        transform:   Albumentations transform a aplicar
    """

    def __init__(
        self,
        image_paths: list,
        mask_paths:  list,
        transform:   A.Compose = None,
    ):
        assert len(image_paths) == len(mask_paths), (
            f"Número de imágenes ({len(image_paths)}) y máscaras "
            f"({len(mask_paths)}) no coincide."
        )
        self.image_paths = sorted(image_paths)
        self.mask_paths  = sorted(mask_paths)
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # ── Leer imagen y máscara con SimpleITK ──────────────────────────────
        sitk_img  = sitk.ReadImage(self.image_paths[idx])
        sitk_mask = sitk.ReadImage(self.mask_paths[idx])

        image = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        mask  = sitk.GetArrayFromImage(sitk_mask).astype(np.int64)

        # Asegurar 2D — CAMUS puede tener dimensión de profundidad = 1
        if image.ndim == 3:
            image = image[0]
        if mask.ndim == 3:
            mask = mask[0]

        # ── Normalización min-max a [0, 1] ───────────────────────────────────
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)

        # ── Albumentations espera HxW o HxWxC ────────────────────────────────
        # Como la imagen es grayscale, agregamos canal
        image_3ch = np.stack([image, image, image], axis=-1)  # HxWx3 — float32
        mask_2d   = mask.astype(np.int64)                     # HxW

        if self.transform:
            augmented = self.transform(image=image_3ch, mask=mask_2d)
            image_t   = augmented["image"]   # tensor (3, H, W) por ToTensorV2
            mask_t    = torch.from_numpy(augmented["mask"]).long()
        else:
            image_t = torch.from_numpy(image_3ch.transpose(2, 0, 1)).float()
            mask_t  = torch.from_numpy(mask_2d).long()

        # Quedarnos con 1 canal (escala de grises) para el modelo
        # segmentation-models-pytorch soporta in_channels=1
        image_t = image_t[[0], :, :]   # (1, H, W)

        return image_t, mask_t


# ── Función de preparación del dataset ───────────────────────────────────────
def prepare_dataset(
    data_root:   str,
    image_size:  int   = IMAGE_SIZE,
    batch_size:  int   = 8,
    val_size:    float = 0.15,
    test_size:   float = 0.15,
    random_seed: int   = 42,
    num_workers: int   = 2,
):
    """
    Carga el dataset CAMUS y devuelve DataLoaders listos para entrenar.

    La división se hace a nivel de PACIENTE para evitar data leakage:
    ningún paciente aparece en más de un split.

    Args:
        data_root:  Ruta a la carpeta raíz donde están los .nii.gz
                    (puede ser 'datos_corazon' como en el notebook original)
        image_size: Tamaño al que se redimensionan las imágenes
        batch_size: Tamaño del batch
        val_size:   Fracción del dataset para validación
        test_size:  Fracción del dataset para test
        random_seed: Semilla para reproducibilidad
        num_workers: Workers para el DataLoader

    Returns:
        dict con 'train', 'val', 'test' DataLoaders y estadísticas
    """
    # Sufijos de imágenes que usamos (ED y ES, vistas 2CH y 4CH)
    VALID_SUFFIXES = [
        "_2CH_ED.nii.gz", "_2CH_ES.nii.gz",
        "_4CH_ED.nii.gz", "_4CH_ES.nii.gz",
    ]

    # Buscar todos los archivos de imagen válidos
    all_files = glob.glob(
        os.path.join(data_root, "**/*.nii.gz"), recursive=True
    )
    image_files = [
        f for f in all_files
        if any(os.path.basename(f).endswith(s) for s in VALID_SUFFIXES)
    ]

    if not image_files:
        raise FileNotFoundError(
            f"No se encontraron imágenes .nii.gz en '{data_root}'.\n"
            "Verificá que el dataset esté descomprimido correctamente."
        )

    # Extraer IDs únicos de pacientes
    patient_ids = sorted(set(
        os.path.basename(f).split("_")[0] for f in image_files
    ))
    print(f"Pacientes encontrados: {len(patient_ids)}")
    print(f"Imágenes totales:      {len(image_files)}")

    # División estratificada a nivel de paciente
    train_p, temp_p = train_test_split(
        patient_ids, test_size=(val_size + test_size), random_state=random_seed
    )
    val_p, test_p = train_test_split(
        temp_p,
        test_size=test_size / (val_size + test_size),
        random_state=random_seed,
    )

    print(f"\nDivisión de pacientes:")
    print(f"  Train: {len(train_p)} pacientes")
    print(f"  Val:   {len(val_p)} pacientes")
    print(f"  Test:  {len(test_p)} pacientes")

    # Función auxiliar: filtrar imágenes por lista de pacientes
    def filter_images(patient_list):
        imgs, masks = [], []
        for img_path in image_files:
            pid = os.path.basename(img_path).split("_")[0]
            if pid in set(patient_list):
                mask_path = img_path.replace(".nii.gz", "_gt.nii.gz")
                if os.path.exists(mask_path):
                    imgs.append(img_path)
                    masks.append(mask_path)
        return imgs, masks

    train_imgs, train_masks = filter_images(train_p)
    val_imgs,   val_masks   = filter_images(val_p)
    test_imgs,  test_masks  = filter_images(test_p)

    print(f"\nPares imagen/máscara:")
    print(f"  Train: {len(train_imgs)}")
    print(f"  Val:   {len(val_imgs)}")
    print(f"  Test:  {len(test_imgs)}")

    # Crear datasets
    train_ds = CamusDataset(train_imgs, train_masks,
                            transform=get_train_transforms(image_size))
    val_ds   = CamusDataset(val_imgs,   val_masks,
                            transform=get_val_transforms(image_size))
    test_ds  = CamusDataset(test_imgs,  test_masks,
                            transform=get_val_transforms(image_size))

    # Crear DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "n_train": len(train_ds),
        "n_val":   len(val_ds),
        "n_test":  len(test_ds),
    }


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "datos_corazon"
    loaders = prepare_dataset(data_root, batch_size=4)

    imgs, masks = next(iter(loaders["train"]))
    print(f"\nBatch de prueba:")
    print(f"  Imágenes: {imgs.shape}  dtype={imgs.dtype}  "
          f"min={imgs.min():.3f}  max={imgs.max():.3f}")
    print(f"  Máscaras: {masks.shape}  dtype={masks.dtype}  "
          f"clases={masks.unique().tolist()}")
