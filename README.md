# Segmentación Automática del Ventrículo Izquierdo en Ecocardiogramas 2D

Trabajo Práctico Final — Visión por Computadora II  
Carrera de Especialización en Inteligencia Artificial (CEIA) — FIUBA

## Descripción

Este proyecto implementa y compara dos arquitecturas de deep learning para la segmentación semántica automática de estructuras cardíacas (ventrículo izquierdo, miocaridio y aurícula izquierda) en ecocardiogramas 2D del dataset público **CAMUS**.

Se utiliza transfer learning con encoders preentrenados en ImageNet, combinado con una loss function Dice + CrossEntropy para manejar el desbalance de clases inherente a imágenes médicas.

### Modelos comparados

| Modelo | Encoder | Params aprox. |
|--------|---------|---------------|
| U-Net | ResNet-34 (ImageNet) | ~24M |
| Attention U-Net | EfficientNet-B0 (ImageNet) | ~6M |

### Métricas reportadas

- **Dice Coefficient** por clase (LV, MYO, LA) — métrica principal
- **IoU (Intersection over Union)** por clase
- **Mean Dice** sobre las 3 estructuras cardíacas

---

## Dataset

**CAMUS** (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)  
Leclerc et al., 2019 — [Paper](https://doi.org/10.1109/TMI.2019.2900516)

- 500 pacientes, vistas apical 2 y 4 cámaras
- 2000 pares imagen/máscara (ED y ES por paciente)
- 4 clases: fondo (0), ventrículo izquierdo (1), miocaridio (2), aurícula izquierda (3)
- Descarga: [Kaggle — CAMUS Human Heart Data](https://www.kaggle.com/datasets/shoybhasan/camus-human-heart-data)

### División del dataset

| Split | Pacientes | Imágenes |
|-------|-----------|----------|
| Train | 350 (70%) | 1400 |
| Val   | 75 (15%)  | 300 |
| Test  | 75 (15%)  | 300 |

---

## Instalación

### Requisitos

- Python 3.9+
- CUDA 11.8+ (para entrenamiento local con GPU)
- O Google Colab con GPU T4 (recomendado)

### Setup local

```bash
git clone https://github.com/TU_USUARIO/camus-lv-segmentation.git
cd camus-lv-segmentation
pip install -r requirements.txt
```

### Setup en Google Colab

Abrir directamente el notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/camus-lv-segmentation/blob/main/notebooks/02_train_colab.ipynb)

En el notebook, el setup del repo se hace automáticamente:

```python
!git clone https://github.com/TU_USUARIO/camus-lv-segmentation.git
%cd camus-lv-segmentation
!pip install -r requirements.txt
```

---

## Uso

### 1. Análisis exploratorio (EDA)

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 2. Entrenamiento

Se recomienda entrenar desde Google Colab usando `notebooks/02_train_colab.ipynb`.

Para entrenamiento local (requiere GPU):

```bash
python src/train.py --model unet --encoder resnet34 --epochs 50
python src/train.py --model attention-unet --encoder efficientnet-b0 --epochs 50
```

### 3. Evaluación

```bash
python src/evaluate.py --checkpoint checkpoints/best_unet.pth
```

### 4. Demo interactiva

```bash
cd app
streamlit run app.py
```

---

## Estructura del repositorio

```
camus-lv-segmentation/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py       # CamusDataset + data augmentation
│   ├── models.py        # U-Net y Attention U-Net con encoders preentrenados
│   ├── losses.py        # Dice Loss + CrossEntropy combinada
│   ├── metrics.py       # Dice e IoU por clase
│   ├── train.py         # Loop de entrenamiento
│   ├── evaluate.py      # Evaluación y visualización de resultados
│   └── utils.py         # Helpers y utilidades
├── notebooks/
│   ├── 01_EDA.ipynb     # Análisis exploratorio del dataset
│   └── 02_train_colab.ipynb  # Entrenamiento en Google Colab
└── app/
    └── app.py           # Interfaz web con Streamlit
```

---

## Resultados

*Se completarán tras el entrenamiento.*

| Modelo | Dice LV | Dice MYO | Dice LA | Mean Dice |
|--------|---------|----------|---------|-----------|
| U-Net + ResNet34 | — | — | — | — |
| Attention U-Net + EfficientNet-B0 | — | — | — | — |
| Referencia (Leclerc et al., 2019) | 0.94 | 0.85 | 0.89 | 0.89 |

---

## Referencias

1. Leclerc, S. et al. (2019). *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography*. IEEE TMI. https://doi.org/10.1109/TMI.2019.2900516

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. https://arxiv.org/abs/1505.04597

3. Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL. https://arxiv.org/abs/1804.03999

4. Ullah, Z. et al. (2025). *Unified Review and Benchmark of Deep Segmentation Architectures for Cardiac Ultrasound on CAMUS*. arXiv:2601.00839.

5. Taghanaki, S.A. et al. (2019). *Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation*. CMIG. https://arxiv.org/abs/1805.02798

6. Yakubovskiy, P. (2020). *Segmentation Models PyTorch*. GitHub. https://github.com/qubvel-org/segmentation_models.pytorch

---

## Autores

- Lourdes Gonzalez Branchi
- [Nombre compañero/a]
- [Nombre compañero/a]

CEIA — FIUBA, 2025
