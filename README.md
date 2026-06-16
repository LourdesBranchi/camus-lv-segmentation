# Segmentación Automática del Ventrículo Izquierdo en Ecocardiogramas 2D

Trabajo Práctico Final - Visión por Computadora II  
Carrera de Especialización en Inteligencia Artificial (CEIA) - FIUBA

## Descripción

Este proyecto implementa y compara dos arquitecturas de deep learning para la segmentación semántica automática de estructuras cardíacas (ventrículo izquierdo, miocaridio y aurícula izquierda) en ecocardiogramas 2D del dataset público **CAMUS**.

Se utiliza transfer learning con encoders preentrenados en ImageNet, combinado con una loss function Dice + CrossEntropy para manejar el desbalance de clases inherente a imágenes médicas.

### Modelos comparados

| Modelo | Encoder | Params aprox. |
|--------|---------|---------------|
| U-Net | ResNet-34 (pre entrenado en ImageNet) | ~24M |
| Attention U-Net | EfficientNet-B0 (pre entrenado en ImageNet) | ~6M |

### Métricas reportadas

- **Dice Coefficient** por clase (LV, MYO, LA) - métrica principal
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

## Estructura del repositorio

```
camus-lv-segmentation/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py            # CamusDataset + data augmentation
│   ├── models.py             # U-Net y Attention U-Net con encoders preentrenados
│   ├── losses.py             # Dice Loss + CrossEntropy combinada
│   ├── metrics.py            # Dice e IoU por clase
│   ├── train.py              # Loop de entrenamiento
│   └── evaluate.py           # Evaluación y visualización de resultados
├── notebooks/
│   ├── 01_EDA.ipynb          # Análisis exploratorio del dataset
│   └── 02_train_colab.ipynb  # Entrenamiento en Google Colab
└── app/
    └── app.py                # Interfaz web con Streamlit
```

---

## Cómo correr los notebooks en Google Colab

### Paso previo — Configurar el Secret de Kaggle

Los notebooks descargan el dataset automáticamente desde Kaggle. Para esto necesitás configurar tu API key **una sola vez**:

1. Entrá a [kaggle.com](https://www.kaggle.com) → tu perfil → **Settings** → **API** → **Create New Token**
2. Se genera la key con este formato:
```json
{"username":"tu_usuario","key":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
```
3. En Google Colab, abrí el panel de **Secrets** (ícono 🔑 en el panel izquierdo)
4. Creá un nuevo secret con:
   - **Name:** `KAGGLE_API_TOKEN`
   - **Value:** el contenido completo otorgado por kaggle
5. Activá el toggle **"Notebook access"** para habilitarlo en el notebook actual

> ⚠️ Este secret hay que habilitarlo cada vez que abrís un notebook nuevo en Colab.

---

### Notebook 1 — Análisis Exploratorio (EDA)

**Propósito:** Explorar el dataset CAMUS antes de entrenar. Genera visualizaciones de distribución de clases, tamaños de imagen, ejemplos de segmentaciones y distribución de intensidades.

**Pasos:**

1. Abrí el notebook en Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LourdesBranchi/camus-lv-segmentation/blob/main/notebooks/01_EDA.ipynb)

2. **No requiere GPU** — podés dejarlo en CPU (Runtime → Change runtime type → None)

3. Verificá que el Secret de Kaggle esté habilitado para este notebook

4. Corré todas las celdas en orden

5. Las figuras generadas se guardan automáticamente en:
```
Google Drive → Mi unidad → camus_eda/
```

---

### Notebook 2 — Entrenamiento en Google Colab

**Propósito:** Entrena los dos modelos de segmentación y genera visualizaciones comparativas.

**Pasos:**

1. Abrí el notebook en Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LourdesBranchi/camus-lv-segmentation/blob/main/notebooks/02_train_colab.ipynb)

2. **Cambiar a GPU** (obligatorio): Elegir **T4 GPU** en el entorno de ejecución.

3. Verificá que el Secret de Kaggle esté habilitado para este notebook.

4. **Configurar la carpeta de Drive** donde se guardan los checkpoints. En la celda de setup de Drive, verificá que `DRIVE_DIR` apunte a la carpeta que querés:
```python
DRIVE_DIR = '/content/drive/MyDrive/camus_checkpoints'
```
Podés cambiarlo a cualquier carpeta de tu Drive.

5. Corré todas las celdas en orden. El entrenamiento completo de ambos modelos tarda aproximadamente **90 minutos** en una GPU T4.

6. Los checkpoints, histories y visualizaciones se guardan automáticamente en tu Google Drive al terminar.

> ⚠️ **Importante:** No cerrés la pestaña del navegador durante el entrenamiento. Si la sesión se desconecta, los checkpoints ya guardados en Drive se preservan pero tendrás que re-ejecutar desde la celda de entrenamiento.

---

## Cómo correr la interfaz web (Streamlit)

La interfaz permite cargar un ecocardiograma y visualizar la segmentación automática en tiempo real.

### Requisitos previos

1. Clonar el repo y instalar dependencias:
```bash
git clone https://github.com/LourdesBranchi/camus-lv-segmentation.git
cd camus-lv-segmentation
pip install -r requirements.txt
```

2. Descargar los checkpoints entrenados desde Google Drive y colocarlos en una carpeta llamada checkpoints:
```
camus-lv-segmentation/
└── checkpoints/
    ├── best_unet_resnet34.pth
    └── best_attention-unet_efficientnet-b0.pth
```

### Correr la app

```bash
streamlit run app/app.py
```

Se abre automáticamente en el navegador en `http://localhost:8501`.

### Uso

1. En el **sidebar izquierdo**, seleccioná el modelo a usar
2. Verificá que la ruta al checkpoint sea correcta
3. En el panel principal, subí una imagen de ecocardiograma (PNG, JPG, TIFF)
4. Hacé click en **Segmentar**
5. El resultado muestra la imagen original, la segmentación y el overlay con las estructuras detectadas

> **Imágenes de prueba:** podés usar cualquier ecocardiograma 2D en escala de grises, pero teniendo en cuenta que el modelo fue entrenado con vistas apical de 2 y 4 cámaras del dataset CAMUS. De todas formas, el repositorio incluye 6 ecocardiogramas de ejemplo en la carpeta `sample_images/`, extraídos del conjunto de test del dataset CAMUS. Estas imagenes pueden ser usadas directamente para probar la interfaz sin necesidad de conseguir imágenes propias.
---

## Resultados

| Modelo | Dice LV | Dice MYO | Dice LA | Mean Dice |
|--------|---------|----------|---------|-----------|
| U-Net + ResNet34 | 0.940 | 0.877 | 0.908 | 0.908 |
| Attention U-Net + EfficientNet-B0 | 0.937 | 0.870 | 0.905 | 0.904 |

---

## Referencias

1. Leclerc, S. et al. (2019). *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography*. IEEE TMI. https://doi.org/10.1109/TMI.2019.2900516

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. https://arxiv.org/abs/1505.04597

3. Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL. https://arxiv.org/abs/1804.03999

4. Taghanaki, S.A. et al. (2019). *Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation*. CMIG. https://arxiv.org/abs/1805.02798

5. Yakubovskiy, P. (2020). *Segmentation Models PyTorch*. GitHub. https://github.com/qubvel-org/segmentation_models.pytorch

---

## Autores

- Lourdes Gonzalez Branchi — lourdes.gbranchi@gmail.com
- Alberto Adriano Levano — aadrianolevano@gmail.com
- Ignacio Vollono — ivollonoc@gmail.com

CEIA — FIUBA, 2025
