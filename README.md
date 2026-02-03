# Laboratorio 3 — Transformación Digital

## Redes convolucionales: sesgo inductivo y diseño de arquitectura

En este curso las redes neuronales no se tratan como cajas negras, sino como componentes arquitectónicos cuyas decisiones de diseño afectan el rendimiento, la escalabilidad y la interpretabilidad. Este laboratorio se centra en las **capas convolucionales** como ejemplo de cómo se introduce sesgo inductivo en sistemas de aprendizaje.

Cada estudiante elige, analiza y experimenta con una arquitectura convolucional usando un dataset real.

---

## Objetivos de aprendizaje

- Entender el papel y la intuición matemática de las capas convolucionales.
- Analizar cómo las decisiones arquitectónicas (tamaño de kernel, profundidad, stride, padding) afectan el aprendizaje.
- Comparar capas convolucionales con capas totalmente conectadas para datos tipo imagen.
- Realizar un EDA mínimo pero significativo para tareas de redes neuronales.
- Comunicar con claridad las decisiones arquitectónicas y experimentales.

---

## Dataset elegido: Fashion-MNIST

- **Origen**: [TensorFlow/Keras](https://www.tensorflow.org/datasets/catalog/fashion_mnist) (también disponible en PyTorch `torchvision.datasets.FashionMNIST`).
- **Contenido**: 70 000 imágenes en escala de grises (60k train, 10k test), 28×28 píxeles, **10 clases** (prendas y calzado: T-shirt, pantalón, pullover, vestido, abrigo, sandalia, camisa, zapatilla, bolso, bota).
- **Por qué es adecuado para convoluciones**: estructura espacial 2D, invariancia traslacional, tamaño manejable en memoria y clases balanceadas; permite evaluar diseño de CNN sin necesidad de recursos pesados.

---

## Estructura del proyecto

```
Laboratorio-3-Transformacion-Digital/
├── README.md
├── requirements.txt
└── 01_eda_dataset.ipynb   # Exploración de datos (EDA) — Tarea 1
```

---

## Cómo ejecutar

### 1. Entorno e instalación

```bash
cd Laboratorio-3-Transformacion-Digital
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Ejecutar el notebook de EDA

```bash
jupyter notebook 01_eda_dataset.ipynb
```

O desde VS Code/Cursor: abrir `01_eda_dataset.ipynb` y ejecutar las celdas.

La primera vez que se ejecute, Keras descargará Fashion-MNIST automáticamente.

---

## Tareas del laboratorio

### Tarea 1: Exploración del dataset (EDA) ✅

En `01_eda_dataset.ipynb` se incluye:

- Tamaño del dataset y distribución de clases.
- Dimensiones y canales de las imágenes.
- Ejemplos de muestras por clase.
- Preprocesado necesario (normalización, forma para CNN).

El objetivo es comprender la estructura de los datos, no estadísticas exhaustivas.

---

## Requisitos

- Python 3.9+
- Dependencias listadas en `requirements.txt` (Jupyter, NumPy, Matplotlib, Seaborn, Pandas, TensorFlow).
