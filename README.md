# 📸 Sistema de Reconocimiento Facial IoT mediante Redes Neuronales Convolucionales (CNN)

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)

Este repositorio contiene el *pipeline* completo para la creación, estandarización y entrenamiento de un modelo biométrico de reconocimiento facial diseñado para integrarse en arquitecturas de Internet de las Cosas (IoT). Utiliza técnicas de visión por computadora para la generación automatizada de conjuntos de datos y Aprendizaje por Transferencia (*Transfer Learning*) para el entrenamiento eficiente del modelo predictivo.

## 👥 Equipo de Desarrollo (ESCOM - IPN)
* **Martínez Marín Sebastián**
* **Mondragón Romero Montserrat**
* **Negrete Pérez Juan Carlos**
* **Sánchez De Jesús Arlet Berenice**
* **Sánchez Valeriano Alexandra**

---

## ⚙️ Arquitectura del Proyecto

El flujo de trabajo (*Data Pipeline*) está estructurado en tres fases secuenciales ejecutables desde entornos de *Google Colab* o localmente.

### Fase 1: Extracción Automatizada (Video a *Frames*)
Para escalar el volumen de datos orgánicos, el script `fase1_extraccion.py` (o la Celda 1) procesa secuencias de video crudas (`.mp4`, `.avi`) proporcionadas por los integrantes. Extrae un fotograma a intervalos regulares (ej. cada 10 *frames*) para capturar diversas posturas, condiciones de iluminación y expresiones faciales, guardándolos en un directorio temporal.

### Fase 2: Unificación, Limpieza y *Data Augmentation*
El script `fase2_unificacion.py` orquesta la limpieza profunda del *dataset*:
* **Estandarización de Formatos:** Conversión de extensiones heterogéneas (incluyendo `.webp` con canal Alpha) a matrices estables RGB.
* **Estructura de Clases:** Generación de subcarpetas automáticas por integrante, incluyendo una clase vital de **"Desconocidos"** para rechazar intentos de intrusión en el sistema IoT.
* **Balanceo de Datos:** Implementación de aumento sintético (Efecto Espejo horizontal y rotaciones vía OpenCV) para garantizar que todas las clases alcancen una meta base (ej. 1,000 imágenes), evitando el sesgo probabilístico en la red neuronal.

### Fase 3: Entrenamiento con Transfer Learning (MobileNetV2)
El script `fase3_entrenamiento.py` define la arquitectura de la Red Neuronal Convolucional (CNN):
* Se instancia **MobileNetV2** con pesos pre-entrenados en *ImageNet*, congelando su base convolucional para optimizar los recursos computacionales.
* Se incorporan capas de regularización (`Dropout` al 50%) y variabilidad sintética en memoria RAM para mitigar el sobreajuste (*Overfitting*).
* Exporta el modelo final compilado en formato `.keras` y el mapeo de clases en un archivo `.json`, listos para ser desplegados en dispositivos de borde (ej. Raspberry Pi).

---

## 🚀 Requisitos e Instalación

Para ejecutar este *pipeline*, asegúrate de tener instalado el siguiente entorno de dependencias:

```bash
pip install opencv-python pillow tensorflow matplotlib numpy
