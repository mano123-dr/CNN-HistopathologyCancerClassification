# Histopathology Cancer Patch Classification
### Using Convolutional Neural Networks and Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-PatchCamelyon-green)](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon)

---

## Project Overview

This project implements a supervised binary image classification pipeline for histopathology cancer patch classification using the **PatchCamelyon (PCam)** dataset. The task is to classify 96 × 96 pixel RGB histopathology image patches as either **non-cancerous (Label 0)** or **cancerous / metastatic (Label 1)**.

Two models are developed, trained and compared:
- **Custom CNN** — a baseline convolutional network trained from scratch
- **MobileNetV2** — a lightweight transfer learning model with ImageNet-pretrained weights

> **Disclaimer:** This is an academic project. The models are not intended for clinical diagnostic use.

---

## Dataset

| Feature | Description |
|---|---|
| Name | PatchCamelyon / Metastatic Tissue Classification |
| Source | Kaggle / PatchCamelyon benchmark |
| Total images | 327,680 |
| Image size | 96 × 96 pixels (RGB) |
| Labels | 0 = Non-cancer, 1 = Cancer |
| File format | H5 (.h5) |

**Dataset link:**  
https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon

Download the dataset and place all H5 files in a folder, then update `DATA_DIR` in the notebook accordingly.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Custom CNN   | 0.7672 | 0.9358 | 0.5735 | 0.7112 | 0.9037 |
| MobileNetV2  | 0.7888 | 0.8226 | 0.7363 | 0.7771 | 0.8749 |

**MobileNetV2 was selected as the preferred model** due to its higher recall (0.74 vs 0.57) and F1-score, meaning it missed fewer cancer-positive patches — a critical consideration in medical imaging tasks.

---

## Repository Structure

```
histopathology-cancer-classification/
│
├── histopathology_cancer_classification.ipynb   # Main notebook (full pipeline)
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
│
└── outputs/                                      # Generated figures (after running)
    ├── figure1_class_distribution.png
    ├── figure2_sample_patches.png
    ├── figure4_5_cnn_accuracy.png
    ├── figure4_5_cnn_loss.png
    ├── figure6_cnn_confusion_matrix.png
    ├── figure7_8_mnv2_accuracy.png
    ├── figure7_8_mnv2_loss.png
    ├── figure9_mnv2_confusion_matrix.png
    ├── figure10_roc_comparison.png
    ├── figure11_performance_comparison.png
    └── figure12_sample_predictions.png
```

---

## Methodology

1. **Dataset loading** — H5 files loaded using `h5py`; labels squeezed to 1D arrays
2. **Data exploration** — class distribution analysis and sample image visualisation
3. **Preprocessing** — pixel normalisation (Custom CNN: `Rescaling(1./255)`; MobileNetV2: `preprocess_input`)
4. **Data pipeline** — memory-efficient batch generator + `tf.data.Dataset`
5. **Augmentation** — random horizontal/vertical flip, rotation (±10%), zoom (±10%)
6. **Custom CNN** — 4 conv blocks + GlobalAveragePooling + Dense + Dropout + Sigmoid output
7. **MobileNetV2** — frozen ImageNet base + new classification head (fine-tuning optional)
8. **Training** — Adam optimiser, binary cross-entropy loss, EarlyStopping + ReduceLROnPlateau
9. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC-AUC

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/histopathology-cancer-classification.git
cd histopathology-cancer-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon) and place the H5 files in a local directory.

### 4. Configure the data path

Open the notebook and update the `DATA_DIR` variable in **Cell 2**:

```python
DATA_DIR = "/path/to/your/patchcamelyon/h5/files"
```

### 5. Run the notebook

```bash
jupyter notebook histopathology_cancer_classification.ipynb
```

Run all cells in order. Training may take several minutes depending on hardware.

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.10 |
| h5py | ≥ 3.7 |
| NumPy | ≥ 1.23 |
| Pandas | ≥ 1.5 |
| Matplotlib | ≥ 3.6 |
| Seaborn | ≥ 0.12 |
| scikit-learn | ≥ 1.2 |

See `requirements.txt` for the full pinned list.

---

## Key References

- Veeling et al. (2018) — PatchCamelyon benchmark dataset
- Ehteshami Bejnordi et al. (2017) — Deep learning for lymph node metastasis detection
- Sandler et al. (2018) — MobileNetV2: Inverted Residuals and Linear Bottlenecks
- Litjens et al. (2017) — Survey on deep learning in medical image analysis
- Powers (2011) — Evaluation: Precision, Recall, F-measure, ROC

---

## License

This project is for academic purposes only. The PatchCamelyon dataset is subject to its own licence terms available on the [Kaggle dataset page](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon).
