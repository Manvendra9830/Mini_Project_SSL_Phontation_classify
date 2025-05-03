# Phonation Classification using Self-Supervised Models

## 🧠 Overview

This project focuses on classifying **phonation types**—namely **breathy**, **neutral**, and **pressed**—in speaking voice signals using advanced **Self-Supervised Learning (SSL)** models like **HuBERT** and **Wav2Vec 2.0** (Base & Large). These models extract rich, contextual embeddings from raw audio which are then fed into traditional classifiers like **SVM**, **MLP**, and **Random Forest** to achieve superior performance over traditional methods like MFCCs.

---

## 🌟 Objectives

- Develop a robust phonation classification pipeline using self-supervised models.
- Extract high-quality features from unlabeled speech data.
- Train multiple classifiers and compare their performances.
- Improve upon the traditional methods' accuracy (~81.67%).

---

## 📂 Dataset

- **Source**: IITM Voice Database
- **Classes**: `Breathy`, `Neutral`, `Pressed`
- **Modality**: Spoken voice signals
- **Format**: WAV files with labeled phonation types

---

## 🧰 Models & Methods

### 🔍 Feature Extraction

- **Self-Supervised Models**:
  - [x] Wav2Vec 2.0 Base
  - [x] Wav2Vec 2.0 Large
  - [x] HuBERT

### 🤖 Classifiers

- [x] Support Vector Machine (SVM)
- [x] Multi-Layer Perceptron (MLP)
- [x] Random Forest

---

## 📈 Results

- Achieved improved accuracy over conventional MFCC-based techniques.
- HuBERT and Wav2Vec-Large gave the best results in combination with SVM and MLP.
- Performed layer-wise analysis of model embeddings for optimal feature usage.
- **Confusion Matrix** and **Layer-wise accuracy plots** validate model robustness.

---

## 🏧 Project Structure

```
├── main.py                  # Main pipeline for feature extraction and classification
├── feature_extraction.py    # Self-supervised model integration
├── classify.py              # Classifier implementations
├── data/                    # Audio dataset
├── models/                  # Pre-trained model checkpoints (or download script)
├── plots/                   # Confusion matrices and evaluation plots
├── utils.py                 # Helper functions
└── requirements.txt         # Required Python packages
```

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- PyTorch
- torchaudio
- scikit-learn
- matplotlib, seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Running the Project

```bash
python main.py
```

---

## 📊 Evaluation

- Accuracy
- Confusion Matrix
- Layer-wise Analysis
- Cross-validation

---

## 📚 References

- Baevski et al., *Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
- Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction*

---

## 👥 Contributors

- **Pavan Kumar** - [CS22B1042]
- **Manvendra Singh** - [CS22B1054]

Under the guidance of **Dr. Kiran Reddy Mittapalle**  
Indian Institute of Information Technology, Raichur

---

## 📃 License

This project is for educational purposes only and may include usage of copyrighted datasets and pre-trained models.
