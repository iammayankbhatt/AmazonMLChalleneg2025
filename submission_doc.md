# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Code Guerrillas  
**Team Members:** Mayank Bhatt, Akshat Semwal, Mimansha Chauhan
**Submission Date:** 13-10-2025

---

## 1. Executive Summary

We developed a multimodal machine learning solution that combines text embeddings, image features, and engineered numeric features to predict product prices. Our approach leverages transformer-based text encoding with SentenceTransformers, CNN-based image feature extraction with EfficientNet-B0, and ensemble modeling with LightGBM, achieving robust price predictions through 5-fold cross-validation.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The challenge involves predicting product prices using catalog content text and product images. Key insights from EDA revealed:

**Key Observations:**
- Product descriptions contain valuable numeric information (quantities, sizes, counts)
- Text length and complexity correlate with product type and potentially price range  
- Image quality varies but provides complementary visual information
- Price distribution is right-skewed, suggesting log transformation would be beneficial

### 2.2 Solution Strategy

**Approach Type:** Multimodal Ensemble  
**Core Innovation:** Fusion of transformer-based text embeddings with CNN image features and engineered numeric features, optimized through cross-validated LightGBM training.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Raw Input (Text + Image)
↓
Text Processing Image Processing Feature Engineering
↓                   ↓                ↓
Sentence Transformer EfficientNet-B0 Numeric Feature Extraction
(384-dim) (1280-dim) (5 numeric features)
↓           ↓               ↓
└───────────────────┼───────────────────────┘
            ↓
Feature Concatenation (1664-dim)
            ↓
LightGBM Regression (5-fold CV)
        ↓
Ensemble Prediction
    ↓
Price Output

text

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing: Lowercasing, URL removal, punctuation removal, whitespace normalization
- Model: SentenceTransformer (all-MiniLM-L6-v2)
- Output: 384-dimensional embeddings

**Image Processing Pipeline:**
- Preprocessing: Resize to 224×224, ImageNet normalization
- Model: EfficientNet-B0 (pretrained on ImageNet1K)
- Output: 1280-dimensional embeddings

**Feature Engineering:**
- Numeric features from text: max, min, sum, mean, count of numbers in description
- Text statistics: length, word count, unique word count

**Ensemble Model:**
- Model: LightGBM Regressor
- Training: 5-fold cross-validation on log1p(price)
- Parameters: learning_rate=0.05, num_leaves=31, feature_fraction=0.8

---

## 4. Model Performance

### 4.1 Validation Results
- **OOF SMAPE Score:** 55.74%
- **Fold SMAPEs:** 56.81%, 55.94%, 55.63%, 55.29%
- **Stability:** ±0.58% standard deviation across folds

### 4.2 Key Findings
- Text embeddings provided the strongest predictive signals
- Image features added complementary information, especially for visually distinctive products  
- Numeric features from text (quantities, sizes) were highly informative
- Log transformation of prices improved model stability and performance

---

## 5. Conclusion

Our multimodal approach successfully combines modern NLP and computer vision techniques with traditional feature engineering. The LightGBM ensemble trained on fused embeddings demonstrates robust price prediction capabilities with consistent cross-validation performance. The solution is scalable, reproducible, and provides a solid foundation for further optimization.

---

## Appendix

### A. Technical Specifications
**Environment:**
- Python: 3.13.5
- PyTorch: 2.8.0+cu128
- CUDA: True
- GPU: NVIDIA GeForce GTX 1650

**Dependencies:** See requirements.txt

### B. File Manifest
- `solution.ipynb` - Complete implementation
- `src/utils.py` - Image download utilities  
- `models/lgb_fold*.pkl` - 5 trained LightGBM models
- `embeddings/` - Precomputed text and image embeddings
- `outputs/test_out.csv` - Final submission predictions
- `outputs/oof_preds.npy` - Validation predictions

### C. Reproducibility Notes
- All random seeds fixed (SEED=42)
- Embeddings cached to avoid recomputation
- Cross-validation strategy ensures robust evaluation
- No external data sources used
