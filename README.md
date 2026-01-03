# Satellite Imagery–Based Property Valuation  
### Multimodal Regression with Explainability

---

## 📌 Project Overview

This project builds a **multimodal house price prediction framework** by combining:

- **Tabular housing data** (structural & locational attributes)
- **Satellite imagery–derived visual features** (neighborhood context)

While tabular features capture intrinsic property characteristics, satellite images encode **external neighborhood signals** such as urban planning quality, greenery, density, accessibility, and water proximity.

The objective is to evaluate whether satellite imagery provides **interpretable predictive lift** over strong tabular baselines, while maintaining **engineering rigor and explainability**.

---

## 📂 Repository Structure & Execution Guide

The project consists of **three notebooks**, each with a clearly defined role.

---

## 1️⃣ `tabular_processing.ipynb`

**Purpose:**  
Feature engineering and strong baseline modeling using **tabular data only**.

**Required Inputs:**
- `train(1).csv`
- `test2.xlsx`

**What it does:**
- EDA (price distribution, size vs price, spatial effects)
- Semantic feature filtering (removes `id`, `zipcode`, redundant proxies)
- Feature selection using **Ridge Regression** and **XGBoost**
- Final selected features:
  - `sqft_living`, `bathrooms`, `grade`, `lat`, `yr_built`, `waterfront`, `view`
- Trains tabular-only models (Ridge / XGBoost)

This establishes a strong and realistic **tabular performance ceiling**.

---

## 2️⃣ `data_fetcher_preprocessing_CNN_grad_cam.ipynb`

**Purpose:**  
Satellite image extraction, CNN-based visual feature learning, and explainability using **Grad-CAM**.

**Required Inputs:**
- `train(1).csv`
- `test2.xlsx`
- `satellite_dataset.zip` (Mapbox-extracted images)
- **Mapbox API token** (already configured inside the notebook)

**What it does:**
- Locality-based sampling (~800 localities from ~16,000 rows)
- Satellite image extraction (~5 images per locality → ~4,000 images)
- Image preprocessing (CNN + Grad-CAM safe)
- CNN training using **pretrained ResNet-34**
- Train/Validation split: **3200 / 800 images**
- Extraction of fixed-length image embeddings
- Visual explainability using **Grad-CAM**

**Outputs Generated:**
- `image_embeddings_train.pt`
- `image_embeddings_val.pt`
- `image_targets_train.pt`
- `image_targets_val.pt`

Grad-CAM confirms the CNN focuses on **greenery, road layouts, water bodies, density, and open spaces**, validating economic relevance.

---

## 3️⃣ `multimodal_training.ipynb`

**Purpose:**  
Train and evaluate the **multimodal regression model** using tabular data + image embeddings.

**Required Inputs:**
- `train(1).csv`
- `test2.xlsx`
- Image embedding files:
  - `image_embeddings_train.pt`
  - `image_embeddings_val.pt`
  - `image_targets_train.pt`
  - `image_targets_val.pt`

**What it does:**
- Aligns tabular rows with locality-level image embeddings
- Trains:
  - Tabular-only baseline
  - Multimodal model (tabular + image embeddings)
- Evaluates both models on the same train/validation split
- Generates final predictions on the test dataset

**Performance (log-price):**

| Model | RMSE | R² |
|------|------|----|
| Tabular Only | 0.224 | 0.809 |
| Tabular + Image | 0.255 | 0.752 |

Satellite imagery adds **interpretable neighborhood context**, even when numeric lift is modest.

---

## 🧠 Key Design Principles

- Locality-level modeling (not point-level)
- No spatial or label leakage
- Representative sampling over brute-force scraping
- Modular multimodal pipeline
- Explainability-first design (Grad-CAM)

---

## 🔹 One-Line Summary

> Satellite imagery is used to learn external neighborhood quality signals that complement structured housing data, enabling a more complete and interpretable property valuation framework.
