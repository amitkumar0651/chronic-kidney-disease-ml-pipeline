# 🏥 Chronic Kidney Disease (CKD) Prediction Pipeline

An end-to-end Machine Learning engineering pipeline designed to predict Chronic Kidney Disease using clinical patient records[cite: 1, 2]. This project covers comprehensive data cleaning, info-leakage preventative imputation, automated preprocessing, feature scaling validation, and a performance evaluation across 7 distinct ML architectures[cite: 1, 2].

---

## 1. Project Executive Summary
The objective of this project was to build a highly robust, end-to-end machine learning pipeline capable of predicting **Chronic Kidney Disease (CKD)** based on a clinical dataset of 400 patient records[cite: 1, 2]. The primary goal was to systematically clean the data, impute missing values without leaking information, evaluate a diverse lineup of 7 classification algorithms, and extract key clinical indicators driving the predictions[cite: 1, 2].

---

## 2. Pipeline Architecture & Methodology

### Phase 1: Data Cleaning & Structural Refinement (Sections 1–11)
* **Anomaly Correction:** Identified and stripped hidden whitespaces and structural typos in categorical features (e.g., converting accidental `'\tno'` to `'no'` and clean binary classes)[cite: 1, 2].
* **Data Type Casting:** Corrected improperly stored columns, transforming variables containing numeric values disguised as strings back into clean floating-point representations[cite: 1, 2].
* **Robust Feature Imputation:** Handled missing values strategically using a split approach to prevent data leakage[cite: 1, 2]. Continuous numerical features were filled using data **medians**, while discrete categorical columns were filled using data **modes** (most frequent values)[cite: 1, 2].

### Phase 2: Feature Engineering & Preprocessing (Sections 12–14)
* **Binary Text Encoding:** Mapped clinical text parameters (`normal`/`abnormal`, `present`/`notpresent`, `yes`/`no`, `good`/`poor`) into structured binary dimensions ($1$ and $0$)[cite: 1, 2].
* **Exploratory Data Analysis (EDA):** Built an automated loop that generated 4 independent clinical distribution plots (`blood_glucose_random`, `albumin`, `blood_urea`, and `age`) segmented by disease classification.
* **Stratified Splitting:** Partitioned the clean dataset into an **80% training set (320 patients)** and a **20% testing set (80 patients)** using stratified splitting to perfectly preserve the clinical representation ratio[cite: 1, 2].

---

## 3. Comprehensive Model Evaluation Leaderboard

We systematically evaluated 7 distinct machine learning architectures across tree ensembles, linear models, instance-based learners, and kernel methods[cite: 1, 2]. By implementing feature scaling via `StandardScaler`, distance-based models (SVM and KNN) experienced massive performance leaps[cite: 1, 2]:

| Rank | Machine Learning Model | Baseline Accuracy (%) | Post-Scaling Accuracy (%) | Key Takeaways & Architectural Insights |
| :--- | :--- | :---: | :---: | :--- |
| **1** | **Random Forest** | **100.00%** | **100.00%** | Flawless performance. Ensemble trees excel at capturing distinct multi-feature clinical thresholds[cite: 1, 2]. |
| **2** | **Gradient Boosting** | **100.00%** | **100.00%** | Perfect prediction by building sequential trees that minimize lingering classification errors[cite: 1, 2]. |
| **3** | **Support Vector Machine (SVM)**| 62.50% | **98.75%** | **Massive 36.25% jump!** Scaling completely resolved the magnitude dominance issues of raw numbers[cite: 1, 2]. |
| **4** | **Logistic Regression** | 93.75% | **98.75%** | Scaling improved convergence, pushing accuracy up by 5.00% due to better feature weighting[cite: 1, 2]. |
| **5** | **Decision Tree** | 98.75% | 98.75% | Highly effective on tabular logic; invariant to feature scaling magnitudes[cite: 1, 2]. |
| **6** | **XGBoost** | 98.75% | 98.75% | Ultra-precise modern tree framework; missed only a single patient case out of 80[cite: 1, 2]. |
| **7** | **K-Nearest Neighbors (KNN)** | 67.50% | **97.50%** | **Massive 30.00% jump!** Standardizing distance measurements stopped large ranges from drowning out small values[cite: 1, 2]. |

---

## 4. Clinical Feature Importance

Extracting internal feature importances from our optimal Random Forest classifier revealed exactly which variables guided the model's logic[cite: 1, 2]. The **Top 5 Most Critical Medical Predictors** identified by the pipeline are[cite: 1, 2]:

1. **Packed Cell Volume (PCV) [~17.16%]:** Measures the proportion of blood made up of red cells[cite: 1, 2].
2. **Haemoglobin [~15.54%]:** Oxygen-carrying protein in the blood[cite: 1, 2].
3. **Serum Creatinine [~15.14%]:** Waste product filtered by the kidneys; a direct indicator of renal filtration efficiency[cite: 1, 2].
4. **Red Blood Cell Count [~14.55%]:** Overall red blood cell density[cite: 1, 2].
5. **Specific Gravity [~9.49%]:** Measures urine concentration, highlighting kidney dilution capabilities[cite: 1, 2].

> **Clinical Validation:** The model's algorithmic selections align flawlessly with real-world pathophysiology[cite: 1, 2]. Severe kidney damage directly disrupts erythropoietin production (causing drops in PCV, Hemoglobin, and RBC counts, indicating anemia) while causing a critical retention of waste toxins like Serum Creatinine in the bloodstream[cite: 1, 2]. The model's independent mathematical path mirrors biological reality perfectly[cite: 1, 2].

---

## 5. Production Engineering Strategy (Serialization)

To ensure production readiness, the optimal models and underlying transformation matrices have been serialized into persistent binary storage formats via `pickle` protocol layers[cite: 1, 2]. This locks the internal parameters and saves state weights permanently[cite: 1, 2]:

* `ckd_random_forest_model.pkl` — Zero-preconditioning tree matrix ensemble for clinical deployment[cite: 1, 2].
* `ckd_svm_scaled_model.pkl` — Optimized hyperplane classifier for alternative dual-verification routines[cite: 1, 2].
* `ckd_scaler.pkl` — Persistent mathematical `StandardScaler` state tracking mean/variance configurations required to process any subsequent new incoming patient vectors safely[cite: 1, 2].

---

## 6. Repository File Architecture
```text
├── ckd_prediction.ipynb         # Main Jupyter Notebook containing all pipeline sections
├── ckd_clean_data.csv           # Final cleaned, imputed, and encoded clinical dataset
├── ckd_random_forest_model.pkl  # Serialized champion Random Forest model
├── ckd_svm_scaled_model.pkl     # Serialized alternative SVM model
├── ckd_scaler.pkl               # Saved StandardScaler preprocessing state
└── README.md                    # Detailed pipeline documentation
