# Heart Failure Prediction
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartfailureprediction-suhanik19.streamlit.app/)  

An interactive machine learning web application built with Streamlit to predict the risk of heart failure based on patient clinical data.  
The project implements Logistic Regression, Random Forest, and a simple Neural Network, with SHAP explainability available for the Random Forest model.

---

## Features
- Predicts patient risk of heart failure from clinical features (age, cholesterol, blood pressure, etc.).
- Implements multiple ML models:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (high performance with SHAP explainability)
  - Neural Network (experimental, non-linear patterns)
- SHAP explainability for Random Forest:
  - Global feature importance bar plots
  - Individual patient waterfall plots
- Interactive Streamlit interface for patient data entry and model selection.

---

## Installation & Setup

### 1. Clone the repository
```
bash
git clone https://github.com/suhanik19/HeartFailurePrediction.git
cd HeartFailurePrediction
```

### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the Application
```
streamlit run app/app.py
```

## Project Structure
```
HeartFailurePrediction/
│── app/
│   └── app.py              # Streamlit app
│
│── src/
│   ├── train_model.py      # Model training functions
│   ├── evaluate_model.py   # Evaluation metrics
│   ├── explainability.py   # SHAP explainability functions
│   └── data_processing.py  # Preprocessing and encoding
│
│── data/
│   └── heart.csv           # Dataset (or instructions to download)
│
│── requirements.txt
│── README.md
│── .gitignore
```
---

## Dataset and Features

The dataset combines multiple heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog), resulting in 918 unique patient records after removing duplicates. Each record contains 11 clinical attributes commonly associated with heart disease:

| Feature                | Description                                                                 | Type / Values                                |
|-------------------------|-----------------------------------------------------------------------------|----------------------------------------------|
| **Age**                | Patient age in years                                                        | Numeric                                      |
| **Sex**                | Gender                                                                      | 0 = Female, 1 = Male                         |
| **Chest Pain Type**     | Type of chest pain                                                          | TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic |
| **Resting Blood Pressure** | Resting blood pressure in mm Hg                                          | Numeric                                      |
| **Serum Cholesterol**   | Cholesterol level in mg/dL                                                  | Numeric                                      |
| **Fasting Blood Sugar** | Blood sugar after fasting                                                   | 1 if > 120 mg/dL, else 0                     |
| **Resting ECG**         | Resting electrocardiogram results                                           | Normal, ST-T wave abnormality, LVH           |
| **Maximum Heart Rate (MaxHR)** | Maximum heart rate during exercise test                              | Numeric (60–202 bpm)                         |
| **Exercise-Induced Angina** | Whether angina occurred during exercise                                 | 1 = Yes, 0 = No                              |
| **Oldpeak**             | ST depression induced by exercise relative to rest                          | Numeric                                      |
| **ST_Slope**            | Slope of the ST segment                                                     | Up, Flat, Down                               |
| **Target (HeartDisease)** | Presence of heart disease                                                 | 1 = Yes, 0 = No                              |

---

## Machine Learning Models & Explainability

- **Logistic Regression**: serves as a simple, interpretable baseline.  
- **Random Forest**: higher performance, with SHAP explainability for both global and individual-level predictions.  
- **Neural Network**: experimental model demonstrating flexibility for non-linear data patterns.  

---

## Changelog

### v1.0 — 09/17/2024
- First working release.

### v1.1 — 12/13/2024
- Utilized Recursive Feature Elimination to identify the most important features for model training, improving model interpretability and reducing overfitting by selecting a subset of relevant features.  

### v1.2 — 08/2025 
- Integrated SHAP explainability for Random Forest (global and per-patient plots).  
- Improved Streamlit application with model selection and enhanced UI.  
- Restricted SHAP explainability to Random Forest.

### v1.3 - 09/2025
- Deployed to Streamlit Cloud for live access. 
- Integrated hyperparameter optimization (GridSearchCV) for Logistic Regression, Random Forest, and Neural Network models.  
- Improved model accuracy and robustness through cross-validation and parameter tuning. 

---

## Future Work 
- Extend explainability for Neural Networks using DeepSHAP or other methods.  

---

## Author
**Suhani Khandelwal**  
Biomedical Engineering + ICS, UC Irvine  

- LinkedIn: [linkedin.com/in/suhanik19](https://linkedin.com/in/suhanik19)  
- Email: *[suhanikhandelwal05@gmail.com]*  

---

## Acknowledgements
This project uses the Heart Disease dataset originally compiled from multiple institutions and made available by the UCI Machine Learning Repository.

**Creators**:  
1. Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.  
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.  
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.  
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.  

**Donor**: David W. Aha (aha@ics.uci.edu)  
