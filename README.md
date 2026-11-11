# MediPredict: Diabetes Risk Prediction (Problem 3)

HackSphere Aayam XI | DSAI Club, NIT Agartala

## Problem Overview

Developed a binary classification machine learning pipeline to estimate the probability (`0.0` to `1.0`) that a patient record belongs to the prediabetes or diabetes category (`diabetes_risk`) using 21 demographic, lifestyle, and clinical health indicators (`202,943` training records and `50,737` test records).

## Approach

1. **Exploratory Data Analysis (EDA):** Checked missing values, analyzed class distribution (~14% positive class imbalance), and examined feature correlations with `diabetes_risk` using Pandas, Matplotlib, and Seaborn.
2. **Feature Engineering:** Created 5 domain-specific clinical interaction features:
   - `Cardio_Risk_Count`: Cumulative count of major cardiometabolic and mobility conditions (`HighBP`, `HighChol`, `HeartDiseaseorAttack`, `Stroke`, and `DiffWalk`).
   - `BMI_x_Age`: Captures how obesity risk compounds across age brackets.
   - `BMI_x_GenHlth`: Interaction between Body Mass Index and self-reported general health.
   - `Total_Unhealthy_Days`: Combined physical and mental poor health days in the past 30 days (`PhysHlth + MentHlth`).
   - `Lifestyle_Score`: Net protective lifestyle score (`PhysActivity + Fruits + Veggies` minus `Smoker + HvyAlcoholConsump`).
3. **Model Training & Validation:** Implemented a **10-Fold Stratified Cross-Validation** (`StratifiedKFold`) scheme to preserve class balance across all splits. Trained a baseline `LogisticRegression` model (with `StandardScaler`) alongside a tuned `XGBClassifier`, blending their out-of-fold and test probability predictions (`92% XGBoost + 8% Logistic Regression`) to maximize **ROC-AUC**.

## Validation Score (10-Fold Stratified Out-of-Fold ROC-AUC)

- **Logistic Regression:** `0.81924`
- **XGBoost Classifier:** `0.82640`
- **Blended Ensemble:** `0.82641`

## Repository Structure

- `train.py`: Complete end-to-end data preprocessing, feature engineering, 10-fold cross-validation training, and submission generation script.
- `MediPredict_Solution.ipynb`: Jupyter Notebook implementation of the pipeline.
- `submission.csv`: Final predicted probabilities for the 50,737 test records.
- `eda_plots.png` & `roc_curve.png`: Saved visualizations from EDA and model evaluation.
- `requirements.txt`: Required Python dependencies.

## How to Run

```bash
pip install -r requirements.txt
python train.py
```
