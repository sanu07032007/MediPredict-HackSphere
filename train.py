import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load the dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sub_df = pd.read_csv("sample_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# 2. Exploratory Data Analysis (EDA) Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=train_df, x="diabetes_risk", ax=axes[0], palette="Blues")
axes[0].set_title("Target Distribution (0 = No Diabetes, 1 = Diabetes Risk)")

sns.boxplot(data=train_df, x="diabetes_risk", y="BMI", ax=axes[1], palette="Blues")
axes[1].set_title("BMI vs Diabetes Risk")
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()

# 3. Feature Engineering
def add_features(df):
    df = df.copy()
    df["Cardio_Risk_Count"] = (
        df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"] + df["DiffWalk"]
    )
    df["BMI_x_Age"] = df["BMI"] * df["Age"]
    df["BMI_x_GenHlth"] = df["BMI"] * df["GenHlth"]
    df["Total_Unhealthy_Days"] = df["PhysHlth"] + df["MentHlth"]
    df["Lifestyle_Score"] = (
        (df["PhysActivity"] + df["Fruits"] + df["Veggies"]) - (df["Smoker"] + df["HvyAlcoholConsump"])
    )
    return df

train_fe = add_features(train_df)
test_fe = add_features(test_df)

target_col = "diabetes_risk"
id_col = "id"
pred_col = "prediction"

features = [c for c in train_fe.columns if c not in [target_col, id_col]]

X = train_fe[features]
y = train_fe[target_col].values
X_test = test_fe[features]

# 4. 10-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_lr = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
test_preds_lr = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))

print("\nTraining 10-Fold Model...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    # Baseline: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_scaled, y_train)
    oof_lr[val_idx] = lr.predict_proba(X_val_scaled)[:, 1]
    test_preds_lr += lr.predict_proba(X_test_scaled)[:, 1] / 10.0
    
    # Main Model: XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric="auc",
        early_stopping_rounds=80,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / 10.0
    
    fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"Fold {fold:2d} ROC-AUC: {fold_auc:.5f}")

# 5. Evaluate and Plot ROC Curve
lr_auc = roc_auc_score(y, oof_lr)
xgb_auc = roc_auc_score(y, oof_xgb)
oof_blend = 0.92 * oof_xgb + 0.08 * oof_lr
blend_auc = roc_auc_score(y, oof_blend)

print("\n--- Final Validation Results ---")
print(f"Logistic Regression ROC-AUC : {lr_auc:.5f}")
print(f"XGBoost ROC-AUC             : {xgb_auc:.5f}")
print(f"Final Blended ROC-AUC       : {blend_auc:.5f}")

fpr, tpr, _ = roc_curve(y, oof_blend)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {blend_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# 6. Save Final Submission
final_predictions = 0.92 * test_preds_xgb + 0.08 * test_preds_lr
submission = pd.DataFrame({
    id_col: test_df[id_col],
    pred_col: final_predictions
})
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv successfully!")