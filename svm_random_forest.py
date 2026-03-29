import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("track_data_final.csv")

# Binary target (top 20%)
threshold = df["track_popularity"].quantile(0.80)
df["popular"] = (df["track_popularity"] >= threshold).astype(int)

# Feature engineering
df["album_release_year"] = pd.to_datetime(
    df["album_release_date"], errors="coerce"
).dt.year

df["track_duration_ms"] = np.log1p(df["track_duration_ms"])
df["artist_followers"] = np.log1p(df["artist_followers"])
df["explicit"] = df["explicit"].astype(int)

FEATURES = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year",
    "album_type",
    "artist_genres"
]

X = df[FEATURES]
y = df["popular"]

# =========================
# PREPROCESSING
# =========================
numeric_features = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year"
]

categorical_features = ["album_type", "artist_genres"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# =========================
# 80 / 10 / 10 SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed, y, test_size=0.20,
    stratify=y, shuffle=True, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=y_temp, shuffle=True, random_state=42
)

# =========================
# CLASS WEIGHTS
# =========================
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: neg / pos}

# =========================
# MODEL (FINAL CHOSEN CONFIG)
# =========================
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# =========================
# SVM MODEL
# =========================
svm_model = SVC(
    kernel="rbf",          # radial basis function kernel
    probability=True,      # enable probability estimates for ROC/PR AUC
    class_weight='balanced',  # handle class imbalance
    random_state=42
)

svm_model.fit(X_train, y_train)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_pred_svm = svm_model.predict(X_test)

# Metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)
pr_auc_svm = average_precision_score(y_test, y_prob_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("\n===== SVM PERFORMANCE =====")
print(f"Accuracy:  {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall:    {recall_svm:.4f}")
print(f"F1 Score:  {f1_svm:.4f}")
print(f"ROC-AUC:   {roc_auc_svm:.4f}")
print(f"PR-AUC:    {pr_auc_svm:.4f}")
print("Confusion Matrix:\n", cm_svm)

# =========================
# RANDOM FOREST MODEL
# =========================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

# Metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
pr_auc_rf = average_precision_score(y_test, y_prob_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\n===== RANDOM FOREST PERFORMANCE =====")
print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1 Score:  {f1_rf:.4f}")
print(f"ROC-AUC:   {roc_auc_rf:.4f}")
print(f"PR-AUC:    {pr_auc_rf:.4f}")
print("Confusion Matrix:\n", cm_rf)

import seaborn as sns

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# SVM Confusion Matrix
plot_confusion_matrix(cm_svm, "SVM Confusion Matrix")

# Random Forest Confusion Matrix
plot_confusion_matrix(cm_rf, "Random Forest Confusion Matrix")

from sklearn.metrics import RocCurveDisplay

# SVM
RocCurveDisplay.from_predictions(y_test, y_prob_svm, name="SVM")
# Random Forest
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest")
plt.plot([0,1], [0,1], "k--")
plt.title("ROC Curve")
plt.grid(True)
plt.show()

from sklearn.metrics import PrecisionRecallDisplay

# SVM
PrecisionRecallDisplay.from_predictions(y_test, y_prob_svm, name="SVM")
# Random Forest
PrecisionRecallDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()
