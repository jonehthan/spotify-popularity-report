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

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(20, activation="relu", kernel_initializer="glorot_normal"),
    Dense(20, activation="relu", kernel_initializer="glorot_normal"),
    Dense(1, activation="sigmoid")
])

optimizer = SGD(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss="hinge",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(curve="PR", name="pr_auc")
    ]
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    class_weight=class_weight,
    verbose=1,
    shuffle=True
)

# =========================
# EVALUATION
# =========================
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n===== FINAL MODEL PERFORMANCE =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# =========================
# LOSS CURVE
# =========================
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
