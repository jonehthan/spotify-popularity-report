import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# =========================
# Load + preprocess
# =========================
df = pd.read_csv("track_data_final.csv")

threshold = df["track_popularity"].quantile(0.80)
df["popular"] = (df["track_popularity"] >= threshold).astype(int)

df["album_release_year"] = pd.to_datetime(
    df["album_release_date"], errors="coerce"
).dt.year

df["track_duration_ms"] = np.log1p(df["track_duration_ms"])
df["artist_followers"] = np.log1p(df["artist_followers"])
df["explicit"] = df["explicit"].astype(int)

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

FEATURES = numeric_features + categorical_features

X = df[FEATURES]
y = df["popular"]

print("\nFEATURES USED:")
for f in FEATURES:
    print(" -", f)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# =========================
# Train / Val / Test split
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# =========================
# PCA (90% variance)
# =========================
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train.toarray())
X_val_pca = pca.transform(X_val.toarray())
X_test_pca = pca.transform(X_test.toarray())

print("\nPCA Components Retained:", X_train_pca.shape[1])

# =========================
# Model builder
# =========================
def build_best_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(20, activation="relu", kernel_initializer="glorot_normal"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="binary_crossentropy"
    )
    return model

# =========================
# Train + Evaluate function
# =========================
def train_and_evaluate(Xtr, Xva, Xte, label):
    model = build_best_model(Xtr.shape[1])

    model.fit(
        Xtr, y_train,
        validation_data=(Xva, y_val),
        epochs=50,
        batch_size=32,
        verbose=0,
        shuffle=True
    )

    y_prob = model.predict(Xte).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n===== {label} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))

# =========================
# Run both experiments
# =========================
train_and_evaluate(
    X_train.toarray(), X_val.toarray(), X_test.toarray(),
    "No PCA"
)

train_and_evaluate(
    X_train_pca, X_val_pca, X_test_pca,
    "With PCA (90% Variance)"
)
