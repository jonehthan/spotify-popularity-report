# import pandas as pd
# import numpy as np
#
# # =========================
# # DATA LOADING & TARGET
# # =========================
# print("Loading data...")
# df = pd.read_csv("track_data_final.csv")
#
# threshold = df["track_popularity"].quantile(0.80)
# df["popular"] = (df["track_popularity"] >= threshold).astype(int)
#
# df["album_release_year"] = pd.to_datetime(
#     df["album_release_date"], errors="coerce"
# ).dt.year
#
# df["track_duration_ms"] = np.log1p(df["track_duration_ms"])
# df["artist_followers"] = np.log1p(df["artist_followers"])
# df["explicit"] = df["explicit"].astype(int)
#
# FEATURES = [
#     "track_number",
#     "track_duration_ms",
#     "explicit",
#     "artist_popularity",
#     "artist_followers",
#     "album_total_tracks",
#     "album_release_year",
#     "album_type",
#     "artist_genres"
# ]
#
# X = df[FEATURES]
# y = df["popular"]
#
# # =========================
# # PREPROCESSING
# # =========================
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
#
# numeric_features = [
#     "track_number",
#     "track_duration_ms",
#     "explicit",
#     "artist_popularity",
#     "artist_followers",
#     "album_total_tracks",
#     "album_release_year"
# ]
#
# categorical_features = ["album_type", "artist_genres"]
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", Pipeline([
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())
#         ]), numeric_features),
#
#         ("cat", Pipeline([
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore"))
#         ]), categorical_features)
#     ]
# )
#
# print("Preprocessing features...")
# X_processed = preprocessor.fit_transform(X).toarray()
#
# # =========================
# # TRAIN / VAL / TEST SPLIT
# # =========================
# from sklearn.model_selection import train_test_split
#
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X_processed, y, test_size=0.2, stratify=y, random_state=42
# )
#
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
# )
#
# # =========================
# # MODEL DEFINITION
# # =========================
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.optimizers import SGD
#
# def build_mlp(
#     input_dim,
#     hidden_layers,
#     hidden_units,
#     dropout_rate,
#     activation,
#     kernel_initializer
# ):
#     model = Sequential()
#     model.add(Input(shape=(input_dim,)))
#
#     for _ in range(hidden_layers):
#         model.add(Dense(
#             hidden_units,
#             activation=activation,
#             kernel_initializer=kernel_initializer
#         ))
#         if dropout_rate > 0:
#             model.add(Dropout(dropout_rate))
#
#     model.add(Dense(1, activation="sigmoid"))
#
#     model.compile(
#         optimizer=SGD(learning_rate=0.01),
#         loss="binary_crossentropy",
#         metrics=[
#             "accuracy",
#             tf.keras.metrics.Precision(name="precision"),
#             tf.keras.metrics.Recall(name="recall"),
#             tf.keras.metrics.AUC(name="roc_auc"),
#             tf.keras.metrics.AUC(name="pr_auc", curve="PR")
#         ]
#     )
#     return model
#
# # =========================
# # HYPERPARAMETER SEARCH
# # =========================
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     average_precision_score
# )
#
# param_grid = {
#     "hidden_layers": [4, 5, 6],
#     "hidden_units": [20, 40],
#     "dropout_rate": [0.0, 0.2],
#     "activation": ["tanh", "relu"],
#     "kernel_initializer": [
#         "glorot_normal",
#     ]
# }
#
# best_pr_auc = 0
# best_config = None
#
# run_id = 1
# total_runs = (
#     len(param_grid["hidden_layers"]) *
#     len(param_grid["hidden_units"]) *
#     len(param_grid["dropout_rate"]) *
#     len(param_grid["activation"]) *
#     len(param_grid["kernel_initializer"])
# )
#
# print(f"Starting hyperparameter search ({total_runs} runs)...")
#
# for layers in param_grid["hidden_layers"]:
#     for units in param_grid["hidden_units"]:
#         for dropout in param_grid["dropout_rate"]:
#             for activation in param_grid["activation"]:
#                 for initializer in param_grid["kernel_initializer"]:
#
#                     print(f"\nRun {run_id}/{total_runs}")
#                     print(f"Layers={layers}, Units={units}, Dropout={dropout}, "
#                           f"Activation={activation}, Init={initializer}")
#
#                     model = build_mlp(
#                         input_dim=X_train.shape[1],
#                         hidden_layers=layers,
#                         hidden_units=units,
#                         dropout_rate=dropout,
#                         activation=activation,
#                         kernel_initializer=initializer
#                     )
#
#                     history = model.fit(
#                         X_train, y_train,
#                         validation_data=(X_val, y_val),
#                         epochs=30,
#                         batch_size=32,
#                         verbose=0,
#                         shuffle=True
#                     )
#
#                     y_val_prob = model.predict(X_val).ravel()
#                     y_val_pred = (y_val_prob >= 0.5).astype(int)
#
#                     acc = accuracy_score(y_val, y_val_pred)
#                     prec = precision_score(y_val, y_val_pred, zero_division=0)
#                     rec = recall_score(y_val, y_val_pred, zero_division=0)
#                     f1 = f1_score(y_val, y_val_pred, zero_division=0)
#                     roc_auc = roc_auc_score(y_val, y_val_prob)
#                     pr_auc = average_precision_score(y_val, y_val_prob)
#
#                     print(
#                         f"Val Acc={acc:.3f}, Prec={prec:.3f}, Recall={rec:.3f}, "
#                         f"F1={f1:.3f}, ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}"
#                     )
#
#                     if pr_auc > best_pr_auc:
#                         best_pr_auc = pr_auc
#                         best_config = {
#                             "hidden_layers": layers,
#                             "hidden_units": units,
#                             "dropout": dropout,
#                             "activation": activation,
#                             "initializer": initializer
#                         }
#
#                     run_id += 1
#
# # =========================
# # FINAL RESULTS
# # =========================
# print("\n==============================")
# print("BEST MODEL (by PR-AUC)")
# print("Best PR-AUC:", best_pr_auc)
# print("Best Configuration:", best_config)
# print("==============================")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD


# =========================
# Load and preprocess data
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

numeric_features = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year"
]

categorical_features = [
    "album_type",
    "artist_genres"
]

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

X_train, X_val, y_train, y_val = train_test_split(
    X_processed.toarray(),
    y,
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=42
)


# =========================
# Model builder
# =========================

def build_mlp(input_dim, activation):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(20, activation=activation, kernel_initializer="glorot_normal"),
        Dense(20, activation=activation, kernel_initializer="glorot_normal"),
        Dense(20, activation=activation, kernel_initializer="glorot_normal"),
        Dense(20, activation=activation, kernel_initializer="glorot_normal"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="binary_crossentropy"
    )

    return model


# =========================
# Train and plot
# =========================

activations = ["relu", "tanh", "elu"]

for act in activations:
    print(f"\nTraining model with activation: {act}")

    model = build_mlp(X_train.shape[1], activation=act)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=1
    )

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(f"Training vs Validation Loss ({act.upper()} Activation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
