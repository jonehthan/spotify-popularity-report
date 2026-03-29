import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("track_data_final.csv")

# Feature engineering
df["album_release_year"] = pd.to_datetime(
    df["album_release_date"], errors="coerce"
).dt.year

df["track_duration_ms"] = np.log1p(df["track_duration_ms"])
df["artist_followers"] = np.log1p(df["artist_followers"])
df["explicit"] = df["explicit"].astype(int)

# Target
threshold = df["track_popularity"].quantile(0.80)
df["popular"] = (df["track_popularity"] >= threshold).astype(int)

# Feature matrix
features = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year"
]

X = df[features]

# 1️⃣ Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 2️⃣ Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 3️⃣ PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(
    np.cumsum(pca.explained_variance_ratio_),
    marker="o"
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["popular"],
    alpha=0.5
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (Colored by Popularity)")
plt.colorbar(label="Popular (1 = Yes)")
plt.tight_layout()
plt.show()
