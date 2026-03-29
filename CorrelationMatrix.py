import pandas as pd
import numpy as np

df = pd.read_csv("track_data_final.csv")

# Feature engineering (same as before)
df["album_release_year"] = pd.to_datetime(
    df["album_release_date"], errors="coerce"
).dt.year

df["track_duration_ms"] = np.log1p(df["track_duration_ms"])
df["artist_followers"] = np.log1p(df["artist_followers"])

df["explicit"] = df["explicit"].astype(int)

# Binary target
threshold = df["track_popularity"].quantile(0.80)
df["popular"] = (df["track_popularity"] >= threshold).astype(int)

numeric_cols = [
    "track_number",
    "track_duration_ms",
    "explicit",
    "artist_popularity",
    "artist_followers",
    "album_total_tracks",
    "album_release_year",
    "track_popularity",
    "popular"
]

corr_df = df[numeric_cols].corr()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

im = plt.imshow(
    corr_df,
    cmap="coolwarm",    # 🔑 diverging colors
    interpolation="nearest",
    vmin=-1,
    vmax=1
)

plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(
    range(len(corr_df.columns)),
    corr_df.columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(corr_df.columns)), corr_df.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
