import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1) Dataset sintetico (6D, 3 classi)
X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=3,
    random_state=42
)

# 2) Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) PCA per confronto
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4) Autoencoder 6D -> 2D
input_dim = X_scaled.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)
code = Dense(encoding_dim, activation='linear')(encoded)

decoded = Dense(4, activation='relu')(code)
decoded = Dense(8, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder = Model(inputs=input_layer, outputs=code)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

# 5) Codifica 2D dall'encoder
X_ae = encoder.predict(X_scaled, verbose=0)

# 6) t-SNE (su dati standardizzati, non su PCA)
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 7) Visualizzazioni (PCA, Autoencoder, t-SNE)
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=50, edgecolor='black', alpha=0.8)
plt.title("PCA (lineare) — 6D → 2D")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(X_ae[:, 0], X_ae[:, 1], c=y, cmap='tab10', s=50, edgecolor='black', alpha=0.8)
plt.title("Autoencoder (non lineare) — 6D → 2D")
plt.xlabel("Z1"); plt.ylabel("Z2")
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=50, edgecolor='black', alpha=0.8)
plt.title("t-SNE (non lineare, manifold) — 6D → 2D")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.grid(True); plt.tight_layout()
plt.show()