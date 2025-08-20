import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 

# Generiamo dati di esempio
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)

# Applichiamo k-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Calcolo dei silhouette score
silhouette_avg = silhouette_score(X, labels)
sample_silhouette_values = silhouette_samples(X, labels)

# Preparo i dati ordinati per cluster e silhouette
sorted_labels = np.argsort(labels)
sorted_scores = sample_silhouette_values[sorted_labels]
sorted_clusters = labels[sorted_labels]

# Colori assegnati ai cluster per chiarezza
cluster_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
bar_colors = [cluster_colors[c] for c in sorted_clusters]

# Grafico semplificato e molto leggibile
plt.figure(figsize=(12, 4))
plt.bar(range(len(X)), sorted_scores, color=bar_colors, edgecolor='black')
plt.axhline(silhouette_avg, color='red', linestyle='--', linewidth=2, label=f'Media silhouette = {silhouette_avg:.2f}')
plt.title("Silhouette Score per ogni punto (colorato per cluster)", fontsize=14)
plt.xlabel("Punti ordinati per cluster", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.xticks([])
plt.legend()
plt.tight_layout()
plt.show()