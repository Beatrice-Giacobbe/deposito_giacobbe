import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# Generazione di dati a forma non circolare (mezzalune) + aggiunta outlier
from sklearn.datasets import make_moons

# Dati principali
X_base, _ = make_moons(n_samples=280, noise=0.1, random_state=42)

# Outlier casuali distribuiti nello spazio
outliers = np.random.uniform(low=-1.5, high=2.5, size=(20, 2))

# Concatenazione
X_with_outliers = np.vstack((X_base, outliers))

# k-Means clustering (forzato in 2 gruppi)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_with_outliers)

# DBSCAN clustering (basato sulla densità, senza specificare k)
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_with_outliers)

# Assegna colori ai cluster DBSCAN (grigio per outlier, colori per cluster)
unique_labels = set(dbscan_labels)
palette = sns.color_palette("Set2", len(unique_labels))
color_map = {
    label: palette[i] if label != -1 else (0.6, 0.6, 0.6)  # grigio per outlier
    for i, label in enumerate(sorted(unique_labels))
}
colors_dbscan = [color_map[label] for label in dbscan_labels]

# Visualizzazione: k-Means vs DBSCAN
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# k-Means
axs[0].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=kmeans_labels, cmap='Set1', s=40, edgecolor='black')
axs[0].set_title("k-Means: forza tutti i punti in 2 cluster")
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")
axs[0].grid(True)

# DBSCAN
axs[1].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=colors_dbscan, s=40, edgecolor='black')
axs[1].set_title("DBSCAN: forma naturale + outlier (grigio)")
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")
axs[1].grid(True)

plt.tight_layout()
plt.show()