import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Simuliamo un dataset Olist semplificato per l'esercizio
np.random.seed(42)
n_customers = 500

olist_df = pd.DataFrame({
    'customer_id': [f"C{i}" for i in range(n_customers)],
    'recency_days': np.random.exponential(scale=90, size=n_customers),  # giorni dall’ultimo ordine
    'frequency_orders': np.random.poisson(lam=3, size=n_customers),      # numero ordini
    'monetary_total': np.random.gamma(shape=2, scale=150, size=n_customers),  # spesa totale
    'avg_review_score': np.clip(np.random.normal(loc=4, scale=0.5, size=n_customers), 1, 5)  # recensioni
})

# Rimuoviamo clienti con 0 ordini (eventuali errori)
olist_df = olist_df[olist_df['frequency_orders'] > 0]

# Selezioniamo le feature per il clustering
features = ['recency_days', 'frequency_orders', 'monetary_total', 'avg_review_score']
X = olist_df[features]

# Standardizziamo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Proviamo diversi k e salviamo silhouette score
silhouette_scores = []
K = range(2, 7)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Grafico silhouette score per diversi k
plt.figure(figsize=(8, 5))
sns.lineplot(x=list(K), y=silhouette_scores, marker='o')
plt.title("Silhouette Score al variare di k")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()