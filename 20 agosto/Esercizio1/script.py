import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
#matplotlib.use("TkAgg") 
df = pd.read_csv("Mall_Customers.csv", sep=",")
print(df.isna().sum().sum())        #non ci sono nan
print(df.describe())
#print(df)

#pre-processing
le = LabelEncoder()
df["Genre"] = le.fit_transform(df["Genre"])  # converto sesso in 0/1
x = df[["Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
data = scaler.fit_transform(x)
#print(data)

#pca per grafico per capire quanti cluster
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clienti proiettati in PCA con tutte le features")
plt.show()

#k-means con solo "Annual Income (k$)", "Spending Score (1-100)"
x = df[["Annual Income (k$)", "Spending Score (1-100)"]]
plt.figure(figsize=(8,6))
plt.scatter(x["Annual Income (k$)"], x["Spending Score (1-100)"], c="blue", s=50, alpha=0.6, edgecolors="k")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Income vs Spending Score")
plt.grid(True)
plt.show()
scaler = StandardScaler()
data = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(data)

plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=100, label='Centroidi')
plt.title("Cluster trovati con k-Means")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.legend()
plt.grid(True)
plt.show()

#individuare cluster con clienti con alto reddito e alto score
centroidi = kmeans.cluster_centers_
print(centroidi)

sum_cent = [c[0] + c[1] for c in centroidi]
cluster_max = sum_cent.index(max(sum_cent))
print("cluster con clienti con alto reddito è il cluster n.", cluster_max+1)

df["cluster"] = labels

#recupero i clienti del cluster con centroide max
cluster10_clients = df[df["cluster"] == cluster_max]
print(cluster10_clients)