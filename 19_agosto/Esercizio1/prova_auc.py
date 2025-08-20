
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
import pandas as pd

path = r"C:\Users\SV273YL\OneDrive - EY\Desktop\datasetConsumption\AEP_hourly.csv"
df = pd.read_csv(path, parse_dates=["Datetime"])
#print(df.head())
# Etichetta: 1 se consumo > mediana, altrimenti 0
df["target"] = (df["AEP_MW"] > df["AEP_MW"].median()).astype(int)
df["hour"] = df['Datetime'].dt.hour
df["month"] = df['Datetime'].dt.month

# Feature e target (come prima)
X = df[["hour", "month"]]
y = df["target"]

# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
auc_tree = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")

# Neural Network con scaling
mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
])
auc_mlp = cross_val_score(mlp_pipeline, X, y, cv=skf, scoring="roc_auc")

print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")
print(f"Neural Network AUC: {auc_mlp.mean():.3f} ± {auc_mlp.std():.3f}")
