from sklearn.model_selection import train_test_split
import pandas as pd

path = r"C:\Users\SV273YL\OneDrive - EY\Desktop\datasetConsumption\AEP_hourly.csv"
df = pd.read_csv(path, parse_dates=["Datetime"])
print(df.head())
# Etichetta: 1 se consumo > mediana, altrimenti 0
df["target"] = (df["AEP_MW"] > df["AEP_MW"].median()).astype(int)
df["hour"] = df['Datetime'].dt.hour
df["month"] = df['Datetime'].dt.month

# Feature: ora, giorno della settimana, mese
X = df[["hour", "month"]]
y = df["target"]

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")