
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

path = r"C:\Users\SV273YL\OneDrive - EY\Desktop\datasetConsumption\AEP_hourly.csv"
df = pd.read_csv(path, parse_dates=["Datetime"])

media = df['AEP_MW'].mean()
#print("val medio", media)

def classifica(el,media):
    if el["AEP_MW"] > media:
        return 1
    else:
        return 0
    
# rispetto a valore globale
df["target_giornaliero"] = df.apply(lambda row: classifica(row, media), axis=1)

#rispetto a media del giorno settimanale
df["Giorno_settimana"] = df["Datetime"].dt.dayofweek
df["media_giornaliera_set"] = df.groupby("Giorno_settimana")["AEP_MW"].transform("mean")
df["target_giornaliero_sett"] = df.apply(lambda row: classifica(row, row["media_giornaliera_set"]), axis=1)

#rispetto al giorno
df['Giorno'] = df["Datetime"].dt.day
df["media_giornaliera"] = df.groupby("Giorno")["AEP_MW"].transform("mean")
df["target_giornaliero"] = df.apply(lambda row: classifica(row, row["media_giornaliera"]), axis=1)

print(df.head(5))

#con decision tree
x = df[['Giorno','Giorno_settimana']]
y = df['target_giornaliero']
X_train, X_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.25, random_state=42
)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

df1 = pd.DataFrame({
    "X_test": list(X_test.to_records(index=False)),  # converte le righe in tuple
    "Pred": y_pred
})
print(df1.head())

print("Decision Tree:")
print(classification_report(y_test, y_pred, digits=3))

