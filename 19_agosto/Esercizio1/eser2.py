import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
path = r"C:\Users\SV273YL\Downloads\archive (1)\AirQualityUCI.csv"
df = pd.read_csv(path, sep=";")

#pre-processing
df = df.iloc[:, :-2]
cols = ["T", "RH", "AH"]
for col in cols:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float).clip(lower=0)
df['PT08.S1(CO)'].clip(lower=0)
df = df.dropna(subset=["PT08.S1(CO)"])
print(df)


#print(df.columns)   #scelgo inquinante PT08.S1(CO)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# calcolo media giornaliera di PT08.S1(CO)
df["media_PT08.S1(CO)_giornaliera"] = df.groupby(df["Date"].dt.date)["PT08.S1(CO)"].transform("mean")

# classifico qualità aria (target)
df["Qualita_aria"] = df.apply(
    lambda row: 1 if row["PT08.S1(CO)"] > row["media_PT08.S1(CO)_giornaliera"] else 0,
    axis=1)
conteggio = df["Qualita_aria"].value_counts()
print(df.isna().sum().sum())

X = df[["PT08.S1(CO)","T", "RH", "AH"]]
y = df["Qualita_aria"]

#print(X)

X_train1, X_test, y_train1, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print("Conteggio prima dello smote",y_train1.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train1, y_train1)
print("Conteggio dopo dello smote",y_train.value_counts())

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("DECISION TREEE:\n", classification_report(y_test, y_pred, digits=3))

#con RANDOM FOREST
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
print("RANDOM FOREST:\n", classification_report(y_test, y_pred1, digits=3))

"""
#con REGRESSIONE LOGISTICA
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred2 = logreg.predict(X_test)
"""





#se volessi fare per media settimanale, calcolo la media settimanale dell'inquinante
#e poi è tutto uguale
df1 = pd.read_csv(path, sep=";")
df1["Datetime"] = pd.to_datetime(
    df1["Date"].astype(str) + " " + df1["Time"].astype(str),
    format="%d/%m/%Y %H.%M.%S",  # il tuo formato: 10/03/2004 18.00.00
    errors="coerce"
)
# Ricavo settimana (numero settimana dell'anno)
df1["Settimana"] = df1["Datetime"].dt.isocalendar().week
