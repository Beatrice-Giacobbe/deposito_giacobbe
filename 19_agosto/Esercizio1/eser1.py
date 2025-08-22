
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
from sklearn.model_selection import cross_val_score


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

#grid search
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10]
}
grid_search = GridSearchCV(tree, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)
auc_tree = cross_val_score(tree, X_train, y_train, cv=skf, scoring="roc_auc")

print("Migliori parametri GridSearch:", grid_search.best_params_)
print("Miglior AUC GridSearch:", grid_search.best_score_)

#optuna

def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    auc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc").mean()
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Migliori parametri Optuna:", study.best_params)
print("Miglior AUC Optuna:", study.best_value)
