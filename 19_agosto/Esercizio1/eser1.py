
import pandas as pd
path = r"C:\Users\SV273YL\OneDrive - EY\Desktop\datasetConsumption\AEP_hourly.csv"
df = pd.read_csv(path, parse_dates=["Datetime"])

df["Data"] = df["Datetime"].dt.date
media = df['AEP_MW'].mean()
print("val medio", media)

# rispetto a valore globale
def classifica_globale(el):
    if el["AEP_MW"] > media:
        return "Alto"
    else:
        return "Basso"
df["target_globale"] = df.apply(classifica_globale, axis=1)

#rispetto a media del giorno settimanale
df["Giorno_settimana"] = df["Datetime"].dt.dayofweek
df["media_giornaliera"] = df.groupby("Giorno_settimana")["AEP_MW"].transform("mean")
def classifica_giornaliera(el):
    if el["AEP_MW"] > el["media_giornaliera"]:
        return "Alto"
    else:
        return "Basso"

df["target_giornaliero"] = df.apply(classifica_giornaliera, axis=1)

print(df)
