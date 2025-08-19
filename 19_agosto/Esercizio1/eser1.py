
import pandas as pd
path = r"C:\Users\SV273YL\OneDrive - EY\Desktop\datasetConsumption\AEP_hourly.csv"
df = pd.read_csv(path, parse_dates=["Datetime"])

media = df['AEP_MW'].mean()
#print("val medio", media)

def classifica(el,media):
    if el["AEP_MW"] > media:
        return "Alto"
    else:
        return "Basso"
    
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
