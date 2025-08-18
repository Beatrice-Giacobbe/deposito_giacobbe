from collections import Counter
file = "C:\\Users\\SV273YL\\OneDrive - EY\\Documents\\GitHub\\deposito_giacobbe\\18_agosto\\Esercizio1\\input.txt"

with open(file, "r", encoding="utf-8") as f:
        contenuto = f.read()
#print(contenuto)
def conta_righe(testo):
        return len(testo.split("\n"))
print("il testo contiene", conta_righe(contenuto), "righe")

def conta_parole(testo):
       parole = len(testo.split())
       return parole
print("il testo contiene", conta_parole(contenuto), "parole")

def freq_parole(testo):
        testo = testo.lower()
        parole = testo.split()
        frequenze = Counter(parole)
        return frequenze

for parola, freq in freq_parole(contenuto).most_common(5):
        print(f"{parola}: {freq}")