import spacy

# Carica il modello per l'italiano
nlp = spacy.load("it_core_news_sm")

# Testo da analizzare
testo = "Analizza grammaticalmente la frase: Il gatto mangia il pesce"

# Analisi
doc = nlp(testo)

# Output analisi
for token in doc:
    print(f"Token: {token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:10} | Dipendenza: {token.dep_:10} | Head: {token.head.text}")