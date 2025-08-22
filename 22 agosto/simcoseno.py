from sentence_transformers import SentenceTransformer
import numpy as np
import ssl
import certifi

# 1) Inizializza il modello (solo CPU) - veloce e multilingue
# Buon compromesso qualità/velocità per IT: paraphrase-multilingual-MiniLM-L12-v2
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name, device="cpu")

# 2) Frasi da confrontare
sentence_1 = "Eventi in montagna nel 2025"
sentence_2 = "Cosa fare sulle Alpi"

# 3) Similarità coseno con embedding normalizzati (più stabile)
def semantic_similarity(sent1: str, sent2: str, model: SentenceTransformer) -> float:
    emb = model.encode([sent1, sent2], convert_to_numpy=True, normalize_embeddings=True)
    # Con vettori unitari il coseno è il prodotto scalare
    return float(np.dot(emb[0], emb[1]))

# 4) Calcolo e stampa del risultato
score = semantic_similarity(sentence_1, sentence_2, model)
print(f"Similarità semantica tra le frasi: {score:.4f}")