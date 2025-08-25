import os
from dotenv import load_dotenv
from openai import AzureOpenAI

#usa gpt 3.5-turbo-instruct che non permette la chatcompletions, ma solo completions
# Carica le variabili dal file .env
load_dotenv()

client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Testo da trasformare in embedding
text = "Hello World"

response = client.embeddings.create(
    model=deployment,
    input=text
)

# l’embedding è un vettore di float
embedding_vector = response.data[0].embedding

print(f"Lunghezza embedding: {len(embedding_vector)}")
print(embedding_vector[:10])  # stampo solo i primi 10 valori
