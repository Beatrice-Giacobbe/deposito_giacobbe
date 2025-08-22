import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Carica le variabili dal file .env
load_dotenv()
client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT1")

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        },
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)

print("Prova streaming:\n")

response = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Spiegami lo streaming completions"}],
    stream=True  # attiva lo streaming
)

for chunk in response:
    # controlla prima che choices esista e contenga elementi
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)