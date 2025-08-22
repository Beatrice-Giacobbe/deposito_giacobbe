import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# Carica le variabili dal file .env
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT1")

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5)
)
def ask():
    return client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Cos'è il backoff esponenziale?"}],
        temperature=0.7,
        max_tokens=200,
        stream=True
    )

response = ask()
print(response.choices[0].message.content)
