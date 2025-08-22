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

response = client.completions.create(
    model=deployment,
    prompt="ciao, scrivi un testo sulla primavera. ",
    
)

print(response.choices[0].text)