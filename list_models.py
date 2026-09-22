"""Step 4: List the models your Nebius account can use.
Run:  python list_models.py
Copy one Nemotron model name into your .env file as NEBIUS_MODEL.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1",
    api_key=os.environ["NEBIUS_API_KEY"],
)

all_ids = sorted(m.id for m in client.models.list())
nemotron = [m for m in all_ids if "nemotron" in m.lower()]

if nemotron:
    print("Nemotron models available to you:")
    for m in nemotron:
        print("  ", m)
else:
    print("No model with 'nemotron' in its name was found. All models:")
    for m in all_ids:
        print("  ", m)
