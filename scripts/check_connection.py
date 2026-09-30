"""Step 5: Check that your Nebius API key and model work.
Run:  python check_connection.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1",
    api_key=os.environ["NEBIUS_API_KEY"],
)

response = client.chat.completions.create(
    model=os.environ["NEBIUS_MODEL"],
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=200,
)

print("Model replied:")
print(response.choices[0].message.content)
