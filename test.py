import os
from dotenv import load_dotenv
from google import genai

print("Step 1: script started")

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Step 2: key loaded, starts with:", key[:5] if key else "NONE FOUND")

client = genai.Client(api_key=key)
print("Step 3: client created, sending request...")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say hello and confirm you're working in one short sentence."
)
print("Step 4: got a response back")

print(response.text)