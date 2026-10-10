import os
import time
from dotenv import load_dotenv
from google import genai
import pandas as pd
import tools as t

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

df = pd.read_csv("serverlife_shifts.csv")

def average_by_day_of_week() -> dict:
    """Returns the average take-home pay per hour, grouped by day of week, sorted highest to lowest."""
    return t.average_by_day_of_week(df)

def best_and_worst_shifts(n: int = 3) -> dict:
    """Returns the n highest-earning and n lowest-earning shifts by total take-home pay."""
    return t.best_and_worst_shifts(df, n)

def overall_summary() -> dict:
    """Returns total shifts worked, total take-home pay, average pay per shift, and average pay per hour."""
    return t.overall_summary(df)


MODEL_FALLBACK_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

def ask_agent(question, retries_per_model=2, wait_seconds=5):
    """Tries each model in MODEL_FALLBACK_CHAIN in order, retrying each a few times
    before giving up on it and moving to the next, more established model."""
    for model_name in MODEL_FALLBACK_CHAIN:
        chat = client.chats.create(
            model=model_name,
            config={"tools": [average_by_day_of_week, best_and_worst_shifts, overall_summary]},
        )
        for attempt in range(1, retries_per_model + 1):
            try:
                response = chat.send_message(question)
                print(f"(answered using {model_name})")
                return response.text
            except Exception as e:
                print(f"{model_name} attempt {attempt} failed: {e}")
                if attempt < retries_per_model:
                    time.sleep(wait_seconds)
        print(f"Giving up on {model_name}, falling back to the next model...")

    return "All models failed — Google's servers may be fully overloaded right now. Try again in a bit."


question = "What day of the week do I get paid the most on average, and what were my best and worst shifts?"
print(ask_agent(question))