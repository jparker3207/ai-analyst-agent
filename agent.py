import os
import time
import concurrent.futures
from dotenv import load_dotenv
from google import genai
import tools as t

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

df = t.load_data("serverlife_shifts.csv")

def list_columns() -> dict:
    """Lists the column names available in the dataset, along with their data types."""
    return t.list_columns(df)

def describe_column(column_name: str) -> dict:
    """Returns summary statistics for a single column in the dataset.

    Args:
        column_name: The name of the column to describe (e.g. "take_home").
    """
    return t.describe_column(df, column_name)

def group_and_aggregate(group_by_column: str, value_column: str, aggregation: str = "mean") -> dict:
    """Groups the data by one column and aggregates another column within each group.

    Args:
        group_by_column: The column to group rows by (e.g. "day_of_week").
        value_column: The numeric column to aggregate within each group (e.g. "take_home").
        aggregation: The aggregation to apply: "mean", "sum", "count", "min", or "max". Defaults to "mean".
    """
    return t.group_and_aggregate(df, group_by_column, value_column, aggregation)

def top_n(column_name: str, n: int = 5, ascending: bool = False) -> dict:
    """Returns the top (or bottom) n rows of the dataset, ranked by a given column.

    Args:
        column_name: The column to rank rows by (e.g. "take_home").
        n: How many rows to return. Defaults to 5.
        ascending: If True, returns the smallest values instead of the largest (the bottom n). Defaults to False.
    """
    return t.top_n(df, column_name, n, ascending)


TOOLS = [list_columns, describe_column, group_and_aggregate, top_n]

MODEL_FALLBACK_CHAIN = [
    "gemini-4-argon",    # newest flagship model as of Oct 2026 -- tried first
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]  # gemini-2.5-flash removed: Google returns 404 "no longer available to new users" for it

API_TIMEOUT_SECONDS = 20


def new_chat(model_index):
    """Starts a brand-new conversation with the given model from MODEL_FALLBACK_CHAIN."""
    return client.chats.create(
        model=MODEL_FALLBACK_CHAIN[model_index],
        config={"tools": TOOLS},
    )


def send_with_timeout(chat, question, timeout=API_TIMEOUT_SECONDS):
    """Sends a message to Gemini, but gives up after `timeout` seconds instead of
    hanging forever if the connection stalls. The SDK itself has no reliable built-in
    timeout, so this runs the call in a background thread and walks away from it if
    it takes too long."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(chat.send_message, question)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"No response after {timeout} seconds -- the connection likely stalled.")
    finally:
        executor.shutdown(wait=False)


def ask_question(model_index, chat, question):
    """Tries to answer `question`, retrying the same model a couple times, then falling
    back through MODEL_FALLBACK_CHAIN as needed -- automatically retrying this SAME
    question on each new model, so the user is never asked to repeat themselves.
    Returns (model_index, chat, answer_text_or_None)."""
    while True:
        answer_text = None
        for attempt in range(1, 3):
            print(f"Contacting {MODEL_FALLBACK_CHAIN[model_index]} (attempt {attempt}, timeout {API_TIMEOUT_SECONDS}s)...", flush=True)
            try:
                response = send_with_timeout(chat, question)
                answer_text = response.text
                break
            except Exception as e:
                print(f"{MODEL_FALLBACK_CHAIN[model_index]} failed (attempt {attempt}): {e}")
                if attempt < 2:
                    print("Retrying in 5 seconds...", flush=True)
                    time.sleep(5)

        if answer_text is not None:
            return model_index, chat, answer_text

        if model_index < len(MODEL_FALLBACK_CHAIN) - 1:
            failed_model = MODEL_FALLBACK_CHAIN[model_index]
            model_index += 1
            print(f"{failed_model} is not responding. Switching to backup model: {MODEL_FALLBACK_CHAIN[model_index]}...")
            print("Retrying your same question on the new model (earlier conversation history is lost).\n")
            chat = new_chat(model_index)
            # loop continues: same question gets tried again, now on the new chat/model
        else:
            return model_index, chat, None


def run_interactive_session():
    """Keeps asking the user for questions and answering them using the same ongoing
    conversation, so the agent remembers earlier questions in this session, until the
    user types 'exit' or 'quit'."""
    print("AI Data Analyst Agent -- ask me anything about serverlife_shifts.csv.")
    print("Type 'exit' or 'quit' to stop.\n")

    model_index = 0
    chat = new_chat(model_index)

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        print("Thinking about your question...", flush=True)
        model_index, chat, answer = ask_question(model_index, chat, question)

        if answer is not None:
            print(f"\nAgent: {answer}\n")
        else:
            print("All backup models failed too. Sorry, I could not get an answer right now -- try again in a bit.\n")


if __name__ == "__main__":
    run_interactive_session()
