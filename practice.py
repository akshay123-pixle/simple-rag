from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

user_input = input("Enter your question: ")

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": user_input
        }
    ],
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    stream=True,
)

print("\nAssistant:\n")

for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()