"""
ragcheck quickstart — Gemini example

Requirements:
    pip install google-generativeai python-dotenv

Set GEMINI_API_KEY in a .env file or your environment.
"""

import os

from dotenv import load_dotenv
import google.generativeai as genai

import ragcheck

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")


def gemini_llm_fn(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text


question = "What causes the northern lights?"
answer = (
    "The northern lights, or aurora borealis, are caused by charged particles "
    "from the sun colliding with gases in Earth's atmosphere near the magnetic poles."
)
contexts = [
    "Aurora borealis occurs when charged particles ejected by the sun travel to Earth "
    "and interact with gases in the upper atmosphere, producing colorful light displays.",
    "Earth's magnetic field channels solar wind particles toward the poles, where they "
    "collide with oxygen and nitrogen atoms, releasing energy as visible light.",
]

result = ragcheck.evaluate(question, answer, contexts, llm_fn=gemini_llm_fn)

print(f"Faithfulness:       {result.faithfulness:.2f}  — {result.reasoning['faithfulness']}")
print(f"Answer relevance:   {result.answer_relevance:.2f}  — {result.reasoning['answer_relevance']}")
print(f"Context precision:  {result.context_precision:.2f}  — {result.reasoning['context_precision']}")
