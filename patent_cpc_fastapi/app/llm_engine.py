import ollama
import json
import re


def classify_with_llm(text: str):
    try:
        response = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a patent classification expert. "
                        "Return ONLY valid JSON in this format:\n"
                        "{ \"codes\": [\"G06N\", \"G06F\"], \"reasoning\": \"...\" }"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Classify this patent text:\n{text}",
                },
            ],
        )

        message = response.get("message", {})
        content = message.get("content", "")

        # extract JSON safely
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return {
            "codes": [],
            "reasoning": "Failed to parse LLM output",
        }

    except Exception as e:
        return {
            "codes": [],
            "reasoning": f"Error calling Ollama: {e}",
        }