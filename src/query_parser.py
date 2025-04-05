# query_parser.py

import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

def parse_query_with_llm(query: str) -> dict:
    prompt = f"""
You are an AI assistant helping recruiters choose SHL assessments.
Extract the following fields from the query:
1. skills → list of technologies, tools, soft skills
2. duration → in minutes (integer)
3. types → list of test types like: Technical, Cognitive, Personality, Behavioral

Query:
\"{query}\"

Respond strictly in this JSON format without explanation:
{{
  "skills": [...],
  "duration": number,
  "types": [...]
}}
"""

    try:
        response = model.generate_content(
            [prompt],
            generation_config={
                "temperature": 0.2,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 256
            }
        )

        print("\n🔎 Gemini Response:")
        print(response.text.strip())

        match = re.search(r"\{.*\}", response.text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError("❌ No valid JSON in response")

    except Exception as e:
        print("⚠️ Error parsing response:", e)
        return {
            "skills": [],
            "duration": None,
            "types": []
        }
