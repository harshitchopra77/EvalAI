import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────────
# Configuration  →  set these in .env
# ─────────────────────────────────────────────
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic")   # anthropic | openai | gemini | custom
AI_API_KEY  = os.getenv("AI_API_KEY",  "")
AI_MODEL    = os.getenv("AI_MODEL",    "claude-opus-4-5")
AI_BASE_URL = os.getenv("AI_BASE_URL", "")            # only for provider=custom


# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────
def build_prompt(data: dict):
    subject   = data.get("subject",   "General")
    question  = data.get("question",  "")
    answer    = data.get("answer",    "")
    max_marks = data.get("max_marks", 10)

    # system = (
    #     "You are a rigorous academic evaluator and writing coach. "
    #     "Evaluate the student answer, then rewrite it as a genuinely better version the student could submit — "
    #     "same academic level, same voice, just correct and complete. "
    #     "Return ONLY valid JSON with no markdown fences and no preamble."
    # )
    system = (
    "You are a strict but fair academic evaluator.\n\n"

    "IMPORTANT RULES:\n"
    "1. If the student's answer is already fully correct, complete, and well-structured, "
    "you MUST award FULL marks.\n"
    "2. Do NOT reduce marks just to suggest improvements.\n"
    "3. Only deduct marks for actual errors, missing points, or poor explanation.\n"
    "4. Minor wording improvements should NOT reduce marks.\n"
    "5. If answer is perfect, say no significant weaknesses.\n\n"

    "After evaluation, you may rewrite the answer, but the rewrite should be OPTIONAL improvement, "
    "not evidence of mistakes.\n\n"

    "Return ONLY valid JSON."
)
    user = f"""Evaluate and improve this student answer.

Subject: {subject}
Question: {question}
Student Answer: {answer}
Maximum Marks: {max_marks}

Return ONLY this JSON:
{{
  "marks_awarded": <integer>,
  "max_marks": {max_marks},
  "percentage": <integer 0-100>,
  "grade": "<A|B|C|D|F>",
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<specific strength>", "..."],
  "weaknesses": ["<specific weakness>", "..."],
  "recommended_changes": [
    {{
      "original_phrase": "<exact phrase from student answer, or 'Missing section'>",
      "issue": "<what is wrong or missing>",
      "fix": "<exact replacement or addition>"
    }}
  ],
  "rewritten_answer": "<Full rewritten answer. Same voice as student. Clearly better, not suspiciously perfect. Prose unless question requires otherwise.>",
  "rewrite_delta": "<2-3 sentences on what changed between original and rewrite>"
}}"""

    return system, user


# ─────────────────────────────────────────────
# Provider callers
# ─────────────────────────────────────────────

def call_openai(system, user):
    base_url = AI_BASE_URL.strip() if AI_BASE_URL else "https://api.openai.com/v1"
    url = f"{base_url}/chat/completions"

    print("USING URL:", url)  # debug

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": AI_MODEL,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        },
        timeout=60,
    )

    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



def evaluate(data: dict) -> dict:
    system, user = build_prompt(data)
    dispatch = { "openai": call_openai}
    caller = dispatch.get(AI_PROVIDER.lower())
    if not caller:
        raise ValueError(f"Unknown AI_PROVIDER: {AI_PROVIDER!r}")
    raw = caller(system, user)
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(clean)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", provider=AI_PROVIDER, model=AI_MODEL)


@app.route("/evaluate", methods=["POST"])
def evaluate_route():
    try:
        data = request.get_json()
        missing = [f for f in ["question", "answer", "max_marks"] if not str(data.get(f, "")).strip()]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
        if not AI_API_KEY:
            return jsonify({"error": "AI_API_KEY not set — add it to your .env file"}), 500
        result = evaluate(data)
        return jsonify({"success": True, "result": result})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {e}"}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text[:300]}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
