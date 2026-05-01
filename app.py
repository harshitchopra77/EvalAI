import os
import json
import requests
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic")   # anthropic | openai
AI_API_KEY  = os.getenv("AI_API_KEY",  "")
AI_MODEL    = os.getenv("AI_MODEL",    "claude-opus-4-5")
AI_BASE_URL = os.getenv("AI_BASE_URL", "")


# ─────────────────────────────────────────────────────────────────────────────
# File text extraction  (.txt | .docx | .pdf)
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(file) -> str:
    filename = file.filename.lower()
    raw = file.read()

    if filename.endswith(".txt"):
        return raw.decode("utf-8", errors="replace").strip()

    elif filename.endswith(".docx"):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        doc = DocxDocument(BytesIO(raw))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    elif filename.endswith(".pdf"):
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n\n".join(parts).strip()
        except ImportError:
            pass
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(raw))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n\n".join(parts).strip()
        except ImportError:
            raise RuntimeError("No PDF library. Run: pip install pdfplumber")

    else:
        raise ValueError("Unsupported file type. Upload .txt, .docx, or .pdf")


def extract_text_from_field(form_text: str, file_field) -> str:
    """Return text from file if uploaded, else from form text field."""
    if file_field and file_field.filename:
        return extract_text(file_field)
    return form_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
def build_single_prompt(data: dict):
    subject        = data.get("subject",        "General")
    question       = data.get("question",       "")
    answer         = data.get("answer",         "")
    teacher_answer = data.get("teacher_answer", "").strip()
    max_marks      = data.get("max_marks",      10)
    has_teacher    = bool(teacher_answer)

    system = (
        "You are a strict but fair academic evaluator.\n\n"
        "RULES:\n"
        "1. Award FULL marks when the answer is fully correct and complete.\n"
        "2. Only deduct marks for actual errors or missing key points.\n"
        "3. Minor wording issues do NOT reduce marks.\n"
        + ("4. A teacher reference answer is provided — compare carefully.\n" if has_teacher else
           "4. Use your own subject knowledge to evaluate.\n")
        + "\nReturn ONLY valid JSON. No markdown fences, no preamble."
    )

    teacher_section = f"\nTeacher's Reference Answer:\n{teacher_answer}\n" if has_teacher else ""

    user = f"""Evaluate this student answer.

Subject: {subject}
Question: {question}{teacher_section}
Student Answer: {answer}
Maximum Marks: {max_marks}

Return ONLY this JSON:
{{
  "marks_awarded": <integer>,
  "max_marks": {max_marks},
  "percentage": <integer 0-100>,
  "grade": "<A|B|C|D|F>",
  "teacher_answer_used": {"true" if has_teacher else "false"},
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength>"],
  "weaknesses": ["<weakness>"],
  "recommended_changes": [
    {{
      "original_phrase": "<phrase or 'Missing section'>",
      "issue": "<what is wrong>",
      "fix": "<correction>"
    }}
  ],
  "rewritten_answer": "<improved answer in student's voice>",
  "rewrite_delta": "<2-3 sentences on what changed>"
}}"""

    return system, user


def build_sheet_prompt(subject: str, question_paper: str, answer_sheet: str,
                       reference: str, question_config: list, total_marks_hint: str):
    """
    question_config: list of dicts like [{"number": "Q1", "marks": 10}, ...]
                     May be empty — AI assigns marks in that case.
    question_paper:  text of the question paper (may be empty if not provided).
    """
    has_ref     = bool(reference.strip())
    has_qpaper  = bool(question_paper.strip())
    has_config  = bool(question_config)
    has_total   = bool(total_marks_hint.strip())

    # Build question config section
    if has_config:
        config_lines = "\n".join(
            f"  {q.get('number','Q'+str(i+1))}: {q.get('marks','?')} marks"
            for i, q in enumerate(question_config)
        )
        computed_total = sum(int(q.get("marks", 0)) for q in question_config if str(q.get("marks","")).isdigit())
        marks_section = f"\nPer-question marks allocation:\n{config_lines}\nTotal available: {computed_total}"
    elif has_total:
        marks_section = f"\nTotal marks available: {total_marks_hint} (distribute proportionally across questions)"
    else:
        marks_section = "\nNo marks specified — assign appropriate marks to each question based on complexity."

    system = (
        "You are a strict but fair academic examiner evaluating a full student answer sheet.\n\n"
        "RULES:\n"
        "1. Identify EVERY question and its answer.\n"
        "2. Evaluate each independently.\n"
        "3. Award FULL marks when an answer is fully correct.\n"
        "4. Deduct only for genuine errors, missing points, or poor explanation.\n"
        "5. Minor wording issues do NOT reduce marks.\n"
        + ("6. A question paper is provided — use it to understand exact questions.\n" if has_qpaper else
           "6. Detect questions from the answer sheet itself.\n")
        + ("7. A teacher model-answer reference is provided — compare each answer carefully.\n" if has_ref else
           "7. Use your own subject knowledge to evaluate.\n")
        + ("8. Per-question mark allocations are specified — award marks strictly within those limits.\n" if has_config else
           "8. Assign marks based on question complexity.\n")
        + "\nReturn ONLY valid JSON. No markdown fences, no preamble."
    )

    qpaper_section = f"\n\n--- QUESTION PAPER ---\n{question_paper.strip()}\n--- END QUESTION PAPER ---" if has_qpaper else ""
    ref_section    = f"\n\n--- TEACHER REFERENCE / MODEL ANSWERS ---\n{reference.strip()}\n--- END REFERENCE ---" if has_ref else ""

    user = f"""Evaluate the entire student answer sheet.

Subject: {subject}{marks_section}{qpaper_section}{ref_section}

--- STUDENT ANSWER SHEET ---
{answer_sheet.strip()}
--- END ANSWER SHEET ---

Instructions:
- Match each answer to its question. Use the question paper if provided, otherwise detect questions from the answer sheet.
- Respect the per-question mark limits if given.
- Evaluate every question-answer pair and sum up totals.

Return ONLY this JSON:
{{
  "subject": "{subject}",
  "total_marks_awarded": <integer>,
  "total_marks_available": <integer>,
  "total_percentage": <integer 0-100>,
  "overall_grade": "<A|B|C|D|F>",
  "overall_summary": "<3-4 sentence summary of the whole paper>",
  "teacher_reference_used": {"true" if has_ref else "false"},
  "question_paper_used": {"true" if has_qpaper else "false"},
  "questions": [
    {{
      "question_number": "<Q1 / 1a / Part A etc.>",
      "question_text": "<question as detected>",
      "student_answer_summary": "<brief summary of what student wrote>",
      "marks_awarded": <integer>,
      "marks_available": <integer>,
      "percentage": <integer 0-100>,
      "grade": "<A|B|C|D|F>",
      "feedback": "<specific feedback>",
      "strengths": ["<strength>"],
      "weaknesses": ["<weakness>"],
      "model_answer_snippet": "<key points expected>"
    }}
  ]
}}"""

    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# AI callers
# ─────────────────────────────────────────────────────────────────────────────
def call_anthropic(system: str, user: str) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": AI_API_KEY, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json={"model": AI_MODEL, "max_tokens": 4096, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def call_openai(system: str, user: str) -> str:
    base_url = AI_BASE_URL.strip() if AI_BASE_URL else "https://api.openai.com/v1"
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"},
        json={"model": AI_MODEL, "max_tokens": 4096,
              "messages": [{"role": "system", "content": system},
                           {"role": "user",   "content": user}]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_ai(system: str, user: str) -> str:
    if AI_PROVIDER.lower() == "anthropic":
        return call_anthropic(system, user)
    elif AI_PROVIDER.lower() == "openai":
        return call_openai(system, user)
    raise ValueError(f"Unknown AI_PROVIDER: {AI_PROVIDER!r}")


def parse_json(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1].lstrip("json").strip() if len(parts) >= 3 else clean.lstrip("`")
    return json.loads(clean)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model=AI_MODEL, provider=AI_PROVIDER)


@app.route("/config")
def config():
    return jsonify({"model": AI_MODEL, "provider": AI_PROVIDER})


@app.route("/evaluate", methods=["POST"])
def evaluate_route():
    """Single-question evaluation. Accepts JSON or multipart/form-data."""
    try:
        if not AI_API_KEY:
            return jsonify({"error": "AI_API_KEY not set"}), 500

        is_multipart = request.content_type and "multipart/form-data" in request.content_type

        if is_multipart:
            subject    = request.form.get("subject",    "General")
            max_marks  = int(request.form.get("max_marks", 10))

            # Question: text field OR uploaded file
            question = extract_text_from_field(
                request.form.get("question", ""),
                request.files.get("question_file")
            )
            if not question:
                return jsonify({"error": "Question is required."}), 400

            # Teacher reference: text field OR file
            teacher_answer = extract_text_from_field(
                request.form.get("teacher_answer", ""),
                request.files.get("teacher_file")
            )

            # Student answer: text field OR file
            answer_file = request.files.get("answer_file")
            if answer_file and answer_file.filename:
                answer = extract_text(answer_file)
            else:
                answer = request.form.get("answer", "").strip()

            if not answer:
                return jsonify({"error": "Student answer is required."}), 400

            data = {"subject": subject, "max_marks": max_marks,
                    "question": question, "answer": answer,
                    "teacher_answer": teacher_answer}

        else:
            data = request.get_json()
            missing = [f for f in ["question", "answer", "max_marks"]
                       if not str(data.get(f, "")).strip()]
            if missing:
                return jsonify({"error": f"Missing: {', '.join(missing)}"}), 400

        system, user = build_single_prompt(data)
        result = parse_json(call_ai(system, user))
        return jsonify({"success": True, "mode": "single", "result": result})

    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {e}"}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text[:300]}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate-sheet", methods=["POST"])
def evaluate_sheet_route():
    """
    Full answer-sheet evaluation.
    Form fields:
      answer_file      – student answer sheet (required)
      question_file    – question paper file  (optional)
      question_text    – question paper text  (optional, ignored if question_file set)
      teacher_file     – model answer file    (optional)
      teacher_text     – model answer text    (optional, ignored if teacher_file set)
      subject          – subject name
      total_marks      – overall total        (optional)
      question_config  – JSON array of {number, marks} per question (optional)
    """
    try:
        if not AI_API_KEY:
            return jsonify({"error": "AI_API_KEY not set"}), 500

        subject     = request.form.get("subject",     "General")
        total_marks = request.form.get("total_marks", "").strip()

        # Parse per-question config from frontend
        q_config_raw = request.form.get("question_config", "[]")
        try:
            question_config = json.loads(q_config_raw)
        except Exception:
            question_config = []

        # Student answer sheet — required
        answer_file = request.files.get("answer_file")
        if not answer_file or not answer_file.filename:
            return jsonify({"error": "Please upload a student answer sheet."}), 400
        answer_sheet = extract_text(answer_file)
        if not answer_sheet:
            return jsonify({"error": "Student answer sheet is empty."}), 400

        # Question paper — optional
        question_paper = extract_text_from_field(
            request.form.get("question_text", ""),
            request.files.get("question_file")
        )

        # Teacher reference — optional
        reference = extract_text_from_field(
            request.form.get("teacher_text", ""),
            request.files.get("teacher_file")
        )

        system, user = build_sheet_prompt(
            subject, question_paper, answer_sheet,
            reference, question_config, total_marks
        )
        result = parse_json(call_ai(system, user))
        return jsonify({"success": True, "mode": "sheet", "result": result})

    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {e}"}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"API error {e.response.status_code}: {e.response.text[:300]}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)