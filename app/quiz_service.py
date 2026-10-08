"""Quiz generation: plan, prompt, sanitize, validate, retry.

The backend is the single decision point for every quiz. Mode, subject
domain, language, allowed question kinds and the exact material the LLM
sees are decided here (spec: int.md) - the prompt only describes the job.
Client-supplied `context` / `subject` never reach the model: topic_only
quizzes send zero notes, course quizzes send only completed-topic notes.

Adapted from int.md to this codebase's real names:
  request  -> app.schemas.QuizGenerateRequest (topic, kind, count, ...)
  question -> app.schemas.QuizQuestion (prompt, correct_index, ...)
  LLM      -> app.ai_service.complete(...) as call_llm / call_llm_small
  data     -> app.course_context.fetch_course_bundle / save_course_domain
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastapi import HTTPException
from pydantic import ValidationError

from app import ai_service
from app.course_context import fetch_course_bundle, save_course_domain, strip_html
from app.schemas import QuizGenerateRequest, QuizKind, QuizQuestion

# =====================================================================
# 1. DOMAIN + KINDS
# =====================================================================

VALID_DOMAINS = {"math", "english", "science", "humanities", "programming", "other"}
ALL_KINDS = ["mcq", "code", "debug", "output"]

CLASSIFY_SYSTEM = """You classify a quiz topic into exactly one subject area.
Reply with one word from this list and nothing else:
math, english, science, humanities, programming, other"""

CallLlmSmall = Callable[[str, str], Awaitable[str]]

_WORD_RE = re.compile(r"[a-z0-9+#']+")

# Keyword buckets per spec domain. Ties and misses fall through to the LLM.
DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "math": {
        "math", "mathematics", "algebra", "geometry", "calculus",
        "trigonometry", "equation", "equations", "fraction", "fractions",
        "percentage", "percentages", "theorem", "matrix", "matrices",
        "determinant", "derivative", "derivatives", "integral", "integrals",
        "probability", "statistics", "statistical", "median", "arithmetic",
        "coordinate", "coordinates", "slope", "polynomial", "exponent",
        "logarithm", "quadratic", "permutation", "permutations", "integers",
        "integrate", "factorize", "numeracy",
    },
    "science": {
        "science", "biology", "chemistry", "physics", "atom", "atoms",
        "molecule", "molecules", "organism", "organisms", "photosynthesis",
        "chlorophyll", "respiration", "gravity", "velocity", "acceleration",
        "enzyme", "enzymes", "genome", "dna", "chromosome", "reaction",
        "reactions", "acid", "alkaline", "ecosystem", "habitat", "climate",
        "planet", "planets", "orbit", "cell", "cells", "tissue", "tissues",
        "organ", "organs", "electron", "proton", "neutron", "compound",
        "electricity", "circuit", "bacteria", "virus", "species",
        "evolution", "molecular", "atomic", "force", "friction", "magnet",
        "energy", "temperature",
    },
    "english": {
        "english", "grammar", "tense", "tenses", "noun", "nouns", "verb",
        "verbs", "adjective", "adjectives", "adverb", "adverbs",
        "preposition", "prepositions", "pronoun", "pronouns", "vocabulary",
        "spelling", "synonym", "synonyms", "antonym", "antonyms", "essay",
        "essays", "paragraph", "paragraphs", "comprehension", "sentence",
        "sentences", "phrase", "phrases", "clause", "clauses", "punctuation",
        "literature", "novel", "novels", "poem", "poems", "poetry", "poet",
        "prose", "narrative", "metaphor", "simile", "alliteration", "prefix",
        "suffix", "author", "chapter", "chapters", "dialogue", "narrator",
        "rhyme", "rhyming",
    },
    "humanities": {
        "history", "historical", "ancient", "medieval", "war", "wars",
        "revolution", "revolutions", "empire", "empires", "civilization",
        "civilisation", "dynasty", "treaty", "battle", "battles", "emperor",
        "republic", "colonial", "colonialism", "archaeology", "monarchy",
        "parliament", "pharaoh", "feudal",
        "islam", "islamic", "quran", "koran", "hadith", "fiqh", "tafseer",
        "salah", "zakat", "surah", "prophet", "imam", "sunnah",
        "urdu", "ghazal", "nazm", "qawaid", "takhallus", "shayari", "adab",
    },
    "programming": {
        "python", "javascript", "typescript", "css", "html", "react", "vue",
        "angular", "node", "npm", "django", "flask", "sql", "mysql", "sqlite",
        "java", "kotlin", "swift", "programming", "programmer", "coding",
        "code", "coder", "function", "variable", "variables", "array",
        "arrays", "loop", "loops", "recursion", "algorithm", "algorithms",
        "compiler", "interpreter", "syntax", "debug", "debugging", "git",
        "regex", "json", "boolean", "oop", "inheritance", "pointer",
        "pointers", "database", "frontend", "backend", "framework", "api",
        "console", "callback", "callbacks", "async", "await", "lambda",
        "datatype", "datatypes", "integer", "exception", "exceptions",
        "module", "modules", "browser", "developer", "development", "web",
        "website", "websites", "script", "scripts", "def",
    },
}


def _keyword_domain(text: str) -> str | None:
    """Domain of `text` from keywords; None when unclear or tied."""
    text = (text or "").strip()
    if not text:
        return None
    if URDU_RE.search(text):
        return "humanities"
    tokens = set(_WORD_RE.findall(text.lower()))
    best: str | None = None
    best_score = 0
    tied = False
    for domain, words in DOMAIN_KEYWORDS.items():
        score = len(tokens & words)
        if score > best_score:
            best, best_score, tied = domain, score, False
        elif score and score == best_score:
            tied = True
    return None if tied or not best_score else best


async def classify_domain(text: str, call_llm_small: CallLlmSmall) -> str:
    """One of VALID_DOMAINS for `text`: keywords first, then one tiny LLM
    call for unclear titles; "other" when both paths fail."""
    quick = _keyword_domain(text)
    if quick:
        return quick
    if not (text or "").strip():
        return "other"
    try:
        raw = await call_llm_small(CLASSIFY_SYSTEM, f"Topic: {text[:500]}")
        word = raw.strip().lower().split()[0].strip(".,\"'")
        return word if word in VALID_DOMAINS else "other"
    except Exception:  # noqa: BLE001 - classification never breaks a quiz
        return "other"


def kinds_for(domain: str, requested: str | None) -> list[str]:
    """Question kinds the backend allows for this quiz. Never the UI's say:
    non-programming is always MCQ-only, even when "mixed" was requested."""
    if domain != "programming":
        return ["mcq"]
    req = (requested or "").strip().lower()
    if req in ("mixed", "", "none"):
        return list(ALL_KINDS)
    if req in ALL_KINDS:
        return [req]
    return list(ALL_KINDS)


def style_for(domain: str, requested: str | None) -> str:
    """Style guidance for the prompt. The UI fans "mixed" out into
    single-kind requests, so the backend never sees the word "mixed"; every
    non-programming quiz is MCQ-only, so the variety hint applies to all of
    them (spec: non-programming + mixed -> vary the MCQ styles)."""
    if domain != "programming":
        return (
            "Vary the styles: conceptual questions, calculations or "
            "applications, true/false-style statements, and short word "
            "problems. All as 4-option MCQs."
        )
    return "Keep a natural balance of question styles."


# =====================================================================
# 2. LANGUAGE (decided by backend, never inferred by the LLM)
# =====================================================================

URDU_RE = re.compile("[؀-ۿݐ-ݿࢠ-ࣿ]")
NATURAL_LANGUAGES = {"english", "urdu"}


def resolve_language(requested: str | None, text: str) -> str:
    """Quiz language: an explicit english/urdu request wins, otherwise the
    script of `text` decides. The request's `language` field carries the
    preferred *coding* language (e.g. "python") and never counts here."""
    req = (requested or "").strip().lower()
    if req in NATURAL_LANGUAGES:
        return req.title()
    return "Urdu" if URDU_RE.search(text or "") else "English"


# =====================================================================
# 3. NOTE FILTERING (used only in course / course_focus modes)
# =====================================================================

CODE_PATTERNS = re.compile(
    r"(```|\bdef\s+\w+\(|\bimport\s+\w+|\bprint\(|\bfor\s+\w+\s+in\s|\bclass\s+\w+|=>|</?\w+>)",
    re.MULTILINE,
)


def looks_like_code(text: str) -> bool:
    return len(CODE_PATTERNS.findall(text or "")) >= 2


def pick_notes(
    notes: list[dict],
    completed_ids: list[Any],
    focus: str,
    domain: str,
    max_notes: int = 8,
    max_chars: int = 6000,
) -> str:
    """Only notes of completed topics survive; for non-programming quizzes
    code-looking notes are dropped as well. A focus text ranks matching
    notes first. `notes` already come scoped to this course."""
    done = {int(i) for i in completed_ids if isinstance(i, (int, str)) and str(i).isdigit()}
    pool: list[tuple[str, str]] = []
    for row in notes:
        raw_id = row.get("topic_id")
        try:
            topic_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            topic_id = None
        if topic_id not in done:
            continue
        title = str(row.get("title") or "").strip() or "Untitled"
        body = strip_html(str(row.get("content") or ""))
        pool.append((title, body))

    if domain != "programming":
        pool = [
            (title, body)
            for title, body in pool
            if not looks_like_code(f"{title}\n{body}")
        ]

    if focus:
        words = [w for w in re.findall(r"\w+", focus.lower()) if len(w) > 2]

        def score(item: tuple[str, str]) -> int:
            blob = f"{item[0]} {item[1]}".lower()
            return sum(1 for w in words if w in blob)

        pool.sort(key=score, reverse=True)

    out: list[str] = []
    total = 0
    for title, body in pool[:max_notes]:
        chunk = f"[{title}]\n{body}".strip()[: max_chars - total]
        if not chunk:
            break
        out.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(out) if out else "No notes provided."


# =====================================================================
# 4. QUIZ PLAN (single source of truth for one generation)
# =====================================================================


@dataclass
class QuizPlan:
    mode: str  # topic_only | course | course_focus | pdf
    topic: str  # topic_only: user topic | course: course title | pdf: "Uploaded PDF"
    domain: str
    kinds: list[str]
    language: str
    notes_text: str = ""
    roadmap_topics: list[str] = field(default_factory=list)
    focus: str = ""
    style_hint: str = ""
    coding_language: str = "python"


async def build_plan(
    req: QuizGenerateRequest,
    user_token: str | None,
    pdf_text: str | None,
    call_llm_small: CallLlmSmall,
) -> QuizPlan:
    """Decide everything the LLM is allowed to know for this request.

    Mode order (spec section 1, adapted): a chosen course wins, then a real
    PDF upload, then a typed topic. The PDF endpoint fills `topic` with the
    uploaded file's name, so the PDF must be checked before the topic or a
    PDF upload would never be used. The client's `context` / `subject` are
    never read: material comes only from the server-side course bundle or
    the uploaded PDF.
    """
    text = (req.topic or "").strip()
    requested = req.kind.value if req.kind else "mixed"
    coding_language = (req.language or "python").strip() or "python"

    # B/C) Course selected: completed topics -> their notes; the typed text
    # is a FOCUS instruction, never the topic or domain.
    if req.course_id is not None:
        bundle = await fetch_course_bundle(req.course_id, user_token)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Course not found.")
        course = bundle["course"]
        title = str(course.get("title") or "").strip() or f"Course {req.course_id}"
        domain = str(course.get("domain") or "").strip().lower()
        if domain not in VALID_DOMAINS:
            domain = await classify_domain(title, call_llm_small)
            await save_course_domain(req.course_id, domain, user_token)
        completed = bundle["completed_titles"]
        if not completed:
            raise HTTPException(
                status_code=400,
                detail="No completed topics in this course yet.",
            )
        notes_text = pick_notes(
            bundle["notes"], bundle["completed_ids"], text, domain
        )
        return QuizPlan(
            mode="course_focus" if text else "course",
            topic=title,
            domain=domain,
            kinds=kinds_for(domain, requested),
            language=resolve_language(req.language, title),
            notes_text=notes_text,
            roadmap_topics=completed,
            focus=text,
            style_hint=style_for(domain, requested),
            coding_language=coding_language,
        )

    # D) Uploaded PDF: its text is the whole source.
    if pdf_text:
        domain = await classify_domain(pdf_text[:2000], call_llm_small)
        return QuizPlan(
            mode="pdf",
            topic="Uploaded PDF",
            domain=domain,
            kinds=kinds_for(domain, requested),
            language=resolve_language(req.language, pdf_text[:500]),
            notes_text=pdf_text[:8000],
            style_hint=style_for(domain, requested),
            coding_language=coding_language,
        )

    # A) General + typed topic: the topic ONLY - no notes, no roadmap, no
    # course data ever leaves the server for this mode.
    if text:
        domain = await classify_domain(text, call_llm_small)
        return QuizPlan(
            mode="topic_only",
            topic=text,
            domain=domain,
            kinds=kinds_for(domain, requested),
            language=resolve_language(req.language, text),
            style_hint=style_for(domain, requested),
            coding_language=coding_language,
        )

    raise HTTPException(
        status_code=400, detail="Write a topic, choose a course, or upload a PDF."
    )


# =====================================================================
# 5. PROMPTS (positive instructions only; inputs arrive as fixed values)
# =====================================================================

SYSTEM_TEMPLATE = r"""You are ATLAS Exam Writer, a calm and precise quiz author for a learning platform.
You write clear, accurate quizzes on one subject at a time.

<assignment>
Mode: <<MODE>>
Subject area: <<SUBJECT_DOMAIN>>
Topic or course: <<TOPIC>>
Quiz language: <<LANGUAGE>>
Number of questions: <<NUM_QUESTIONS>>
Difficulty: <<DIFFICULTY>>
Allowed question kinds: <<ALLOWED_KINDS>>
Style guidance: <<STYLE_HINT>>
<<CODING_LANGUAGE_LINE>>
</assignment>

<mode_guide>
topic_only: The topic above is your whole brief. Write the quiz from standard curriculum knowledge of that topic.
course: The learner is studying the course above. The completed roadmap topics and the learner's notes appear in the user message. Cover those topics, and use the notes to match the depth and wording the learner studied.
course_focus: Same as course, and the learner also gave a focus request in <focus>. Choose the roadmap topics and notes that match the focus, and build most of the quiz around it. If the focus names something outside the course, cover it within the course's subject area.
pdf: The PDF text in the user message is your source. Cover its main ideas.
</mode_guide>

<how_you_work>
1. Every question belongs to the subject area above and the material for this mode.
2. You write every word of the quiz (questions, options, explanations) in <<LANGUAGE>>.
3. You write mathematics in plain text with Unicode symbols: 1/2, 3 × 4, √16, x², π, ≤, ≥, ≠.
4. Your reply is one JSON object and nothing else: it starts with { and ends with }.
   There are no code fences and no text outside it.
   Inside strings, the only backslash sequences are \" and \n.
5. Each multiple-choice question has exactly four options and exactly one correct answer.
   Wrong options are plausible. The explanation is one or two short sentences.
6. Difficulty "easy" tests recall and direct application, "medium" tests application and
   short reasoning, "hard" tests multi-step reasoning and tricky distinctions.
</how_you_work>

<output_format>
<<OUTPUT_FORMAT>>
</output_format>

You always deliver the full quiz. If the material is thin, you complete it with
standard knowledge of the subject."""

_OUTPUT_SHAPES: dict[str, str] = {
    "mcq": (
        "{\n"
        '      "kind": "mcq",\n'
        '      "prompt": "text of the question",\n'
        '      "options": ["first", "second", "third", "fourth"],\n'
        '      "correct_index": 0,\n'
        '      "explanation": "why the answer is correct"\n'
        "    }"
    ),
    "code": (
        "{\n"
        '      "kind": "code",\n'
        '      "prompt": "text of the task",\n'
        '      "starter_code": "the skeleton to complete, or an empty string",\n'
        '      "language": "<<LANG>>",\n'
        '      "reference_solution": "a working solution",\n'
        '      "rubric": ["what a correct solution does"],\n'
        '      "explanation": "one or two sentences"\n'
        "    }"
    ),
    "debug": (
        "{\n"
        '      "kind": "debug",\n'
        '      "prompt": "text of the task",\n'
        '      "code": "code containing one deliberate bug",\n'
        '      "buggy_line": 3,\n'
        '      "explanation": "one or two sentences"\n'
        "    }"
    ),
    "output": (
        "{\n"
        '      "kind": "output",\n'
        '      "prompt": "text of the task",\n'
        '      "code": "code to read",\n'
        '      "correct_output": "the exact output, verbatim",\n'
        '      "explanation": "one or two sentences"\n'
        "    }"
    ),
}


def build_output_format(plan: QuizPlan) -> str:
    """Only the shapes whose kind is allowed for this quiz - a math quiz is
    never shown a code question shape."""
    shapes = [
        _OUTPUT_SHAPES[kind].replace("<<LANG>>", plan.coding_language)
        for kind in plan.kinds
        if kind in _OUTPUT_SHAPES
    ]
    body = ",\n    ".join(shapes)
    return (
        '{\n  "questions": [\n    ' + body + "\n  ]\n}\n"
        "Each object carries only the fields shown for its kind, "
        "including every field of that kind."
    )


def build_system_prompt(plan: QuizPlan, n: int, difficulty: str) -> str:
    coding_line = ""
    if set(plan.kinds) & {"code", "debug", "output"}:
        coding_line = f"Coding language for code questions: {plan.coding_language}"
    return (
        SYSTEM_TEMPLATE
        .replace("<<MODE>>", plan.mode)
        .replace("<<SUBJECT_DOMAIN>>", plan.domain)
        .replace("<<TOPIC>>", plan.topic)
        .replace("<<LANGUAGE>>", plan.language)
        .replace("<<NUM_QUESTIONS>>", str(n))
        .replace("<<DIFFICULTY>>", difficulty)
        .replace("<<ALLOWED_KINDS>>", ", ".join(plan.kinds))
        .replace("<<STYLE_HINT>>", plan.style_hint)
        .replace("<<CODING_LANGUAGE_LINE>>", coding_line)
        .replace("<<OUTPUT_FORMAT>>", build_output_format(plan))
    )


def build_user_message(plan: QuizPlan) -> str:
    parts = ["Create the quiz now."]
    if plan.mode in ("course", "course_focus"):
        parts.append(
            "<completed_roadmap_topics>\n"
            + "\n".join(plan.roadmap_topics)
            + "\n</completed_roadmap_topics>"
        )
        parts.append(f"<notes>\n{plan.notes_text}\n</notes>")
    if plan.mode == "course_focus":
        parts.append(f"<focus>\n{plan.focus}\n</focus>")
    if plan.mode == "pdf":
        parts.append(f"<pdf_text>\n{plan.notes_text}\n</pdf_text>")
    # topic_only: nothing else. No notes block at all.
    return "\n\n".join(parts)


# =====================================================================
# 6. JSON SANITIZER (fixes the 502s: LaTeX backslashes break json.loads)
# =====================================================================

LATEX_MAP = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±", r"\pi": "π",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\ne": "≠",
    r"\approx": "≈", r"\infty": "∞", r"\theta": "θ", r"\alpha": "α", r"\beta": "β",
    r"\degree": "°", r"\circ": "°", r"\rightarrow": "→", r"\to": "→", r"\Delta": "Δ",
}
_LATEX_KEYS = sorted(LATEX_MAP, key=len, reverse=True)
_LATEX_RE = re.compile("|".join(re.escape(k) + r"(?![A-Za-z])" for k in _LATEX_KEYS))


def _frac(match: re.Match) -> str:
    a, b = match.group(1).strip(), match.group(2).strip()

    def wrap(part: str) -> str:
        return part if re.fullmatch(r"[\w.]+", part) else f"({part})"

    return f"{wrap(a)}/{wrap(b)}"


def sanitize_llm_json(raw: str) -> str:
    """Make an LLM reply parseable: strip fences, convert LaTeX to Unicode,
    then drop every backslash that is not a legitimate JSON escape."""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    s = s[start : end + 1]

    # LaTeX BEFORE escape cleanup: \frac and \times look like valid JSON
    # escapes (\f, \t) but are not.
    for _ in range(3):
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _frac, s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", s)
    s = re.sub(r"\\sqrt\b", "√", s)
    s = _LATEX_RE.sub(lambda m: LATEX_MAP[m.group(0)], s)
    s = re.sub(r"\\[()\[\]]", "", s)
    s = s.replace("$", "")
    s = re.sub(r"\^\{?2\}?", "²", s)
    s = re.sub(r"\^\{?3\}?", "³", s)

    # Drop any backslash that is not a legitimate JSON escape.
    s = re.sub(r'\\(?!["\\/nrt]|u[0-9a-fA-F]{4})', "", s)
    return s


# =====================================================================
# 7. VALIDATION
# =====================================================================


def validate_quiz(data: Any, plan: QuizPlan, n: int) -> list[QuizQuestion]:
    """Parse-checked, rule-checked questions - or ValueError (retry path)."""
    if not isinstance(data, dict):
        raise ValueError("the reply was not a JSON object")
    raw = data.get("questions")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected a non-empty 'questions' array")
    if len(raw) < max(1, n - 1):
        raise ValueError(f"expected about {n} questions, got {len(raw)}")

    questions: list[QuizQuestion] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"question {index} is not an object")
        payload = dict(item)
        kind = str(payload.get("kind") or "").strip().lower()
        if not kind and len(plan.kinds) == 1:
            kind = plan.kinds[0]
        if kind not in plan.kinds:
            raise ValueError(
                f"question {index}: kind '{kind or 'missing'}' is not allowed "
                f"for {plan.domain} (allowed: {', '.join(plan.kinds)})"
            )
        payload["kind"] = kind
        if kind == "code" and not str(payload.get("language") or "").strip():
            payload["language"] = plan.coding_language
        try:
            question = QuizQuestion.model_validate(payload)
        except ValidationError as exc:
            errors = exc.errors()
            detail = str(errors[0].get("msg", "invalid")) if errors else "invalid"
            raise ValueError(f"question {index}: {detail}") from exc
        _validate_question(question, plan, index)
        questions.append(question)
    return questions


def _validate_question(q: QuizQuestion, plan: QuizPlan, index: int) -> None:
    blob = " ".join(
        part
        for part in [
            q.prompt,
            q.code,
            q.starter_code,
            " ".join(q.options),
            q.explanation,
            q.reference_solution,
            q.correct_output,
        ]
        if part
    )

    if q.kind is QuizKind.MCQ:
        if len(q.options) != 4:
            raise ValueError(f"question {index}: an MCQ needs exactly 4 options")
        if q.correct_index is None or not 0 <= q.correct_index < 4:
            raise ValueError(
                f"question {index}: correct_index must be 0, 1, 2 or 3"
            )
    elif q.kind is QuizKind.CODE:
        if not q.reference_solution.strip():
            raise ValueError(
                f"question {index}: a code question needs a reference_solution"
            )
    elif q.kind is QuizKind.DEBUG:
        if not q.code.strip():
            raise ValueError(f"question {index}: a debug question needs code")
        if q.buggy_line is None or q.buggy_line < 1:
            raise ValueError(
                f"question {index}: buggy_line must be a 1-based line number"
            )
    elif q.kind is QuizKind.OUTPUT:
        if not q.code.strip() or not q.correct_output.strip():
            raise ValueError(
                f"question {index}: an output question needs code and correct_output"
            )

    if plan.domain != "programming" and (
        q.code.strip()
        or q.starter_code.strip()
        or "```" in blob
        or looks_like_code(blob)
    ):
        raise ValueError(f"question {index}: code content in a non-programming quiz")

    has_urdu = bool(URDU_RE.search(blob))
    if plan.language == "English" and has_urdu:
        raise ValueError(f"question {index}: Urdu text in an English quiz")
    if plan.language == "Urdu" and not has_urdu:
        raise ValueError(f"question {index}: English-only text in an Urdu quiz")


# =====================================================================
# 8. GENERATION WITH RETRY
# =====================================================================


async def generate_quiz(
    plan: QuizPlan,
    n: int,
    difficulty: str,
    call_llm: Callable[[str, str], Awaitable[ai_service.AIResult]],
) -> tuple[list[QuizQuestion], str]:
    """call_llm(system, user) -> AIResult. Up to 3 attempts: sanitize ->
    parse -> validate, with a short nudge naming the last rejection. Any
    failure ends as RuntimeError, which the route turns into a clean 503."""
    system = build_system_prompt(plan, n, difficulty)
    user = build_user_message(plan)

    last_err = ""
    result: ai_service.AIResult | None = None
    for _ in range(3):
        nudge = ""
        if last_err:
            nudge = (
                f"\n\nYour previous reply could not be used ({last_err}). "
                f"Please send the complete quiz again as one JSON object, in "
                f"{plan.language}, using only these kinds: {', '.join(plan.kinds)}."
            )
        result = await call_llm(system, user + nudge)
        try:
            data = json.loads(sanitize_llm_json(result.text))
            return validate_quiz(data, plan, n), result.model
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_err = str(exc)[:200]
            print("QUIZ_REJECTED:", last_err)
    raise RuntimeError(f"Quiz generation failed: {last_err}")
