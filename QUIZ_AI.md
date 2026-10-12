# Quiz AI — Every Prompt + Backend Flow

Single reference for **every prompt the quiz AI sees** (directly or indirectly)
and how the backend assembles, sends, validates and retries them.

**Source files**

| Concern | File |
| --- | --- |
| System prompt, plan, sanitize, validate, retry | `app/quiz_service.py` |
| Endpoints, LLM wrappers, evaluate dispatch | `app/routers/quizzes.py` |
| Evaluate + JSON rules (shared prompts) | `app/prompts.py` |
| Model routing, fallback, temperature, tokens | `app/ai_service.py` |
| Request/response models | `app/schemas.py` |
| Bearer auth | `app/deps.py` |

---

## 1. Model routing (what actually executes the prompts)

| Call | Task | Provider → model | Fallback | Temperature | Max tokens |
| --- | --- | --- | --- | --- | --- |
| Quiz generation (`_call_llm`) | `QUIZ` | Groq → `qwen/qwen3.8-27b` | Groq → `openai/gpt-oss-20b` | **0.3** (override) | 4096 |
| Domain classification fallback (`_call_llm_small`) | `QUIZ` | Groq → `qwen/qwen3.8-27b` | Groq → `openai/gpt-oss-20b` | **0.0** | **40** |
| Code answer evaluation | `CODE` | Groq → `qwen/qwen3.8-27b` | Groq → `openai/gpt-oss-20b` | 0.2 | 2048 |

- Fallback chain runs inside `ai_service.complete()` (candidates de-duped, first success wins).
- MCQ evaluation is **local** (`_evaluate_mcq`) — no AI call at all.
- Messages are always exactly two: `system` + `user` (plus an appended retry nudge on the user message).
- Rule 2 title/topic detection (`detect_dev_language`) and keyword domain hits
  make **no LLM call at all** — the classifier only runs on unclear titles.

---

## 2. Prompt A — Domain classification (indirect, fallback only)

Two simple decisions, no heavyweight classifier:

### Rule 2 — title/topic detection (pure code, no LLM, no DB column)

`detect_dev_language(text)` matches the title/topic/PDF head against a fixed
marker list with word boundaries (`(?<![a-z0-9])marker(?![a-z])`, lowercased):

```
javascript, typescript, next.js, powershell, golang, python, html, css,
react, rust, java, kotlin, ruby, swift, c++, c#, bash, shell, cmd,
terminal, node, sql, php, go
```

- Aliases: `cmd` / `shell` / `terminal` / `powershell` → **bash**,
  `node` / `next.js` → **javascript**, `golang` → **go**.
- Boundaries matter: `java` never matches inside `javascript`, `react` never
  matches inside `reaction`. The `(?![a-z])` lookahead makes the longest
  marker win (`javascript` over `java`, `golang` over `go`).
- Returns the language (`"python"`, `"html"`, `"bash"`, …) or `None`.

`_domain_for(text, call_llm_small)`: a marker hit **is** the domain
(`"programming"`, no LLM call); otherwise the baseline `classify_domain` runs.

### Domain classification (`classify_domain`)

Keyword shortcut first (`_keyword_domain` + `DOMAIN_KEYWORDS`): a clear,
untied match is final with no LLM call. Misses fire one tiny LLM call.
Exceptions never break a quiz (`other`).

**System (`CLASSIFY_SYSTEM`, `quiz_service.py`):**

```text
You classify a quiz topic into exactly one subject area.
Reply with one word from this list and nothing else:
math, english, science, humanities, programming, other
```

**User:** `Topic: {text[:500]}`

### Where each decision runs (`build_plan`)

| Branch | Domain | Coding language |
| --- | --- | --- |
| Course | title marker → `programming` (no LLM, nothing saved, stored domain ignored); else stored `course.domain`, classified + `save_course_domain` lazily when missing | `detect_dev_language(title)` or `""` |
| PDF | `_domain_for(pdf_text[:2000])` | `detect_dev_language(pdf_text[:2000])` or `""` |
| Typed topic | `_domain_for(topic)` | `detect_dev_language(topic)` or `""` |

**Coding language priority — nothing else ever counts:**

1. what the **title/topic/PDF head names** (Rule 2),
2. empty otherwise (no request fallback, no `"python"` default, no DB column).

---

## 3. Prompt B — Quiz generation system prompt (direct, main)

Built by `build_system_prompt(plan, n, difficulty)` from **`SYSTEM_TEMPLATE`** —
verbatim template:

```text
CRITICAL RULES:
1. CONTEXT: Only use the provided course notes. If no course is selected, use only your general knowledge of the topic. Never mix courses.
2. SUBJECT BOUNDARIES: If the topic is non-programming (like English Grammar), DO NOT write any code or programming logic.
3. QUESTION TYPES: If the user requested only MCQs, generate ONLY MCQs. Do not add code blocks or debug questions.
4. PROGRAMMING LANGUAGES: If the course is a specific language (e.g., Ruby, Rust, React), write questions strictly for that language. Do not default to Python.

You are ATLAS Exam Writer, a calm and precise quiz author for a learning platform.
You write clear, accurate quizzes on one subject at a time.

<assignment>
Mode: <<MODE>>
Subject area: <<SUBJECT_DOMAIN>>
Programming course: <<IS_PROGRAMMING>>
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
course_focus: Same as course, and the learner also gave a focus request in <focus>. Choose the roadmap topics and notes that match the focus, and build most of the quiz around it. If the focus names something outside the course, the focus wins: write those questions about the focus topic itself and never swap in unrelated course subjects.
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
standard knowledge of the subject.
```

**Placeholder replacement (`build_system_prompt`):**

| Placeholder | Value |
| --- | --- |
| `<<TOPIC>>` | `plan.topic` — typed topic / course title / `"Uploaded PDF"` |
| `<<SUBJECT_DOMAIN>>` | `plan.domain` — one of `math, english, science, humanities, programming, other` |
| `<<IS_PROGRAMMING>>` | `true` when `plan.domain == "programming"`, else `false` (the boolean Rule 2 hands to the prompt) |
| `<<MODE>>` | `topic_only` \| `course` \| `course_focus` \| `pdf` |
| `<<LANGUAGE>>` | `plan.language` — `English` \| `Urdu` (backend-decided) |
| `<<NUM_QUESTIONS>>` | request `count` (1-20) |
| `<<DIFFICULTY>>` | `easy` \| `medium` \| `hard` |
| `<<ALLOWED_KINDS>>` | non-programming → always `mcq`; programming → requested kind or all of `mcq, code, debug, output` (Rule 3: the UI's MCQ-only request is never widened) |
| `<<STYLE_HINT>>` | `style_for(domain)` — non-programming gets the MCQ variety hint; programming gets "Keep a natural balance of question styles." |
| `<<CODING_LANGUAGE_LINE>>` | `Coding language for code questions: {coding_language}` only when code/debug/output are allowed **and** `plan.coding_language` is non-empty; otherwise empty |
| `<<OUTPUT_FORMAT>>` | `build_output_format(plan)` — JSON shapes for the allowed kinds only (`_OUTPUT_SHAPES`); the code shape's `<<LANG>>` becomes `plan.coding_language`; when `debug` is allowed the `DEBUG_AUTHORING` note is appended: `buggy_line` = 1-based line of the bug inside `code`, `reference_solution` = full corrected code that must differ from the buggy code |

---

## 4. Prompt C — Quiz generation user message (direct)

Built by `build_user_message(plan)` — the ONLY material the model receives:

| Mode | User message (verbatim assembly) |
| --- | --- |
| `topic_only` | `Create the quiz now.` |
| `course` | `Create the quiz now.` + `<completed_roadmap_topics>…</completed_roadmap_topics>` + `<notes>…</notes>` |
| `course_focus` | same as `course` + `<focus>…</focus>` |
| `pdf` | `Create the quiz now.` + `<pdf_text>…</pdf_text>` |

- Notes = `pick_notes(...)`: **completed topics only**, non-programming quizzes
  drop code-looking notes, focus text ranks matching notes first,
  max **8 notes / 6000 chars**, joined with `\n\n---\n\n`.
- PDF text = extracted text capped at **8000 chars** (route cap 30 000).
- Client-supplied `context` / `subject` fields are **never read or sent**
  (Rule 1: custom topics get zero course material).

**Retry nudge (appended to the user message on attempt 2 and 3):**

```text

Your previous reply could not be used ({last_err}). Please send the complete quiz again as one JSON object, in {language}, using only these kinds: {kinds}.
```

When `debug` is among the allowed kinds, one more sentence is appended:

```text
 For debug questions: buggy_line must be a real 1-based line of the code and reference_solution must be the corrected code.
```

---

## 5. Prompt D — Answer evaluation (indirect, code/debug/output only)

`POST /api/quizzes/evaluate` → MCQ judged locally; other kinds call
`ai_service.complete_json(Task.CODE, prompts.evaluate_messages(req))`.

**System (`prompts.py`):**

```text
You are ATLAS, a strict but fair automated code assessor. Judge only what is written. {JSON_RULE}
{EVALUATE_RULE}
```

where `JSON_RULE` is:

```text
CRITICAL OUTPUT CONTRACT: You MUST return ONLY valid JSON. Do not include markdown formatting like ```json ... ```, no commentary, no explanation, no prose before or after the JSON. Your entire reply must be a single JSON value that json.loads() can parse on the first try.
```

and `EVALUATE_RULE` (`prompts.py:56`, verbatim):

```text
Grade the submission and return a JSON object with exactly these keys:
{
  "is_correct": boolean,
  "score": number between 0 and 100,
  "feedback": string (2-5 sentences, direct and specific)
}

If the user's answer is incorrect, provide exactly ONE concise, clear explanation
for why the answer is wrong. Do not repeat, rephrase, or output the explanation
twice. After that single explanation, provide an 'Expected Format' example showing
exactly how they should have written the output (e.g., 'Just write the final
number', or 'Write the code in a single block'). Keep the entire "feedback" to at
most 4 sentences: one explanation, then the Expected Format example. Be a
supportive teacher. Put the Expected Format example inside "feedback" - never add
extra keys to the JSON. When the answer is correct, say so briefly and warmly.

In "feedback", use the same rich style for math and symbols (x², ≤, ≠, √, H₂O,
and inline $...$ LaTeX like $\frac{1}{2}$ for complex expressions). Any code
examples inside feedback stay plain ASCII.
```

**User message template (`evaluate_messages`):**

```text
Question kind: {kind}
Language: {language}
Question: {question}
Original code: 
---
{code or 'not provided'}
---
Expected output: {expected_output or 'not provided'}
Reference solution: {reference_solution or 'not provided'}
Declared buggy line: {buggy_line or 'not given'}
User submission: 
---
{answer}
---
```

Score is clamped to 0-100; `is_correct` defaults to `score >= 100` when absent.

---

## 6. End-to-end backend flow

### `POST /api/quizzes/generate`

```
auth (Depends get_current_user)          -> 401 without a valid Supabase JWT
course_id falsy (null/missing/0) -> None -> no course bundle ever fetched
build_plan(req, auth, pdf_text=None, _call_llm_small)
  ├─ course_id set  : fetch this course's bundle only (404/400)
  │                   Rule 2: title marker -> domain programming, language
  │                   detected, no LLM, nothing saved; else stored domain
  │                   (classify + save lazily), kinds, language,
  │                   notes = pick_notes(completed only)
  │                   -> mode course_focus|course
  ├─ pdf_text       : mode pdf, _domain_for(text[:2000]) -> domain,
  │                   language detected from the head, notes = text[:8000]
  └─ typed topic    : mode topic_only, _domain_for(topic) -> domain,
                      language detected from the topic  [zero notes]
generate_quiz(plan, n, difficulty, _call_llm)          [up to 3 attempts]
  system = build_system_prompt(...)      user = build_user_message(...) + nudge
  call -> sanitize_llm_json -> json.loads -> validate_quiz
    fail -> PRINT "QUIZ_REJECTED: ..." -> retry with nudge
    (debug-kind plans also append the debug-fix sentence)
  3 failures -> RuntimeError -> HTTP 503 "Quiz could not be generated, please retry."
success -> QuizGenerateResponse(questions, model)
```

### `POST /api/quizzes/generate-pdf`

Same `_generate(...)` with `source_text` from the upload:

- max **10 MB**, encrypted/malformed/unscannable → clean 400s,
- text extracted in memory only, capped at **30 000 chars**, temp file closed
  immediately, bytes dropped after extraction (**nothing persisted**),
- `topic` comes from the form; mode still resolves via `build_plan`.

### `POST /api/quizzes/evaluate`

MCQ → local compare. code/debug/output → Prompt D (section 5).

---

## 7. Sanitize + validation rules (what happens to the raw reply)

`sanitize_llm_json(raw)` (`app/quiz_service.py`):

1. strip markdown fences, cut first `{` … last `}`,
2. LaTeX → Unicode (`\frac{a}{b}` → `a/b`, `\times` → `×`, `\sqrt` → `√`, …),
3. drop `$`, convert `^2`/`^3` → `²`/`³`,
4. remove every backslash that is not a legal JSON escape (`\"`, `\\`, `\n`, `\uXXXX`).

`validate_quiz(data, plan, n)` — a rejection here triggers the retry nudge:

- must be one JSON object with a non-empty `questions` array,
- `len(questions) >= max(1, n - 1)`,
- every `kind` must be in `plan.kinds` (Rule 3: a math quiz, or an MCQ-only
  programming request, can never receive `code`/`debug`/`output`),
- mcq: exactly 4 options, `correct_index` in 0-3; code: needs `reference_solution`;
  output: needs `code` + `correct_output`,
- **debug (light checks only)**: `code` non-empty, `buggy_line` inside
  `1..len(code.splitlines())`, non-empty `reference_solution`, and the fix
  must actually change the code (whitespace-normalized equality → rejected).
  No AST parsing, no similarity scoring,
- `code` / `debug` / `output` questions get a backend-decided `language` in
  this order: `plan.coding_language` (Rule 2, title/topic) → marker read off
  the question's own prompt/code (catches a typo'd title like "HTLM BASICS"
  when the question says "HTML snippet") → the model's value. Never a blind
  `"python"`; genuinely unknown stays empty (the UI hides the badge),
- **non-programming quizzes may not contain any code** (fences or code-shaped text),
- language script check: English quiz → no Urdu script; Urdu quiz → must contain
  Urdu script (per-question, enforced on prompt/options/explanation/code fields).

---

## 8. Security / privacy guarantees baked into the prompts

- The model only ever sees: `plan.topic`, `plan.domain`, mode/language/counts,
  and server-side material (course bundle notes / PDF text). **Client `context`
  and `subject` never reach the model** (Rule 1), and each course fetch is
  scoped to the requested `course_id` — notes of other courses are never loaded.
- The system prompt opens with the four **CRITICAL RULES** (context, subject
  boundaries, question types, programming languages); the `<course_focus>`
  "focus wins" rule stops injected course notes from redirecting the quiz to
  another subject.
- Anonymous requests are rejected before any plan is built (401).
- Course bundles are fetched with the caller's JWT (RLS applies); PDFs live in
  request memory only and are never stored.
