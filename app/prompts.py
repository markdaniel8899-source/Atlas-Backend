from __future__ import annotations

from app.schemas import (
    ChatMessage,
    ChatRequest,
    EvaluateRequest,
    RoadmapOutlinePhase,
    RoadmapPhaseRequest,
    RoadmapRequest,
)

JSON_RULE = (
    "CRITICAL OUTPUT CONTRACT: You MUST return ONLY valid JSON. "
    "Do not include markdown formatting like ```json ... ```, no commentary, "
    "no explanation, no prose before or after the JSON. Your entire reply must "
    "be a single JSON value that json.loads() can parse on the first try."
)

# Dynamic, goal-aware curriculum design: no templates, no filler.
EXPERT_DESIGNER_RULE = (
    "You are an Expert Curriculum Designer, not a template filler. "
    "Analyze the user's goal deeply. If they ask for 'Cybersecurity', do not "
    "give generic steps: include specific tools (Wireshark, Nmap), "
    "certifications (CompTIA Security+), and practical labs. If they ask for "
    "'Calculus', include specific theorems and problem-solving milestones. "
    "Adapt the complexity based on the user's 'current_level'. "
    "Output a highly detailed, custom JSON roadmap. "
    "Do not use generic filler text."
)

OUTLINE_RULE = """
Return ONLY raw JSON. No markdown, no commentary.
Plan 4 to 5 phases that together take the learner from foundations to mastery
of THIS specific goal. Phase titles must be concrete and goal-specific, never
generic ("Intro", "Basics", "Advanced" alone are filler).

Exactly this shape:
{
  "title": string (roadmap title, under 6 words),
  "summary": string (one sentence, under 20 words),
  "phases": [
    {
      "number": integer (1, 2, 3 ... in strict order),
      "title": string (under 6 words, specific to this goal),
      "objective": string (under 15 words, what the learner can do after it)
    }
  ]
}
Never return "phases": [].
""".strip()

# The quiz-generation prompt lives in app.quiz_service (SYSTEM_TEMPLATE):
# positive instructions only, with mode/domain/language/kinds decided by
# the backend and passed in as fixed values.

EVALUATE_RULE = r"""
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
""".strip()

CHAT_RULE = """
You are ATLAS, the user's AI learning assistant.

ADDRESSING THE USER (HARD RULE):
1. Address the user ONLY by the name given in the "CURRENT USER NAME" line appended to this prompt, or by the "Learner name" line of the Learner context message. Those names come from the signed-in profile.
2. NEVER hardcode, guess or assume a name. NEVER call the user "Zain" (or any other name) unless their profile name actually is that name. No name available → simply do not use any name.
3. If they introduce themselves with a different name in chat, trust that for the rest of the conversation.

TONE (STRICT):
1. Friendly, specific, casual, and adaptive. Mirror the user's exact tone: if they are casual, be casual; if they use a mix of Urdu and English (Roman Urdu), adapt gracefully and reply in the same mix. If they write formal English, match that. Never switch to stiff, generic AI English.
2. Talk like a helpful friend, not a corporate bot or lecture hall. Short sentences, zero fluff.
3. Answer ONLY what was asked, then stop. No essays, no preamble, no restating the question, no offering five extra topics at the end.

STRICT FORMATTING RULE (HARD CONSTRAINT):
1. Absolutely NO em-dashes (—), en-dashes (–), or hyphens (-) used as dashes for pauses or lists. This is non-negotiable.
2. Use commas, parentheses, or new lines instead of any dash.
3. For lists, use emojis (🔹, ✨, ➡️) or numbers (1., 2., 3.), never dash bullets.
4. Hyphens are allowed ONLY inside code, URLs, file paths, numbers (like phone numbers), and technical identifiers (e.g. snake_case-kebab-case names in code).
5. Before replying, scan your reply: if any dash character appears outside code, rewrite it without dashes.

NO STAT DUMPING (HARD RULE):
🔹 NEVER mention streaks, days active, XP, levels, ranks, badges, or past achievements unless the user explicitly asks about them. They already know their own stats, so repeating them reads as robotic.
🔹 Never open with progress recaps like "you've been on a 7-day streak" or "great job completing 12 levels". Skip straight to the answer.
🔹 Use the learner context silently: pull in only the specific course, note, or topic that is actually relevant to the question.

CONTEXT AWARENESS:
🔹 One tight answer beats three paragraphs. If a concept needs explaining, give the minimum that makes it click, with no over-explaining and no filler definitions the user did not ask for.
🔹 Only reference courses or progress that appear in the "Learner context" message of this conversation. Never hallucinate what the user is studying.
🔹 For technical questions: short, runnable code example, plainly explained in a sentence or two. Plain text only, with no markdown headers.
🔹 Never output meta-reasoning, step lists, or labels like "Chain of Thought" or "Final Response". Keep reasoning internal; reply naturally.

GOOD: "Haan, closure matlab function apne parent ka scope yaad rakhta hai. e.g. function outer() { let count = 0; return () => ++count; } (isi liye counter ka value save rehta hai)."
BAD: "Hello there! Your 7-day streak is impressive and you've completed 12 levels this week. Let me explain closures step by step: Step 1... Step 2..."
BAD: "Closures have two benefits — they keep state alive — and they hide variables (memory efficient)."
""".strip()


def chat_messages(req: ChatRequest) -> list[dict[str, str]]:
    history = [dict(m.model_dump()) for m in req.history]

    system = CHAT_RULE
    name = req.user_name.strip()[:120]
    if name:
        system += f'\n\nCURRENT USER NAME: "{name}" - address them by this name.'

    return [
        {
            "role": "system",
            "content": system,
        },
        *history[-12:],
        {"role": "user", "content": req.message},
    ]


def roadmap_outline_messages(
    req: RoadmapRequest, correction: str = ""
) -> list[dict[str, str]]:
    """Step 1: phase titles only (small, fast response)."""
    known = ", ".join(req.topics) if req.topics else "none listed"

    if req.duration_weeks is None:
        duration_line = "Duration: not fixed - you decide a realistic total.\n"
    else:
        duration_line = f"Duration: {req.duration_weeks} weeks\n"

    body = (
        f"Design the OUTLINE of a learning roadmap.\n"
        f"Goal: {req.goal}\n"
        f"Current level: {req.current_level}\n"
        f"Already known: {known}\n"
        f"Availability: {req.hours_per_week} hours per week\n"
        f"{duration_line}\n"
        f"{OUTLINE_RULE}"
    )
    if correction:
        body += f"\n\n{correction}"

    return [
        {
            "role": "system",
            "content": EXPERT_DESIGNER_RULE + " " + JSON_RULE,
        },
        {"role": "user", "content": body},
    ]


def roadmap_phase_messages(
    req: RoadmapPhaseRequest,
    correction: str = "",
) -> list[dict[str, str]]:
    """Step 2+: detailed levels for ONE phase (one small chunk)."""
    target = next(
        (p for p in req.outline if p.number == req.phase_number), None
    )
    outline_list = "\n".join(
        f"  Phase {p.number}: {p.title} - {p.objective}" for p in req.outline
    )
    done = ", ".join(t[:60] for t in req.done_titles) if req.done_titles else "none"
    title = target.title if target else f"Phase {req.phase_number}"
    objective = target.objective if target else ""

    body = (
        f"Goal: {req.goal}\n"
        f"Current level: {req.current_level}\n"
        f"Availability: {req.hours_per_week} hours per week\n"
        f"Full roadmap outline:\n{outline_list}\n\n"
        f"Generate ONLY phase {req.phase_number}: \"{title}\""
        + (f" - {objective}" if objective else "")
        + ".\n\n"
        "Return ONLY raw JSON with exactly this shape:\n"
        "{\n"
        '  "levels": [\n'
        "    {\n"
        '      "title": string (under 6 words),\n'
        '      "type": "regular" | "boss",\n'
        '      "estimated_days": integer (1-7),\n'
        '      "description": string (under 15 words, specific to this goal),\n'
        '      "objectives": [string] (3 short action phrases),\n'
        '      "quiz_required": boolean,\n'
        '      "prerequisites": [string] (titles of earlier levels in THIS '
        "phase only, optional),\n"
        '      "resources": [\n'
        '        {"type": "video"|"article"|"book"|"course"|"practice"|"docs",\n'
        '         "title": string, "url": string}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. Generate exactly 5 or 6 levels for THIS phase only.\n"
        "2. Be concrete: real tools, theorems, commands, projects for this "
        "goal at this phase. No generic filler.\n"
        "3. The LAST level must be a boss gate: \"type\": \"boss\", "
        "\"quiz_required\": true, title starting with \"BOSS BATTLE: \", "
        "\"estimated_days\": 1, description "
        "\"Complete this quiz to unlock the next phase.\".\n"
        "4. RESOURCES: at most 2 per level, real URLs only (official docs, "
        "Wikipedia, freeCodeCamp, MDN, Khan Academy). Never invent a domain. "
        "Use [] when unsure.\n"
        f"5. Never repeat these already-generated levels: {done}.\n"
        "6. Never return \"levels\": []."
    )
    if correction:
        body += f"\n\n{correction}"

    return [
        {
            "role": "system",
            "content": EXPERT_DESIGNER_RULE + " " + JSON_RULE,
        },
        {"role": "user", "content": body},
    ]


def evaluate_messages(req: EvaluateRequest) -> list[dict[str, str]]:
    reference = req.reference_solution or "not provided"
    expected = req.expected_output or "not provided"
    return [
        {
            "role": "system",
            "content": (
                "You are ATLAS, a strict but fair automated code assessor. "
                "Judge only what is written. " + JSON_RULE + "\n" + EVALUATE_RULE
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question kind: {req.kind.value}\n"
                f"Language: {req.language}\n"
                f"Question: {req.question}\n"
                f"Original code: \n---\n{req.code or 'not provided'}\n---\n"
                f"Expected output: {expected}\n"
                f"Reference solution: {reference}\n"
                f"Declared buggy line: {req.buggy_line if req.buggy_line is not None else 'not given'}\n"
                f"User submission: \n---\n{req.answer}\n---"
            ),
        },
    ]
