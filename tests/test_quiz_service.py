"""Unit + Section 8 tests for the quiz engine (int.md).

Run from backend/:  python -m unittest discover -s tests -v

Section 8 mapping:
  1 -> test_s8_1_basic_math_topic + test_s8_1_route_prompt_and_response
  2 -> test_s8_2_english_grammar
  3 -> test_s8_3_python_topic_kinds
  4 -> test_s8_4_course_mode
  5 -> test_s8_5_course_focus
  6 -> test_s8_6_pdf_mode
  7 -> test_s8_7_general_empty_is_400
  8 -> test_s8_8_latex_sanitizer

Course rows/notes are mocked bundles (no live Supabase or auth token is
available in tests); everything else runs the real build_plan / prompt /
sanitizer / validation / retry code.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import ai_service
from app.quiz_service import (
    QuizPlan,
    build_plan,
    build_output_format,
    build_system_prompt,
    build_user_message,
    classify_domain,
    detect_dev_language,
    generate_quiz,
    kinds_for,
    pick_notes,
    resolve_language,
    sanitize_llm_json,
    style_for,
    validate_quiz,
)
from app.routers import quizzes as quizzes_router
from app.schemas import QuizGenerateRequest
from app.deps import get_current_user


def run(coro):
    return asyncio.run(coro)


def make_req(**kwargs) -> QuizGenerateRequest:
    payload = {
        "topic": "",
        "kind": "mcq",
        "count": 5,
        "difficulty": "medium",
        "language": "python",
        "context": "",
        "course_id": None,
        "subject": "",
    }
    payload.update(kwargs)
    return QuizGenerateRequest(**payload)


def mcq_item(prompt: str = "What is 2 + 2?") -> dict:
    return {
        "kind": "mcq",
        "prompt": prompt,
        "options": ["3", "4", "5", "6"],
        "correct_index": 1,
        "explanation": "2 + 2 = 4.",
    }


def mcq_json(n: int) -> str:
    return json.dumps({"questions": [mcq_item() for _ in range(n)]})


async def llm_domain(system: str, user: str) -> str:
    return "science"


async def llm_boom(system: str, user: str) -> str:
    raise RuntimeError("classify LLM unavailable")


async def llm_never(system: str, user: str) -> str:
    raise AssertionError("keywords should decide the domain without the LLM")


def make_bundle(**overrides) -> dict:
    bundle = {
        "course": {
            "id": 1,
            "title": "Python Foundations",
            "description": "",
            "domain": "programming",
            "status": "active",
            "progress_percentage": 40,
        },
        "topics": [
            {"id": 10, "title": "Loops", "status": "completed"},
            {"id": 11, "title": "Functions", "status": "completed"},
            {"id": 12, "title": "OOP", "status": "in_progress"},
        ],
        "roadmap_title": "Python path",
        "nodes": [
            {"title": "Loops", "status": "completed"},
            {"title": "Functions", "status": "completed"},
            {"title": "OOP", "status": "active"},
        ],
        "completed_ids": [10, 11],
        "completed_titles": ["Loops", "Functions"],
        "notes": [
            {
                "id": 100,
                "title": "Loops notes",
                "content": "<p>for x in range(5): run()</p>",
                "topic_id": 10,
                "course_id": 1,
            },
            {
                "id": 101,
                "title": "Functions notes",
                "content": "<p>def greet(name): return hi</p>",
                "topic_id": 11,
                "course_id": 1,
            },
            {
                "id": 102,
                "title": "OOP notes",
                "content": "<p>classes and objects</p>",
                "topic_id": 12,
                "course_id": 1,
            },
        ],
    }
    bundle.update(overrides)
    return bundle


class DomainAndKindsTests(unittest.TestCase):
    def test_classify_keywords(self):
        cases = {
            "Basic Math": "math",
            "English Grammar": "english",
            "Python list comprehensions": "programming",
            "Photosynthesis in plants": "science",
            "History of Rome": "humanities",
            "اردو ادب": "humanities",
        }
        for text, expected in cases.items():
            self.assertEqual(run(classify_domain(text, llm_never)), expected, text)

    def test_classify_llm_fallback(self):
        self.assertEqual(run(classify_domain("Biryani cooking class", llm_domain)), "science")
        self.assertEqual(run(classify_domain("Biryani cooking class", llm_boom)), "other")
        self.assertEqual(run(classify_domain("Something", AsyncMock(return_value="banana"))), "other")
        self.assertEqual(run(classify_domain("", llm_never)), "other")

    def test_kinds_for(self):
        self.assertEqual(kinds_for("math", "mixed"), ["mcq"])
        self.assertEqual(kinds_for("math", "code"), ["mcq"])
        self.assertEqual(kinds_for("english", "debug"), ["mcq"])
        self.assertEqual(
            kinds_for("programming", "mixed"),
            ["mcq", "code", "debug", "output"],
        )
        self.assertEqual(kinds_for("programming", "debug"), ["debug"])
        self.assertEqual(kinds_for("programming", "mcq"), ["mcq"])

    def test_style_for(self):
        self.assertIn("Vary the styles", style_for("math", "mcq"))
        self.assertIn("4-option MCQs", style_for("english", "mixed"))
        self.assertIn("natural balance", style_for("programming", "mixed"))

    def test_resolve_language(self):
        self.assertEqual(resolve_language("english", "anything"), "English")
        self.assertEqual(resolve_language("Urdu", "anything"), "Urdu")
        # "python" is the coding language - it never decides the quiz language.
        self.assertEqual(resolve_language("python", "Basic Math"), "English")
        self.assertEqual(resolve_language(None, "اردو قواعد"), "Urdu")
        self.assertEqual(resolve_language(None, "English Grammar"), "English")


class SanitizerTests(unittest.TestCase):
    def test_s8_8_latex_sanitizer(self):
        raw = (
            '```json\n{"questions":[{"kind":"mcq",'
            '"prompt":"What is \\frac{1}{2} + \\frac{1}{2}?",'
            '"options":["1/2","1","2","\\sqrt{4}"],'
            '"correct_index":1,'
            '"explanation":"Because \\frac{1}{2}+\\frac{1}{2}=1 and 2 \\times 3 = 6"}]}\n```'
        )
        cleaned = sanitize_llm_json(raw)
        data = json.loads(cleaned)  # the old 502: this json.loads used to blow up
        question = data["questions"][0]
        self.assertNotIn("\\frac", question["prompt"])
        self.assertIn("1/2", question["prompt"])
        self.assertNotIn("\\times", question["explanation"])
        self.assertIn("2 × 3", question["explanation"])
        self.assertNotIn("\\sqrt", question["options"][3])
        self.assertTrue(question["options"][3].startswith("√"))

    def test_invalid_backslashes_are_dropped(self):
        cleaned = sanitize_llm_json(r'{"a": "50\% done", "b": "tab\there"}')
        data = json.loads(cleaned)
        self.assertEqual(data["a"], "50% done")
        self.assertEqual(data["b"], "tab\there")  # \t is a legal JSON escape

    def test_no_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            sanitize_llm_json("I cannot help with that.")


class PromptTests(unittest.TestCase):
    def _plan(self, **kwargs) -> QuizPlan:
        base = dict(
            mode="topic_only",
            topic="Basic Math",
            domain="math",
            kinds=["mcq"],
            language="English",
            style_hint=style_for("math", "mcq"),
        )
        base.update(kwargs)
        return QuizPlan(**base)

    def test_positive_prompt_has_no_negative_rules(self):
        system = build_system_prompt(self._plan(), 10, "medium")
        # The CRITICAL RULES block at the very top is the one deliberate
        # negative-rule area (the user's four rules); everything after the
        # writer persona must stay positive-only.
        head, sep, body = system.partition("\n\nYou are ATLAS Exam Writer")
        self.assertTrue(sep, "rules block must precede the writer persona")
        self.assertTrue(head.startswith("CRITICAL RULES:"), "rules must be first")
        for rule in ("1. CONTEXT:", "2. SUBJECT BOUNDARIES:", "3. QUESTION TYPES:", "4. PROGRAMMING LANGUAGES:"):
            self.assertIn(rule, head)
        for banned in ("FORBIDDEN", "STRICTLY", "STRICT", "DO NOT", "Never", "python", "print(", "def "):
            self.assertNotIn(banned, body)
        self.assertIn("Subject area: math", body)
        self.assertIn("Programming course: false", body)
        self.assertIn("Allowed question kinds: mcq", body)
        self.assertIn("Quiz language: English", body)

    def test_output_format_only_allowed_kinds(self):
        math_format = build_output_format(self._plan())
        self.assertIn('"kind": "mcq"', math_format)
        self.assertNotIn('"kind": "code"', math_format)
        self.assertNotIn('"kind": "debug"', math_format)

        prog = self._plan(domain="programming", kinds=["mcq", "code", "debug", "output"])
        prog_format = build_output_format(prog)
        for kind in ("mcq", "code", "debug", "output"):
            self.assertIn(f'"kind": "{kind}"', prog_format)

    def test_coding_language_line_only_for_programming(self):
        # No detected language -> no line, even on a programming quiz.
        math_system = build_system_prompt(self._plan(), 5, "medium")
        self.assertNotIn("Coding language", math_system)

        prog = self._plan(domain="programming", kinds=["code"])
        self.assertNotIn("Coding language", build_system_prompt(prog, 5, "medium"))

        prog = self._plan(
            domain="programming", kinds=["code"], coding_language="python"
        )
        prog_system = build_system_prompt(prog, 5, "medium")
        self.assertIn("Coding language for code questions: python", prog_system)

        # Detected language but MCQ-only quiz -> no coding line.
        mcq_only = self._plan(
            domain="programming", kinds=["mcq"], coding_language="python"
        )
        self.assertNotIn("Coding language", build_system_prompt(mcq_only, 5, "medium"))

    def test_topic_only_user_message_has_no_material(self):
        message = build_user_message(self._plan())
        self.assertEqual(message, "Create the quiz now.")
        for tag in ("<notes>", "<completed_roadmap_topics>", "<focus>", "<pdf_text>"):
            self.assertNotIn(tag, message)

    def test_course_user_message_blocks(self):
        plan = self._plan(
            mode="course_focus",
            topic="Python Foundations",
            notes_text="[Loops notes]\nfor x in range(5)",
            roadmap_topics=["Loops", "Functions"],
            focus="loops only",
        )
        message = build_user_message(plan)
        self.assertIn("<completed_roadmap_topics>\nLoops\nFunctions\n</completed_roadmap_topics>", message)
        self.assertIn("<notes>\n[Loops notes]", message)
        self.assertIn("<focus>\nloops only\n</focus>", message)


class BuildPlanTests(unittest.TestCase):
    def test_s8_1_basic_math_topic(self):
        req = make_req(
            topic="Basic Math",
            kind="code",  # worst case: a mixed fan-out batch asked for code
            context="PYTHON NOTES: def foo(): print(1)",
            subject="programming",
        )
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock()) as fetch:
            plan = run(build_plan(req, None, None, llm_never))
        fetch.assert_not_awaited()  # no course_id -> zero course context
        self.assertEqual(plan.mode, "topic_only")
        self.assertEqual(plan.domain, "math")
        self.assertEqual(plan.kinds, ["mcq"])
        self.assertEqual(plan.notes_text, "")
        self.assertEqual(plan.language, "English")
        self.assertEqual(plan.roadmap_topics, [])

    def test_s8_2_english_grammar(self):
        req = make_req(topic="English Grammar")
        plan = run(build_plan(req, None, None, llm_never))
        self.assertEqual(plan.domain, "english")
        self.assertEqual(plan.kinds, ["mcq"])
        self.assertEqual(plan.language, "English")
        self.assertEqual(plan.notes_text, "")

    def test_s8_3_python_topic_kinds(self):
        req = make_req(topic="Python list comprehensions", kind="mcq")
        plan = run(build_plan(req, None, None, llm_never))
        self.assertEqual(plan.domain, "programming")
        # "Mixed set" fans out per kind; the mixed expansion includes them all.
        self.assertEqual(
            kinds_for(plan.domain, "mixed"),
            ["mcq", "code", "debug", "output"],
        )

    def test_s8_4_course_mode(self):
        bundle = make_bundle()
        with patch(
            "app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)
        ) as fetch, patch(
            "app.quiz_service.save_course_domain", AsyncMock(return_value=True)
        ) as save:
            req = make_req(topic="", course_id=1)
            plan = run(build_plan(req, "tok-123", None, llm_never))

        self.assertEqual(plan.mode, "course")
        self.assertEqual(plan.topic, "Python Foundations")
        self.assertEqual(plan.domain, "programming")
        self.assertEqual(plan.roadmap_topics, ["Loops", "Functions"])
        self.assertIn("[Loops notes]", plan.notes_text)
        self.assertIn("[Functions notes]", plan.notes_text)
        self.assertNotIn("OOP notes", plan.notes_text)  # topic not completed
        self.assertIn("for x in range(5)", plan.notes_text)  # HTML stripped
        fetch.assert_awaited_once_with(1, "tok-123")
        save.assert_not_awaited()  # domain already stored on the course
        message = build_user_message(plan)
        self.assertIn("<completed_roadmap_topics>", message)
        self.assertIn("<notes>", message)
        self.assertNotIn("<focus>", message)

    def test_s8_4_course_missing_is_404(self):
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as ctx:
                run(build_plan(make_req(course_id=9), None, None, llm_never))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_course_without_completed_topics_is_400(self):
        bundle = make_bundle(completed_ids=[], completed_titles=[], notes=[])
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)):
            with self.assertRaises(HTTPException) as ctx:
                run(build_plan(make_req(course_id=1), None, None, llm_never))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_course_domain_lazily_classified_and_saved(self):
        bundle = make_bundle(
            course={"id": 1, "title": "Basic Math", "domain": "", "description": ""}
        )
        with patch(
            "app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)
        ), patch(
            "app.quiz_service.save_course_domain", AsyncMock(return_value=True)
        ) as save:
            plan = run(build_plan(make_req(course_id=1), "tok", None, llm_never))
        self.assertEqual(plan.domain, "math")
        save.assert_awaited_once_with(1, "math", "tok")

    def test_s8_5_course_focus(self):
        bundle = make_bundle(
            notes=[
                {
                    "id": 1,
                    "title": "Recursion tricks",
                    "content": "<p>call the function again</p>",
                    "topic_id": 10,
                    "course_id": 1,
                },
                {
                    "id": 2,
                    "title": "Loops guide",
                    "content": "<p>while and for</p>",
                    "topic_id": 11,
                    "course_id": 1,
                },
            ]
        )
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)):
            req = make_req(topic="loops and functions only", course_id=1)
            plan = run(build_plan(req, None, None, llm_never))

        self.assertEqual(plan.mode, "course_focus")
        self.assertEqual(plan.topic, "Python Foundations")  # topic stays the course
        self.assertEqual(plan.focus, "loops and functions only")
        self.assertEqual(plan.domain, "programming")
        # Focus-matching notes are ranked first.
        self.assertLess(
            plan.notes_text.index("Loops guide"),
            plan.notes_text.index("Recursion tricks"),
        )
        message = build_user_message(plan)
        self.assertIn("<focus>\nloops and functions only\n</focus>", message)

    def test_domain_comes_from_course_never_focus(self):
        bundle = make_bundle(course={"id": 1, "title": "Basic Math", "domain": "math"})
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)):
            plan = run(
                build_plan(make_req(topic="python loops", course_id=1), None, None, llm_never)
            )
        self.assertEqual(plan.domain, "math")
        self.assertEqual(plan.mode, "course_focus")

    def test_non_programming_course_drops_code_looking_notes(self):
        bundle = make_bundle(
            course={"id": 2, "title": "Basic Math", "domain": "math"},
            notes=[
                {
                    "id": 1,
                    "title": "Sneaky code note",
                    "content": "<p>def add(a): print(add(1))</p>",
                    "topic_id": 10,
                    "course_id": 2,
                },
                {
                    "id": 2,
                    "title": "Clean algebra note",
                    "content": "<p>a squared plus b squared</p>",
                    "topic_id": 11,
                    "course_id": 2,
                },
            ],
        )
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)):
            plan = run(build_plan(make_req(course_id=2), None, None, llm_never))
        self.assertNotIn("Sneaky code note", plan.notes_text)
        self.assertIn("Clean algebra note", plan.notes_text)

    def test_s8_6_pdf_mode(self):
        pdf_text = "Physics syllabus: gravity, velocity and acceleration of falling objects. " * 5
        req = make_req(topic="")
        plan = run(build_plan(req, None, pdf_text, llm_never))
        self.assertEqual(plan.mode, "pdf")
        self.assertEqual(plan.topic, "Uploaded PDF")
        self.assertEqual(plan.domain, "science")
        self.assertEqual(plan.notes_text, pdf_text[:8000])
        self.assertIn("<pdf_text>", build_user_message(plan))

    def test_pdf_endpoint_filename_topic_does_not_hijack_mode(self):
        # The UI sends the PDF file name as `topic`; the upload must still win.
        pdf_text = "Gravity pulls objects toward the earth."
        req = make_req(topic="physics-syllabus")
        plan = run(build_plan(req, None, pdf_text, llm_never))
        self.assertEqual(plan.mode, "pdf")

    def test_course_wins_over_pdf_and_topic(self):
        bundle = make_bundle()
        with patch("app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)):
            req = make_req(topic="focus text", course_id=1)
            plan = run(build_plan(req, None, "pdf bytes here", llm_never))
        self.assertEqual(plan.mode, "course_focus")

    def test_s8_7_general_empty_is_400(self):
        with self.assertRaises(HTTPException) as ctx:
            run(build_plan(make_req(topic=""), None, None, llm_never))
        self.assertEqual(ctx.exception.status_code, 400)


class PickNotesTests(unittest.TestCase):
    def test_only_completed_topics_and_domain_filter(self):
        notes = [
            {"id": 1, "title": "Clean algebra", "content": "<p>a squared</p>", "topic_id": 10},
            {"id": 2, "title": "Sneaky code", "content": "<p>def add(a): print(add(1))</p>", "topic_id": 11},
            {"id": 3, "title": "Other topic", "content": "<p>geometry</p>", "topic_id": 99},
        ]
        math_out = pick_notes(notes, [10, 11], "", "math")
        self.assertIn("Clean algebra", math_out)
        self.assertNotIn("Sneaky code", math_out)
        self.assertNotIn("Other topic", math_out)

        prog_out = pick_notes(notes, [10, 11], "", "programming")
        self.assertIn("Sneaky code", prog_out)

        self.assertEqual(pick_notes([], [10], "", "math"), "No notes provided.")


class ValidateQuizTests(unittest.TestCase):
    def setUp(self):
        self.math_plan = QuizPlan(
            mode="topic_only",
            topic="Basic Math",
            domain="math",
            kinds=["mcq"],
            language="English",
        )

    def test_valid_quiz_passes(self):
        data = {"questions": [mcq_item(), mcq_item("What is 3 + 3?")]}
        questions = validate_quiz(data, self.math_plan, 2)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].kind.value, "mcq")

    def test_rejects_code_kind_in_math(self):
        data = {
            "questions": [
                {
                    "kind": "code",
                    "prompt": "Write a function",
                    "reference_solution": "def f(): pass",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_quiz(data, self.math_plan, 1)

    def test_rejects_code_content_in_non_programming(self):
        data = {"questions": [mcq_item(prompt="def add(a):\n    print(add(1))")]}
        with self.assertRaisesRegex(ValueError, "code content"):
            validate_quiz(data, self.math_plan, 1)

    def test_rejects_urdu_in_english_quiz(self):
        data = {"questions": [mcq_item(prompt="یہ جملہ درست ہے؟")]}
        with self.assertRaisesRegex(ValueError, "Urdu text in an English quiz"):
            validate_quiz(data, self.math_plan, 1)

    def test_rejects_english_only_text_in_urdu_quiz(self):
        urdu_plan = QuizPlan(
            mode="topic_only",
            topic="اردو قواعد",
            domain="humanities",
            kinds=["mcq"],
            language="Urdu",
        )
        with self.assertRaisesRegex(ValueError, "English-only text in an Urdu quiz"):
            validate_quiz({"questions": [mcq_item()]}, urdu_plan, 1)

    def test_rejects_wrong_option_count_and_index(self):
        data = {"questions": [mcq_item()]}
        data["questions"][0]["options"] = ["a", "b", "c"]
        with self.assertRaisesRegex(ValueError, "exactly 4 options"):
            validate_quiz(data, self.math_plan, 1)

        data = {"questions": [mcq_item()]}
        data["questions"][0]["correct_index"] = 9
        with self.assertRaisesRegex(ValueError, "correct_index"):
            validate_quiz(data, self.math_plan, 1)

    def test_missing_kind_defaults_to_the_single_allowed_kind(self):
        item = mcq_item()
        del item["kind"]
        questions = validate_quiz({"questions": [item]}, self.math_plan, 1)
        self.assertEqual(questions[0].kind.value, "mcq")

    def test_rejects_too_few_questions(self):
        data = {"questions": [mcq_item(), mcq_item()]}
        with self.assertRaisesRegex(ValueError, "expected about 5"):
            validate_quiz(data, self.math_plan, 5)


class Rule2DetectionTests(unittest.TestCase):
    def test_detect_dev_language(self):
        cases = {
            "Python Foundations": "python",
            "React Hooks": "react",
            "JavaScript basics": "javascript",
            "TypeScript generics": "typescript",
            "Next.js routing": "javascript",
            "C++ pointers": "c++",
            "C# generics": "c#",
            "Kotlin coroutines": "kotlin",
            "Ruby gems": "ruby",
            "Rust ownership": "rust",
            "HTML forms": "html",
            "CSS grid": "css",
            "CMD basics": "bash",
            "learn golang": "go",
            "Swift structs": "swift",
            "SQL joins": "sql",
        }
        for text, expected in cases.items():
            self.assertEqual(detect_dev_language(text), expected, text)

    def test_word_boundaries(self):
        self.assertIsNone(detect_dev_language("reaction kinetics"))  # react
        self.assertIsNone(detect_dev_language("English Grammar"))
        self.assertIsNone(detect_dev_language("History of Rome"))
        self.assertIsNone(detect_dev_language(""))
        self.assertEqual(detect_dev_language("javascript tutorial"), "javascript")

    def test_rule2_course_title_wins_without_llm_or_save(self):
        bundle = make_bundle(
            course={"id": 3, "title": "Ruby Fundamentals", "domain": "", "description": ""}
        )
        with patch(
            "app.quiz_service.fetch_course_bundle", AsyncMock(return_value=bundle)
        ), patch(
            "app.quiz_service.save_course_domain", AsyncMock(return_value=True)
        ) as save:
            plan = run(build_plan(make_req(course_id=1), "tok", None, llm_never))
        self.assertEqual(plan.domain, "programming")
        self.assertEqual(plan.coding_language, "ruby")
        self.assertEqual(plan.kinds, ["mcq"])  # the UI asked for MCQs only
        save.assert_not_awaited()

    def test_rule2_topic_detection(self):
        plan = run(build_plan(make_req(topic="CMD basics", kind="code"), None, None, llm_never))
        self.assertEqual(plan.domain, "programming")
        self.assertEqual(plan.coding_language, "bash")
        self.assertEqual(plan.kinds, ["code"])

    def test_rule2_detected_language_drives_the_prompt(self):
        plan = run(build_plan(make_req(topic="HTML forms", kind="code"), None, None, llm_never))
        system = build_system_prompt(plan, 5, "medium")
        self.assertIn("Programming course: true", system)
        self.assertIn("Coding language for code questions: html", system)
        self.assertNotIn("Coding language for code questions: python", system)


class Rule3QuestionTypeTests(unittest.TestCase):
    def setUp(self):
        self.plan = QuizPlan(
            mode="topic_only",
            topic="Python loops",
            domain="programming",
            kinds=["mcq"],
            language="English",
            coding_language="python",
        )

    def test_prompt_offers_only_mcq_shape(self):
        system = build_system_prompt(self.plan, 5, "medium")
        self.assertIn("Allowed question kinds: mcq", system)
        self.assertIn("3. QUESTION TYPES:", system)
        self.assertNotIn('"kind": "code"', system)
        self.assertNotIn('"kind": "debug"', system)

    def test_code_reply_is_rejected(self):
        data = {
            "questions": [
                {
                    "kind": "code",
                    "prompt": "Write a loop",
                    "reference_solution": "for i in range(3): pass",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_quiz(data, self.plan, 1)


class LightDebugValidatorTests(unittest.TestCase):
    def setUp(self):
        self.plan = QuizPlan(
            mode="topic_only",
            topic="Python bugs",
            domain="programming",
            kinds=["debug"],
            language="English",
            coding_language="python",
        )

    def item(self, **overrides) -> dict:
        base = {
            "kind": "debug",
            "prompt": "Fix the bug",
            "code": "x = 1\nprint(x)\ny = 2",
            "buggy_line": 2,
            "reference_solution": "x = 1\ny = 2\nprint(y)",
            "explanation": "The order is wrong.",
        }
        base.update(overrides)
        return base

    def test_valid_debug_passes_and_gets_the_detected_language(self):
        questions = validate_quiz({"questions": [self.item()]}, self.plan, 1)
        self.assertEqual(questions[0].language, "python")

    def test_buggy_line_must_point_at_a_real_line(self):
        with self.assertRaisesRegex(ValueError, "must point at a line"):
            validate_quiz({"questions": [self.item(buggy_line=0)]}, self.plan, 1)
        with self.assertRaisesRegex(ValueError, "must point at a line"):
            validate_quiz({"questions": [self.item(buggy_line=9)]}, self.plan, 1)

    def test_fix_is_required_and_must_change_the_code(self):
        with self.assertRaisesRegex(ValueError, "needs a reference_solution"):
            validate_quiz({"questions": [self.item(reference_solution="")]}, self.plan, 1)
        with self.assertRaisesRegex(ValueError, "identical to the buggy code"):
            validate_quiz(
                {"questions": [self.item(reference_solution="x = 1\nprint(x)\ny = 2")]},
                self.plan,
                1,
            )

    def test_debug_nudge_names_the_fix_fields(self):
        calls: list[str] = []

        async def flaky(system: str, user: str) -> ai_service.AIResult:
            calls.append(user)
            if len(calls) == 1:
                text = json.dumps({"questions": [self.item(buggy_line=9)]})
            else:
                text = json.dumps({"questions": [self.item()]})
            return ai_service.AIResult(text=text, model="mock-model", task=ai_service.Task.QUIZ)

        questions, _ = run(generate_quiz(self.plan, 1, "medium", flaky))
        self.assertEqual(len(questions), 1)
        self.assertIn("must be a real 1-based line", calls[1])


class LanguageDecisionTests(unittest.TestCase):
    def test_detected_language_overrides_the_models_guess(self):
        plan = QuizPlan(
            mode="topic_only",
            topic="Python loops",
            domain="programming",
            kinds=["code"],
            language="English",
            coding_language="python",
        )
        item = {
            "kind": "code",
            "prompt": "Write a function",
            "language": "ruby",
            "reference_solution": "def f(): pass",
        }
        questions = validate_quiz({"questions": [item]}, plan, 1)
        self.assertEqual(questions[0].language, "python")

    def test_undetected_language_passes_the_models_value_through(self):
        plan = QuizPlan(
            mode="topic_only",
            topic="Algorithm design",
            domain="programming",
            kinds=["code"],
            language="English",
            coding_language="",
        )
        item = {
            "kind": "code",
            "prompt": "Write a function",
            "language": "ruby",
            "reference_solution": "def f(): pass",
        }
        questions = validate_quiz({"questions": [item]}, plan, 1)
        self.assertEqual(questions[0].language, "ruby")

    def test_falls_back_to_the_question_content_when_the_title_said_nothing(self):
        # "HTLM BASICS" (typo) matches no title marker - the question itself
        # says "HTML snippet", so the badge must show html, not python.
        plan = QuizPlan(
            mode="topic_only",
            topic="HTLM BASICS",
            domain="programming",
            kinds=["output"],
            language="English",
            coding_language="",
        )
        item = {
            "kind": "output",
            "prompt": "What is the exact output of the HTML snippet when rendered in a browser?",
            "code": '<a href="https://example.com">Click here</a>',
            "correct_output": "Click here",
            "explanation": "The anchor renders as a link.",
        }
        questions = validate_quiz({"questions": [item]}, plan, 1)
        self.assertEqual(questions[0].language, "html")

    def test_no_language_anywhere_stays_empty(self):
        plan = QuizPlan(
            mode="topic_only",
            topic="Algorithms",
            domain="programming",
            kinds=["code"],
            language="English",
            coding_language="",
        )
        item = {
            "kind": "code",
            "prompt": "Write a sorting routine",
            "reference_solution": "def sort(a): return sorted(a)",
        }
        questions = validate_quiz({"questions": [item]}, plan, 1)
        self.assertEqual(questions[0].language, "")


class GenerateQuizTests(unittest.TestCase):
    def setUp(self):
        self.plan = QuizPlan(
            mode="topic_only",
            topic="Basic Math",
            domain="math",
            kinds=["mcq"],
            language="English",
        )

    def test_retry_after_refusal_then_success(self):
        calls: list[str] = []

        async def flaky(system: str, user: str) -> ai_service.AIResult:
            calls.append(user)
            if len(calls) == 1:
                return ai_service.AIResult(
                    text="I can't comply with that request.",
                    model="mock-model",
                    task=ai_service.Task.QUIZ,
                )
            return ai_service.AIResult(
                text=mcq_json(3), model="mock-model", task=ai_service.Task.QUIZ
            )

        questions, model = run(generate_quiz(self.plan, 3, "medium", flaky))
        self.assertEqual(len(questions), 3)
        self.assertEqual(model, "mock-model")
        self.assertEqual(len(calls), 2)
        self.assertIn("could not be used", calls[1])
        self.assertIn("English", calls[1])  # nudge names the quiz language

    def test_latex_reply_is_sanitized_not_retried(self):
        latex_reply = json.dumps(
            {
                "questions": [
                    {
                        "kind": "mcq",
                        "prompt": "What is \\frac{1}{2} + \\frac{1}{2}?",
                        "options": ["1/2", "1", "2", "3"],
                        "correct_index": 1,
                        "explanation": "Half plus half is 1.",
                    }
                ]
            }
        )

        async def once(system: str, user: str) -> ai_service.AIResult:
            return ai_service.AIResult(
                text=latex_reply, model="mock-model", task=ai_service.Task.QUIZ
            )

        questions, _ = run(generate_quiz(self.plan, 1, "medium", once))
        self.assertIn("1/2", questions[0].prompt)

    def test_every_attempt_failing_raises_runtime_error(self):
        async def always_bad(system: str, user: str) -> ai_service.AIResult:
            return ai_service.AIResult(
                text="not json", model="mock-model", task=ai_service.Task.QUIZ
            )

        with self.assertRaises(RuntimeError):
            run(generate_quiz(self.plan, 3, "medium", always_bad))


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(quizzes_router.router)
    # Routes require a valid Supabase JWT now; skip auth in unit tests.
    app.dependency_overrides[get_current_user] = lambda: "test-user-id"
    return TestClient(app)


class RouteTests(unittest.TestCase):
    def test_s8_7_general_empty_is_400(self):
        client = make_client()
        response = client.post(
            "/api/quizzes/generate",
            json={
                "topic": "",
                "kind": "mcq",
                "count": 5,
                "difficulty": "medium",
                "language": "python",
                "context": "",
                "course_id": None,
                "subject": "",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Write a topic", response.json()["detail"])

    def test_s8_1_route_prompt_and_response(self):
        captured: list[list[dict]] = []

        async def fake_complete(task, messages, **kwargs):
            captured.append(list(messages))
            return ai_service.AIResult(
                text=mcq_json(5), model="mock-model", task=task
            )

        client = make_client()
        with patch("app.ai_service.complete", fake_complete):
            response = client.post(
                "/api/quizzes/generate",
                json={
                    "topic": "Basic Math",
                    "kind": "code",  # even a code request must stay MCQ
                    "count": 5,
                    "difficulty": "medium",
                    "language": "python",
                    "context": "PYTHON NOTES: def foo(): print(1)",
                    "course_id": None,
                    "subject": "programming",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["model"], "mock-model")
        self.assertEqual(len(body["questions"]), 5)
        self.assertTrue(all(q["kind"] == "mcq" for q in body["questions"]))

        self.assertEqual(len(captured), 1)
        system = captured[0][0]["content"]
        user = captured[0][1]["content"]
        self.assertIn("Subject area: math", system)
        self.assertIn("Allowed question kinds: mcq", system)
        self.assertNotIn("STRICTLY", system)
        self.assertNotIn("FORBIDDEN", system)
        # The CRITICAL RULES block may name Python as its example, but the
        # client's context/subject must never leak into the prompt as material.
        self.assertNotIn("python notes", system.lower())
        self.assertNotIn("def foo", system.lower())
        self.assertNotIn("print(1)", system.lower())
        self.assertTrue(system.startswith("CRITICAL RULES:"), "rules must be first")
        self.assertIn("Programming course: false", system)
        self.assertEqual(user, "Create the quiz now.")  # zero notes sent


if __name__ == "__main__":
    unittest.main()
