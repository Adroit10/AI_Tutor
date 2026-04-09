from __future__ import annotations

import os
import json
import textwrap
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))
except ImportError:
    GROQ_AVAILABLE = False
    _groq_client = None


try:
    from openai import OpenAI as _OAI
    _or_client = _OAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    OR_AVAILABLE = bool(os.environ.get("OPENROUTER_API_KEY"))
except ImportError:
    OR_AVAILABLE = False
    _or_client = None


GROQ_MODEL = "llama-3.3-70b-versatile"

OR_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

Level = Literal["beginner", "intermediate", "advanced"]


def _chat(
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
   
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    if GROQ_AVAILABLE:
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Groq] Error: {e} — trying OpenRouter fallback")

    if OR_AVAILABLE:
        try:
            resp = _or_client.chat.completions.create(
                model=OR_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenRouter] Error: {e}")

    raise RuntimeError(
        "\n\nNo LLM backend available.\n"
        "Set one of:\n"
        "  export GROQ_API_KEY='gsk_...'          # free at console.groq.com\n"
        "  export OPENROUTER_API_KEY='sk-or-...'  # free at openrouter.ai\n"
    )


_TUTOR_SYSTEM = textwrap.dedent("""\
    You are an expert AI tutor with deep knowledge across STEM subjects.
    Your teaching style is Socratic, clear, and builds genuine intuition.
    Always prefer concrete examples over abstract statements.
    Use analogies the student can relate to based on their level.
    Never truncate your answer — finish every section fully.
""")

_QUIZ_SYSTEM = textwrap.dedent("""\
    You are an expert educational assessment designer.
    Create well-crafted quizzes that test conceptual understanding,
    not rote memorization.  Always include the answer key.
""")

_FOLLOWUP_SYSTEM = textwrap.dedent("""\
    You are an expert tutor identifying gaps in student understanding.
    Your follow-up questions should probe deeper reasoning,
    reveal misconceptions, and encourage independent thinking.
""")

_DIAGRAM_SYSTEM = textwrap.dedent("""\
    You are an expert at converting educational explanations into
    clean Mermaid.js diagram definitions.
    Output ONLY valid Mermaid syntax — no prose, no code fences.
    Prefer flowchart TD or graph LR for concept maps.
    Keep labels short (≤ 6 words each).
""")


def _level_note(level: Level) -> str:
    notes = {
        "beginner":     "Use very simple language, analogies to everyday life, and avoid jargon.",
        "intermediate": "Assume basic familiarity; focus on mechanisms and edge cases.",
        "advanced":     "Use precise technical language; include mathematical derivations where relevant.",
    }
    return notes.get(level, notes["beginner"])


def build_prompt(query: str, context: str, level: Level = "beginner") -> str:
    
    return textwrap.dedent(f"""\
        Level: {level}
        Audience note: {_level_note(level)}

        === Retrieved Context ===
        {context}
        === End Context ===

        Question: {query}

        Answer using this STRICT structure:
        1. **Intuition** - one real-life analogy that clicks immediately
        2. **Step-by-step explanation** - numbered, detailed steps
        3. **Mathematical intuition** - only if applicable
        4. **Real-world example** - concrete and specific
        5. **Common mistakes / misconceptions** - warn the student
        6. **Summary** - 3-4 crisp bullet points

        Think step by step before writing each section (Chain-of-Thought).
        Do NOT skip sections.  Do NOT truncate.
    """)


def generate_answer(query: str, context: str, level: Level = "beginner") -> str:
    """Generate a full structured tutor explanation."""
    user_msg = build_prompt(query, context, level)
    return _chat(_TUTOR_SYSTEM, user_msg, max_tokens=1500, temperature=0.6)


def generate_quiz(query: str, context: str) -> str:
    """Generate 3 MCQs + 2 short-answer questions with an answer key."""
    user_msg = textwrap.dedent(f"""\
        Topic: {query}

        Context:
        {context}

        Create:
        • 3 multiple-choice questions (4 options each, one correct)
        • 2 short-answer questions requiring conceptual reasoning

        Format each MCQ as:
        Q1. <question>
        A) ...  B) ...  C) ...  D) ...
        ✓ Correct: <letter>

        Format short-answer as:
        SA1. <question>
        Expected answer: <answer>

        Make questions probe understanding, not just recall.
    """)
    return _chat(_QUIZ_SYSTEM, user_msg, max_tokens=800, temperature=0.5)


def generate_followup(question: str, answer: str) -> str:
  
    user_msg = textwrap.dedent(f"""\
        Original question: {question}

        Tutor's explanation:
        {answer}

        Generate 3 follow-up questions that:
        1. Test whether the student truly understood (not just memorised)
        2. Reveal a common misconception related to this topic
        3. Encourage the student to connect this concept to something new

        Number them FQ1, FQ2, FQ3.
    """)
    return _chat(_FOLLOWUP_SYSTEM, user_msg, max_tokens=400, temperature=0.7)


def generate_diagram_prompt(query: str, context: str = "", level: Level = "beginner") -> str:
    
    user_msg = textwrap.dedent(f"""\
        Topic: {query}
        Student level: {level}

        Relevant context:
        {context[:1200]}

        Produce a Mermaid.js diagram that shows the key concepts and
        their relationships for this topic.
        Use flowchart TD (top-down) or graph LR (left-right).
        Include 6-12 nodes.  Keep labels concise.
        Output ONLY the raw Mermaid definition — nothing else.
    """)
    return _chat(_DIAGRAM_SYSTEM, user_msg, max_tokens=400, temperature=0.3)


def build_finetune_sample(
    query: str,
    context: str,
    answer: str,
    level: Level = "beginner",
) -> dict:

    instruction = build_prompt(query, context, level)
    return {
        "instruction": instruction,
        "input": "", 
        "output": answer,
        "level": level,
    }