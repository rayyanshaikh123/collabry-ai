from typing import Dict, Any


def build_quiz_prompt(topic: str, params: Dict[str, Any] | None = None) -> str:
  """
  Server-side mirror of the quiz generation prompt previously built on the frontend.
  The textual instructions are preserved to avoid changing generation behavior.
  """
  params = params or {}
  num_questions = params.get("numberOfQuestions") or params.get("num_questions") or 5
  difficulty = params.get("difficulty") or "medium"
  base_prompt = (params.get("prompt") or "").strip()
  if not base_prompt:
      base_prompt = "Create a practice quiz with exactly 5 multiple choice questions about:"

  return f"""###QUIZ_GENERATION_REQUEST###

{base_prompt} {topic}

Return ONLY valid JSON (no markdown, no code fences):
[
  {{
    "question": "...",
    "options": ["A...", "B...", "C...", "D..."],
    "correctAnswer": 0,
    "explanation": "..."
  }}
]

Rules:
- Exactly {int(num_questions)} questions
- Exactly 4 options per question
- correctAnswer is an integer 0-3 (index into options)
- difficulty: {difficulty} (use it to tune question depth)
- Explanations must be 1-2 sentences
- JSON only"""


def build_flashcards_prompt(topic: str) -> str:
  """
  Server-side mirror of the flashcards generation prompt from the frontend.
  """
  return f"""Create flashcards for: {topic}

Return ONLY valid JSON (no markdown, no code fences):
{{
  "title": "Flashcards: {topic}",
  "cards": [
    {{"id":"card-1","front":"...","back":"...","category":"Basics"}}
  ]
}}

Rules:
- Exactly 15 cards
- front: short question/term (<= 120 chars)
- back: 1-3 concise sentences
- category: one of Basics, Definitions, Processes, Applications, Pitfalls
- Use only info implied by the selected sources
- JSON only (no extra text)"""


def build_mindmap_prompt(topic: str) -> str:
  """
  Server-side mirror of the mind map generation prompt from the frontend.
  """
  return f"""Create a mind map for: {topic}

Return ONLY valid JSON (no markdown, no code fences):
{{
  "nodes": [
    {{"id":"root","label":"{topic}","level":0}},
    {{"id":"node-1","label":"...","level":1}}
  ],
  "edges": [
    {{"from":"root","to":"node-1"}}
  ]
}}

Rules:
- 12-20 nodes
- 2-3 levels deep
- Every edge must reference existing node ids
- Short labels (2-6 words)
- JSON only"""


def build_summary_prompt(topic: str) -> str:
  """
  Server-side mirror of the 'reports' (summary) generation prompt from the frontend.
  """
  return f"""Generate a comprehensive study report for: {topic}

Analyze the selected source materials and create a structured report with the following sections:

1. Executive Summary (2-3 paragraphs)
2. Key Concepts (5-10 main concepts with explanations)
3. Learning Objectives (What should be mastered)
4. Detailed Analysis (Deep dive into important topics)
5. Practical Applications (Real-world use cases)
6. Study Recommendations (How to learn this effectively)
7. Assessment Criteria (What to focus on for testing)
8. Additional Resources (Recommended readings/videos)

Format in clear markdown with headers and bullet points.
Do not include JSON.
No preamble; output the report only."""


def build_course_finder_prompt(topic: str) -> str:
  """
  Server-side mirror of the course-finder generation prompt from the frontend.
  """
  return f"""[COURSE_FINDER_REQUEST]

Topic: {topic}

You MUST call the tool `search_web` before answering. Do not use memory.

Return ONLY valid JSON (no markdown, no code fences):
{{"tool": null, "answer": "<COURSE_LIST>"}}

Where <COURSE_LIST> is 5-8 lines. EACH line MUST be exactly:
- [Course Title](https://direct.course.url) - Platform: X | Rating: X.X/5 | Price: $X

Rules:
- Use direct course pages (not search result pages)
- If rating/price missing, write "Not provided"
- No extra commentary; JSON only"""


def build_artifact_prompt(artifact: str, topic: str, params: Dict[str, Any] | None = None) -> str:
  """
  High-level dispatcher used by API routes to map an artifact_request payload
  to the exact internal prompt string expected by the planner/agent.
  """
  artifact_lower = (artifact or "").lower()
  if artifact_lower == "quiz":
      return build_quiz_prompt(topic, params)
  if artifact_lower == "flashcards":
      return build_flashcards_prompt(topic)
  if artifact_lower == "mindmap":
      return build_mindmap_prompt(topic)
  if artifact_lower == "summary":
      return build_summary_prompt(topic)
  if artifact_lower == "course-finder":
      return build_course_finder_prompt(topic)

  # Fallback: treat as a generic topic question if the artifact type is unknown.
  return f"Please help me study the following topic in depth: {topic}"

