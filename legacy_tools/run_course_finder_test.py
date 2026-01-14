"""Run a single non-interactive test of the agent for COURSE_FINDER_REQUEST.

This script creates an agent using the default config and sends the
COURSE_FINDER_REQUEST as a single user input. It prints streaming tokens to stdout.

Usage: python legacy_tools/run_course_finder_test.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# Ensure stdout/stderr can handle unicode on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.agent import create_agent
from config import CONFIG

# Streaming printer similar to main_cli
_stream_buffer = ""

def stream_printer(chunk: str):
    try:
        print(chunk, end="", flush=True)
    except Exception:
        print(chunk)

COURSE_FINDER_REQUEST = '''[COURSE_FINDER_REQUEST]

Find the best online courses about "Arrays in Programming" from the internet.

**CRITICAL: YOU MUST USE WEB_SEARCH TOOL**
1. Call the web_search tool with queries such as: "best courses Arrays in Programming", "Arrays in Programming online course", or "Arrays in Programming tutorial course"
2. Do NOT answer from model memory — web_search must be used first.

**EXTRACTION REQUIREMENTS:**
From the web_search tool results, extract for EACH course (where available):
- Course title (exact name from the course page)
- Course URL (direct course page URL)
- Platform name (Coursera, Udemy, edX, Codecademy, etc.)
- Rating (format as X.X/5 if available)
- Price (format as $XX or "Free" if available)

**OUTPUT FORMAT - MANDATORY:**
Return a JSON object exactly like: {"tool": null, "answer": "<COURSE_LIST>"}
Where <COURSE_LIST> is the courses each on its own line, formatted as:
[Course Title](https://course.url) - Platform: X | Rating: X.X/5 | Price: $X

Requirements:
- Provide 5-8 courses when possible
- Use real course URLs (not search result pages)
- One course per line, no extra commentary
'''


def main():
    agent, llm, tools, memory = create_agent(user_id="test_user", session_id="test_session", config=CONFIG)
    print(f"Loaded tools: {list(tools.keys())}")
    print("--- BEGIN LLM OUTPUT ---")
    agent.handle_user_input_stream(COURSE_FINDER_REQUEST, stream_printer)
    print("\n--- END LLM OUTPUT ---")

if __name__ == '__main__':
    main()
