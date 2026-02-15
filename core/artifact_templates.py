"""
Artifact Generation Templates for Collabry Study Assistant.

This module contains specialized prompt templates for generating
structured artifacts (quizzes, mindmaps, flashcards, etc.) that
can be saved to the Studio and rendered in specialized components.

These templates ensure consistent output formatting that matches
the frontend component expectations.
"""

# =========================
# QUIZ GENERATION
# =========================

QUIZ_GENERATION_TEMPLATE = """You are a quiz generator. Generate {num_questions} multiple-choice questions about: {topics}

Difficulty level: {difficulty}
{prompt}

**IMPORTANT: Your response MUST be a valid JSON array. Do NOT write any text before or after the JSON.**

Required JSON structure:
[
  {{
    "question": "What is the time complexity of binary search?",
    "options": ["O(1)", "O(n)", "O(log n)", "O(n^2)"],
    "correctAnswer": 2,
    "explanation": "Binary search divides the array in half each time."
  }},
  {{
    "question": "Which data structure uses FIFO?",
    "options": ["Stack", "Queue", "Array", "Tree"],
    "correctAnswer": 1,
    "explanation": "Queue follows First-In-First-Out principle."
  }}
]

Rules:
1. Start your response with [ character
2. End your response with ] character
3. Each question must have:
   - "question": string (the question text)
   - "options": array of exactly 4 strings
   - "correctAnswer": number (0, 1, 2, or 3 - the index of correct answer in options array)
   - "explanation": string (why the answer is correct)
4. Generate exactly {num_questions} questions
5. Separate questions with comma
6. Base questions on provided source material
7. Make wrong options plausible
8. Difficulty: {difficulty}

DO NOT include:
- Markdown code blocks (no ```)
- "Answer: A" format
- "Question 1:" labels
- Any text outside the JSON array

Your first character must be [
Your last character must be ]"""

DEFAULT_QUIZ_PROMPT = "Create a practice quiz with multiple choice questions about:"

# =========================
# MINDMAP GENERATION
# =========================

MINDMAP_GENERATION_TEMPLATE = """Create a mind map about: {topics}

Use the content from the selected sources to build the mind map structure.

Output ONLY a JSON object with this exact structure (no markdown, no code blocks, no extra text):
{{
  "nodes": [
    {{"id": "root", "label": "{topics}", "level": 0}}
  ],
  "edges": []
}}

Requirements:
- 10-20 nodes total based on the source content
- 2-3 levels deep showing concept hierarchy
- Labels should be 2-5 words extracted from source material
- Connect related concepts with edges (parent-child relationships)
- Each node id must be unique (use "node-" prefix with numbers)
Output ONLY the JSON object, nothing else."""

# =========================
# FLASHCARDS GENERATION
# =========================

FLASHCARDS_GENERATION_TEMPLATE = """Create flashcards for studying: {topics}

Use the content from the selected sources to create study flashcards.

Output ONLY a JSON object with this exact structure (no markdown, no code blocks):
{{
  "title": "Flashcards: {topics}",
  "cards": [
    {{
      "id": "card-1",
      "front": "What is...",
      "back": "The answer is...",
      "category": "Basics"
    }}
  ]
}}

Requirements:
- Generate 15-20 flashcards based on source content
- Front: Clear question or concept to test
- Back: Concise answer or explanation (2-3 sentences max)
- Category: Group related cards (Basics, Definitions, Applications, etc.)
- Cover key concepts, definitions, processes, and applications
- Make questions specific and answers clear
Output ONLY the JSON object, nothing else."""

# =========================
# COURSE FINDER
# =========================

COURSE_FINDER_TEMPLATE = """Generate a list of recommended online courses about "{topics}".

FORMAT EACH COURSE EXACTLY LIKE THIS:

**[Arrays for Beginners - Master Data Structures](https://www.udemy.com/course/arrays-for-beginners)**  
Platform: Udemy | Rating: 4.5/5 | Price: $22

FORMAT RULES (CRITICAL):
1. Start with ** (two asterisks)
2. Then [ (opening square bracket)
3. Then the course title
4. Then ] (closing square bracket)
5. Then ( (opening parenthesis)
6. Then the full URL starting with https://
7. Then ) (closing parenthesis)
8. Then ** (two asterisks)
9. Then TWO SPACES
10. Then press Enter
11. Next line starts with "Platform: "
12. Then another blank line before the next course

EXAMPLE (copy this exactly):

**[Complete Arrays and Lists Tutorial](https://www.coursera.org/learn/arrays-tutorial)**  
Platform: Coursera | Rating: 4.8/5 | Price: Free

**[Data Structures with Python - Arrays Module](https://www.edx.org/course/python-arrays)**  
Platform: edX | Rating: 4.6/5 | Price: $99

Generate 5 courses about "{topics}" in this exact format.
Use these platform URL patterns:
- Udemy: https://www.udemy.com/course/topic-name
- Coursera: https://www.coursera.org/learn/topic-name
- edX: https://www.edx.org/course/topic-name
- Codecademy: https://www.codecademy.com/learn/topic-name

START with the first course. No introduction text."""

# =========================
# REPORTS GENERATION
# =========================

REPORTS_GENERATION_TEMPLATE = """Generate a comprehensive study report for: {topics}

Analyze the selected source materials and create a structured report with the following sections:

1. Executive Summary (2-3 paragraphs)
2. Key Concepts (5-10 main concepts with explanations)
3. Learning Objectives (What should be mastered)
4. Detailed Analysis (Deep dive into important topics)
5. Practical Applications (Real-world use cases)
6. Study Recommendations (How to learn this effectively)
7. Assessment Criteria (What to focus on for testing)
8. Additional Resources (Recommended readings/videos)

Format the report in clear markdown with headers, bullet points, and emphasis.
Base all content on the actual source material provided.
Make it comprehensive but readable (aim for 800-1200 words).
Output ONLY the markdown report, nothing else."""

# =========================
# INFOGRAPHIC GENERATION
# =========================

INFOGRAPHIC_GENERATION_TEMPLATE = """Create an infographic data structure for: {topics}

Analyze the selected sources and extract key visual elements.

Output ONLY a JSON object with this structure (no markdown, no code blocks):
{{
  "title": "Main topic title",
  "subtitle": "Brief description",
  "sections": [
    {{
      "type": "stats",
      "title": "Key Statistics",
      "items": [
        {{"label": "Statistic 1", "value": "85%", "icon": "📊"}},
        {{"label": "Statistic 2", "value": "2.5M", "icon": "👥"}}
      ]
    }},
    {{
      "type": "timeline",
      "title": "Timeline",
      "events": [
        {{"year": "2020", "event": "Key event description"}},
        {{"year": "2021", "event": "Another event"}}
      ]
    }},
    {{
      "type": "comparison",
      "title": "Comparison",
      "items": [
        {{
          "label": "Option A",
          "pros": ["Pro 1", "Pro 2"],
          "cons": ["Con 1"]
        }},
        {{
          "label": "Option B",
          "pros": ["Pro 1"],
          "cons": ["Con 1", "Con 2"]
        }}
      ]
    }}
  ]
}}

Requirements:
- Extract 3-5 sections from source content
- Include real data/statistics when available
- Use relevant emojis for icons
- Timeline events should be chronological if source has historical context
- Comparisons should highlight key differences
Output ONLY the JSON object."""

# =========================
# STUDY PLAN GENERATION
# =========================

STUDY_PLAN_TEMPLATE = """Create a detailed study plan for: {topics}

Duration: {duration} (e.g., "1 week", "2 weeks", "1 month")
Study hours per day: {hours_per_day}

Analyze the selected source materials and create a structured study plan.

Output format (Markdown):

# Study Plan: {topics}

## Overview
Brief description of what will be covered and learning objectives.

## Week-by-Week Breakdown

### Week 1: [Topic Focus]
**Learning Objectives:**
- Objective 1
- Objective 2

**Daily Schedule:**

**Day 1:**
- 09:00-10:00: [Activity] - [Specific topic]
- 10:15-11:00: [Activity] - [Specific topic]
- 11:00-11:30: Practice exercises

**Day 2:**
[Continue pattern...]

**Resources Needed:**
- Source material references
- Additional resources if needed

**Checkpoints:**
- End of week assessment: [What to verify you've learned]

[Repeat for each week...]

## Tips for Success
- Practical advice for following the plan
- How to adjust if falling behind
- Study techniques to use

Requirements:
- Base the plan on actual source content
- Be realistic with time allocations
- Include breaks and review sessions
- Provide specific topics/chapters for each day
- Include checkpoints to verify progress
Output ONLY the markdown study plan."""

# =========================
# PRACTICE PROBLEMS GENERATION
# =========================

PRACTICE_PROBLEMS_TEMPLATE = """Generate practice problems for: {topics}

Difficulty: {difficulty}
Number of problems: {num_problems}

Use the content from the selected sources to create practice problems.

Format each problem as:

**Problem [N]:** [Problem statement with all necessary information]

**Solution:**
[Step-by-step solution with explanations]

**Key Concepts:**
- Concept 1
- Concept 2

**Difficulty:** {difficulty}

---

Requirements:
- Generate exactly {num_problems} problems
- Each problem should test understanding and application
- Solutions should be detailed with step-by-step explanations
- Problems should increase in complexity
- Cover different aspects of the topic
- Base problems on source material content
Output problems in the format above, no extra text."""

# =========================
# SUMMARY GENERATION
# =========================

SUMMARY_GENERATION_TEMPLATE = """Create a comprehensive summary of: {topics}

Length: {length} (brief/moderate/detailed)

Analyze the selected sources and create a well-structured summary.

Format:

## {topics} - Summary

### Main Concepts
Brief overview connecting all major ideas.

### Key Points
- **Point 1:** Explanation with details from sources
- **Point 2:** Explanation with details from sources
- **Point 3:** Explanation with details from sources
[Continue for all major points...]

### Important Definitions
- **Term 1:** Definition and context
- **Term 2:** Definition and context
[Continue for all key terms...]

### Critical Insights
[2-3 paragraphs discussing deeper understanding, connections, and implications]

### Key Takeaways
1. Takeaway 1
2. Takeaway 2
3. Takeaway 3

Requirements:
- Base entirely on source content
- Be accurate and comprehensive
- Use clear, concise language
- Highlight connections between concepts
- Length guidance: brief (300-500 words), moderate (500-800 words), detailed (800-1200 words)
Output ONLY the markdown summary."""

# =========================
# CONCEPT MAP GENERATION
# =========================

CONCEPT_MAP_TEMPLATE = """Create a concept map for: {topics}

Use the content from the selected sources to build a concept map showing relationships.

Output ONLY a JSON object (no markdown, no code blocks):
{{
  "concepts": [
    {{
      "id": "concept-1",
      "label": "Main Concept",
      "description": "Brief description",
      "level": 0
    }}
  ],
  "relationships": [
    {{
      "from": "concept-1",
      "to": "concept-2",
      "type": "leads-to",
      "label": "relationship description"
    }}
  ]
}}

Relationship types:
- "leads-to": Causal or sequential relationship
- "part-of": Hierarchical relationship  
- "relates-to": General association
- "contrasts-with": Opposing concepts
- "example-of": Specific instance

Requirements:
- 15-25 concepts based on source content
- Clear, concise labels (2-4 words)
- Meaningful relationships with descriptive labels
- Multiple levels of hierarchy
- Cover breadth and depth of topic
Output ONLY the JSON object."""

# =========================
# HELPER FUNCTIONS
# =========================

def format_quiz_prompt(
    topics: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    custom_prompt: str = None
) -> str:
    """Format quiz generation prompt with parameters."""
    prompt = custom_prompt or DEFAULT_QUIZ_PROMPT
    return QUIZ_GENERATION_TEMPLATE.format(
        prompt=prompt,
        topics=topics,
        num_questions=num_questions,
        difficulty=difficulty
    )

def format_mindmap_prompt(topics: str) -> str:
    """Format mindmap generation prompt."""
    return MINDMAP_GENERATION_TEMPLATE.format(topics=topics)

def format_flashcards_prompt(topics: str) -> str:
    """Format flashcards generation prompt."""
    return FLASHCARDS_GENERATION_TEMPLATE.format(topics=topics)

def format_course_finder_prompt(topics: str) -> str:
    """Format course finder prompt."""
    return COURSE_FINDER_TEMPLATE.format(topics=topics)

def format_reports_prompt(topics: str) -> str:
    """Format reports generation prompt."""
    return REPORTS_GENERATION_TEMPLATE.format(topics=topics)

def format_infographic_prompt(topics: str) -> str:
    """Format infographic generation prompt."""
    return INFOGRAPHIC_GENERATION_TEMPLATE.format(topics=topics)

def format_study_plan_prompt(
    topics: str,
    duration: str = "2 weeks",
    hours_per_day: int = 2
) -> str:
    """Format study plan generation prompt."""
    return STUDY_PLAN_TEMPLATE.format(
        topics=topics,
        duration=duration,
        hours_per_day=hours_per_day
    )

def format_practice_problems_prompt(
    topics: str,
    num_problems: int = 5,
    difficulty: str = "medium"
) -> str:
    """Format practice problems generation prompt."""
    return PRACTICE_PROBLEMS_TEMPLATE.format(
        topics=topics,
        num_problems=num_problems,
        difficulty=difficulty
    )

def format_summary_prompt(
    topics: str,
    length: str = "moderate"
) -> str:
    """Format summary generation prompt."""
    return SUMMARY_GENERATION_TEMPLATE.format(
        topics=topics,
        length=length
    )

def format_concept_map_prompt(topics: str) -> str:
    """Format concept map generation prompt."""
    return CONCEPT_MAP_TEMPLATE.format(topics=topics)


def post_process_course_output(text: str) -> str:
    """
    Post-process course output to ensure proper formatting.
    
    Takes plain text course listings and converts them to proper markdown format.
    """
    import re
    
    lines = text.strip().split('\n')
    formatted_courses = []
    current_course = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with "Platform:"
        if line.startswith('Platform:'):
            if current_course:
                # Add platform info to current course
                formatted_courses.append(f"{current_course}  ")
                formatted_courses.append(line)
                formatted_courses.append("")  # Blank line
                current_course = None
        else:
            # This is a course title line
            # Try to extract course info
            if 'Platform:' in line:
                # Course and platform on same line - split them
                parts = line.split('Platform:', 1)
                course_title = parts[0].strip()
                platform_info = 'Platform:' + parts[1].strip()
                
                # Format course title as markdown link if not already
                if not course_title.startswith('**['):
                    # Generate a URL based on the course title and platform
                    platform_lower = platform_info.lower()
                    course_slug = re.sub(r'[^a-z0-9]+', '-', course_title.lower()).strip('-')
                    
                    if 'udemy' in platform_lower:
                        url = f"https://www.udemy.com/course/{course_slug}/"
                    elif 'coursera' in platform_lower:
                        url = f"https://www.coursera.org/learn/{course_slug}"
                    elif 'edx' in platform_lower:
                        url = f"https://www.edx.org/course/{course_slug}"
                    elif 'codecademy' in platform_lower:
                        url = f"https://www.codecademy.com/learn/{course_slug}"
                    elif 'pluralsight' in platform_lower:
                        url = f"https://www.pluralsight.com/courses/{course_slug}"
                    else:
                        url = f"https://example.com/courses/{course_slug}"
                    
                    formatted_courses.append(f"**[{course_title}]({url})**  ")
                else:
                    formatted_courses.append(f"{course_title}  ")
                
                formatted_courses.append(platform_info)
                formatted_courses.append("")  # Blank line
            else:
                # Store for next iteration
                current_course = line
    
    # Handle any remaining course
    if current_course:
        formatted_courses.append(f"{current_course}  ")
        formatted_courses.append("")
    
    return '\n'.join(formatted_courses)

# =========================
# ARTIFACT TYPE DETECTION
# =========================

ARTIFACT_PATTERNS = {
    'quiz': [
        'create a quiz', 'make a quiz', 'test me', 'practice quiz',
        'generate questions', 'multiple choice', 'mcq'
    ],
    'mindmap': [
        'create a mind map', 'make a mindmap', 'mind map',
        'concept map', 'visual map', 'topic map'
    ],
    'flashcards': [
        'create flashcards', 'make flashcards', 'flash cards',
        'study cards', 'memory cards'
    ],
    'course-finder': [
        'find courses', 'recommend courses', 'best courses',
        'online courses', 'where to learn', 'course suggestions'
    ],
    'reports': [
        'create a report', 'generate report', 'study report',
        'comprehensive report', 'analysis report'
    ],
    'infographic': [
        'create infographic', 'visual summary', 'infographic',
        'data visualization'
    ],
    'study-plan': [
        'create study plan', 'study schedule', 'learning plan',
        'study roadmap', 'plan my studies'
    ],
    'practice-problems': [
        'practice problems', 'exercises', 'practice questions',
        'problem set', 'work through problems'
    ],
    'summary': [
        'summarize', 'create summary', 'summary of',
        'sum up', 'brief overview'
    ]
}

def detect_artifact_type(message: str) -> str:
    """
    Detect what type of artifact the user is requesting.
    
    Returns artifact type or None if no match.
    """
    message_lower = message.lower()
    
    for artifact_type, patterns in ARTIFACT_PATTERNS.items():
        if any(pattern in message_lower for pattern in patterns):
            return artifact_type
    
    return None
