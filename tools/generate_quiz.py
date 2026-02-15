"""
Generate Quiz Tool - Create quizzes from study materials.

This tool generates multiple-choice or short-answer quizzes
based on the user's study materials.
"""

import json
from typing import Optional, List, Any
from langchain_core.tools import tool
from rag.retriever import get_retriever
from core.llm import get_async_openai_client, get_llm_config


@tool
async def generate_quiz(
    topic: Optional[str] = "main concepts",
    num_questions: Any = 10,
    difficulty: str = "medium",
    notebook_id: Optional[str] = None,
    user_id: str = "default",
    source_ids: Optional[List[str]] = None,
) -> str:
    """
    Generate a quiz based on study materials.
    
    Use this tool when the user wants to test their knowledge.
    
    Examples of when to use:
    - "Make me a quiz on biology"
    - "Test me on the French Revolution"
    - "Create 5 questions about photosynthesis"
    
    Args:
        topic: The topic for the quiz (REQUIRED)
        num_questions: Number of questions (default 10)
        difficulty: easy, medium, or hard
        notebook_id: Target notebook (optional, auto-injected)
        user_id: User identifier (optional, auto-injected)
    
    Returns:
        JSON quiz with questions, options, and correct answers
    """
    try:
        # Validate and robustly parse inputs
        try:
            if isinstance(num_questions, str):
                if num_questions.lower() in ["less", "fewer"]:
                    num_questions = 5
                elif num_questions.lower() in ["more", "many"]:
                    num_questions = 20
                else:
                    # Try to extract numbers from string if possible
                    import re
                    match = re.search(r'\d+', num_questions)
                    num_questions = int(match.group()) if match else 10
            else:
                num_questions = int(num_questions)
        except Exception:
            num_questions = 10

        num_questions = min(max(num_questions, 1), 20)  # Limit 1-20
        difficulty = difficulty.lower() if difficulty else "medium"
        
        # Get retriever
        retriever = get_retriever(
            user_id=user_id,
            notebook_id=notebook_id,
            source_ids=source_ids,
            k=15
        )
        
        # Retrieve relevant documents
        query = topic if topic else "all topics and concepts"
        docs = retriever.invoke(query)
        
        if not docs:
            return "No documents found. Please upload study materials first."
        
        # Combine content
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate quiz using LLM
        client = get_async_openai_client()
        config = get_llm_config()
        
        system_prompt = f"""You are a quiz generator for students.

Create {num_questions} multiple-choice questions at {difficulty} difficulty level.

Requirements:
- Each question must have 4 options (A, B, C, D)
- Only one correct answer per question
- Questions should test understanding, not just memorization
- Provide brief explanations for correct answers

Return ONLY valid JSON in this exact format:
{{
  "quiz_title": "Quiz on [topic]",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Brief explanation why A is correct"
    }}
  ]
}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a quiz from these study materials:\n\n{combined_text[:6000]}"}
        ]
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        quiz_json = response.choices[0].message.content
        
        # Validate JSON
        try:
            quiz_data = json.loads(quiz_json)
            return json.dumps(quiz_data, indent=2)
        except json.JSONDecodeError:
            # Extract JSON from markdown if needed
            if "```json" in quiz_json:
                quiz_json = quiz_json.split("```json")[1].split("```")[0].strip()
            elif "```" in quiz_json:
                quiz_json = quiz_json.split("```")[1].split("```")[0].strip()
            
            return quiz_json
    
    except Exception as e:
        return f"Error generating quiz: {str(e)}"
