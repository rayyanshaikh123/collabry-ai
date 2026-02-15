"""
Generate Flashcards Tool - Create study flashcards.

This tool generates flashcards from study materials
for spaced repetition learning.
"""

import json
from typing import Optional, List, Any
from langchain_core.tools import tool
from rag.retriever import get_retriever
from core.llm import get_async_openai_client, get_llm_config


@tool
async def generate_flashcards(
    topic: Optional[str] = "key concepts",
    num_cards: Any = 10,
    notebook_id: Optional[str] = None,
    user_id: str = "default",
    source_ids: Optional[List[str]] = None,
) -> str:
    """
    Generate flashcards from study materials.
    
    Use this tool when the user wants flashcards for memorization.
    
    Examples of when to use:
    - "Make me flashcards for biology"
    - "Create study cards about the French Revolution"
    - "Generate 10 flashcards on vocabulary"
    
    Args:
        topic: The topic for flashcards (REQUIRED)
        num_cards: Number of flashcards (default 10)
        notebook_id: Target notebook (optional, auto-injected)
        user_id: User identifier (optional, auto-injected)
    
    Returns:
        JSON array of flashcards with front/back content
    """
    try:
        # Validate and robustly parse inputs
        try:
            if isinstance(num_cards, str):
                if num_cards.lower() in ["less", "fewer"]:
                    num_cards = 5
                elif num_cards.lower() in ["more", "many"]:
                    num_cards = 20
                else:
                    # Try to extract numbers from string if possible
                    import re
                    match = re.search(r'\d+', num_cards)
                    num_cards = int(match.group()) if match else 10
            else:
                num_cards = int(num_cards)
        except Exception:
            num_cards = 10

        num_cards = min(max(num_cards, 1), 50)  # Limit 1-50
        
        # Get retriever
        retriever = get_retriever(
            user_id=user_id,
            notebook_id=notebook_id,
            source_ids=source_ids,
            k=15
        )
        
        # Retrieve relevant documents
        query = topic if topic else "key terms concepts and definitions"
        docs = retriever.invoke(query)
        
        if not docs:
            return "No documents found. Please upload study materials first."
        
        # Combine content
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate flashcards using LLM
        client = get_async_openai_client()
        config = get_llm_config()
        
        system_prompt = f"""You are a flashcard generator for students.

Create {num_cards} flashcards for effective studying and memorization.

Requirements:
- Front: Clear, concise question or term
- Back: Accurate, detailed answer or definition
- Focus on key concepts, definitions, and important facts
- Make cards specific and testable
- Include examples where helpful

Return ONLY valid JSON in this exact format:
{{
  "title": "Flashcards: [topic]",
  "cards": [
    {{
      "front": "Question or term",
      "back": "Answer or definition",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate flashcards from these study materials:\n\n{combined_text[:6000]}"}
        ]
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.6,
            max_tokens=2500
        )
        
        flashcards_json = response.choices[0].message.content
        
        # Validate JSON
        try:
            flashcards_data = json.loads(flashcards_json)
            return json.dumps(flashcards_data, indent=2)
        except json.JSONDecodeError:
            # Extract JSON from markdown if needed
            if "```json" in flashcards_json:
                flashcards_json = flashcards_json.split("```json")[1].split("```")[0].strip()
            elif "```" in flashcards_json:
                flashcards_json = flashcards_json.split("```")[1].split("```")[0].strip()
            
            return flashcards_json
    
    except Exception as e:
        return f"Error generating flashcards: {str(e)}"
