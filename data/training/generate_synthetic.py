"""
Synthetic Training Data Generator for Collabry Fine-Tuning

This script generates diverse training examples for each tool using GPT-4.
It creates variations of user queries and proper tool calling sequences.
"""

import os
import json
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any
import random
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for Collabry assistant
SYSTEM_PROMPT = """You are an intelligent study assistant with tools for searching documents, generating quizzes, flashcards, mindmaps, study plans, and summarizing content."""

# Tool definitions for generation
TOOLS = {
    "search_sources": {
        "description": "Search user's uploaded documents",
        "example_queries": [
            "What did my notes say about X?",
            "Find information about Y in my documents",
            "Search my PDFs for Z"
        ],
        "params": ["query", "user_id", "notebook_id"]
    },
    "generate_quiz": {
        "description": "Create practice quizzes",
        "example_queries": [
            "Make me a quiz on X",
            "Test me on Y",
            "Create N questions about Z"
        ],
        "params": ["notebook_id", "user_id", "num_questions", "difficulty", "topic"]
    },
    "generate_flashcards": {
        "description": "Create flashcards for memorization",
        "example_queries": [
            "Create flashcards for X",
            "Make cards to memorize Y",
            "Flashcards for Z vocabulary"
        ],
        "params": ["notebook_id", "user_id", "topic", "num_cards"]
    },
    "generate_mindmap": {
        "description": "Create visual concept maps",
        "example_queries": [
            "Create a concept map for X",
            "Visualize Y",
            "Mind map of Z"
        ],
        "params": ["topic", "user_id", "notebook_id", "include_sources"]
    },
    "generate_study_plan": {
        "description": "Generate structured study schedules",
        "example_queries": [
            "Help me plan studying X",
            "Create a study schedule for Y",
            "Make a learning plan for Z"
        ],
        "params": ["subject", "topics", "user_id", "duration_days", "daily_hours", "difficulty"]
    },
    "summarize_notes": {
        "description": "Condense large documents",
        "example_queries": [
            "Summarize my notes on X",
            "Give me the main ideas of Y",
            "TLDR my Z lecture"
        ],
        "params": ["notebook_id", "user_id", "topic", "length"]
    }
}

# Topics for diversity
TOPICS = [
    "biology", "chemistry", "physics", "mathematics", "history",
    "literature", "psychology", "economics", "computer science", "geography",
    "art history", "philosophy", "sociology", "political science", "astronomy",
    "geology", "anatomy", "statistics", "calculus", "algebra",
    "genetics", "evolution", "quantum mechanics", "thermodynamics", "electromagnetism",
    "American Revolution", "World War I", "Renaissance", "Ancient Rome", "Medieval Europe",
    "Shakespeare", "Greek mythology", "Python programming", "Machine learning", "Data structures"
]


async def generate_user_query_variant(tool_name: str, topic: str) -> str:
    """Generate a natural variant of a user query for a tool."""
    tool_info = TOOLS[tool_name]
    example_query = random.choice(tool_info["example_queries"])
    
    prompt = f"""Generate a natural user query that would require the {tool_name} tool.

Topic: {topic}
Example pattern: {example_query}

Generate ONE natural query that a student might ask. Be creative and vary the phrasing.
Return ONLY the query, no explanation.

Examples:
- "Can you quiz me on {topic}?"
- "I need flashcards for {topic}"
- "What do my notes say about {topic}?"
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()


async def generate_tool_call_example(tool_name: str, topic: str, user_query: str) -> Dict[str, Any]:
    """Generate a complete training example with tool call."""
    
    # Generate tool call arguments
    tool_args = {"user_id": "user123"}
    
    if tool_name == "search_sources":
        tool_args["query"] = topic
        tool_args["notebook_id"] = "nb456"
    
    elif tool_name == "generate_quiz":
        tool_args.update({
            "notebook_id": "nb123",
            "num_questions": random.choice([5, 10, 15]),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "topic": topic
        })
    
    elif tool_name == "generate_flashcards":
        tool_args.update({
            "notebook_id": "nb456",
            "topic": topic,
            "num_cards": random.choice([5, 10, 15, 20])
        })
    
    elif tool_name == "generate_mindmap":
        tool_args.update({
            "topic": topic,
            "notebook_id": "nb789",
            "include_sources": random.choice([True, False])
        })
    
    elif tool_name == "generate_study_plan":
        duration = random.choice([7, 14, 21, 30])
        tool_args.update({
            "subject": topic,
            "topics": f"{topic} fundamentals, advanced concepts, practice",
            "duration_days": duration,
            "daily_hours": random.choice([1.0, 1.5, 2.0, 3.0]),
            "difficulty": random.choice(["beginner", "intermediate", "advanced"])
        })
    
    elif tool_name == "summarize_notes":
        tool_args.update({
            "notebook_id": "nb789",
            "topic": topic,
            "length": random.choice(["brief", "moderate", "detailed"])
        })
    
    # Generate mock tool response and assistant message
    tool_response = await generate_mock_tool_response(tool_name, topic, tool_args)
    assistant_message = await generate_assistant_response(tool_name, topic, tool_response)
    
    # Build the training example
    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{random.randint(1, 9999)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": f"call_{random.randint(1, 9999)}",
                "content": tool_response
            },
            {"role": "assistant", "content": assistant_message}
        ]
    }
    
    return example


async def generate_mock_tool_response(tool_name: str, topic: str, args: Dict) -> str:
    """Generate a realistic mock response from a tool."""
    
    if tool_name == "search_sources":
        return f"Retrieved information about {topic}: [relevant content from user's documents about the topic]"
    
    elif tool_name == "generate_quiz":
        return json.dumps({
            "quiz": {
                "title": f"{topic} Quiz",
                "questions": [
                    {
                        "question": f"Sample question about {topic}",
                        "options": ["Option A", "Option B", "Option C", "Option D"],
                        "correct": 0,
                        "explanation": f"Explanation about {topic}"
                    }
                ]
            }
        })
    
    elif tool_name == "generate_flashcards":
        return json.dumps({
            "flashcards": [
                {"front": f"{topic} term 1", "back": "Definition 1"},
                {"front": f"{topic} term 2", "back": "Definition 2"}
            ]
        })
    
    elif tool_name == "generate_mindmap":
        return json.dumps({
            "title": topic,
            "root": {
                "label": topic,
                "children": [
                    {"label": "Concept 1", "children": []},
                    {"label": "Concept 2", "children": []}
                ]
            }
        })
    
    elif tool_name == "generate_study_plan":
        return json.dumps({
            "title": f"{topic} Study Plan",
            "totalDays": args["duration_days"],
            "dailyHours": args["daily_hours"],
            "tasks": [
                {"day": 1, "title": "Introduction", "description": f"Learn {topic} basics"}
            ]
        })
    
    elif tool_name == "summarize_notes":
        return f"Summary of {topic}: Key concepts include [main points from the topic]."
    
    return "Tool response"


async def generate_assistant_response(tool_name: str, topic: str, tool_response: str) -> str:
    """Generate natural assistant response incorporating tool result."""
    
    prompt = f"""You are a friendly study assistant. A tool just returned this result for a query about {topic}:

Tool: {tool_name}
Result: {tool_response[:200]}...

Write a natural, helpful response to the student that incorporates this tool result.
Be encouraging and conversational. Keep it concise (2-4 sentences).
"""
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()


async def generate_no_tool_example(topic: str) -> Dict[str, Any]:
    """Generate an example where no tool is needed (direct response)."""
    
    # General knowledge questions that don't need tools
    query_types = [
        f"What is {topic}?",
        f"Can you explain {topic}?",
        f"Tell me about {topic}",
        f"How does {topic} work?",
        f"Why is {topic} important?"
    ]
    
    user_query = random.choice(query_types)
    
    # Generate direct response
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        temperature=0.7,
        max_tokens=250
    )
    
    assistant_message = response.choices[0].message.content.strip()
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_message}
        ]
    }


async def generate_multi_tool_example(topic: str) -> Dict[str, Any]:
    """Generate an example that uses multiple tools in sequence."""
    
    # Common pattern: search first, then generate something
    user_query = f"Create a quiz on {topic} from my notes"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        # First tool: search
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_a",
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "arguments": json.dumps({"query": topic, "user_id": "user123", "notebook_id": "nb456"})
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "content": f"Found notes about {topic}: [key concepts and details]"
        },
        # Second tool: generate quiz
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_b",
                "type": "function",
                "function": {
                    "name": "generate_quiz",
                    "arguments": json.dumps({
                        "notebook_id": "nb456",
                        "user_id": "user123",
                        "num_questions": 10,
                        "difficulty": "medium",
                        "topic": f"{topic} from notes"
                    })
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "content": json.dumps({"quiz": {"title": f"{topic} Quiz", "questions": []}})
        },
        {
            "role": "assistant",
            "content": f"I've created a quiz based on your {topic} notes! Here are the questions..."
        }
    ]
    
    return {"messages": messages}


async def generate_training_dataset(
    examples_per_tool: int = 20,
    no_tool_examples: int = 10,
    multi_tool_examples: int = 15
) -> List[Dict[str, Any]]:
    """Generate complete training dataset."""
    
    print("🤖 Generating synthetic training data...")
    examples = []
    
    # Generate single-tool examples
    for tool_name in TOOLS.keys():
        print(f"  Generating {examples_per_tool} examples for {tool_name}...")
        for i in range(examples_per_tool):
            topic = random.choice(TOPICS)
            user_query = await generate_user_query_variant(tool_name, topic)
            example = await generate_tool_call_example(tool_name, topic, user_query)
            examples.append(example)
            
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{examples_per_tool}")
    
    # Generate no-tool examples
    print(f"  Generating {no_tool_examples} no-tool examples...")
    for i in range(no_tool_examples):
        topic = random.choice(TOPICS)
        example = await generate_no_tool_example(topic)
        examples.append(example)
    
    # Generate multi-tool examples
    print(f"  Generating {multi_tool_examples} multi-tool examples...")
    for i in range(multi_tool_examples):
        topic = random.choice(TOPICS)
        example = await generate_multi_tool_example(topic)
        examples.append(example)
    
    print(f"\n✅ Generated {len(examples)} total examples!")
    return examples


async def main():
    """Main generation script."""
    
    # Generate minimal dataset (18 single-tool + 3 no-tool + 3 multi-tool = 24 examples)
    # Reduced to respect API rate limits and minimize costs
    dataset = await generate_training_dataset(
        examples_per_tool=3,  # 3 per tool × 6 tools = 18
        no_tool_examples=3,
        multi_tool_examples=3
    )
    
    # Shuffle for better training
    random.shuffle(dataset)
    
    # Split train/val (90/10)
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Save training data
    train_path = "data/training/synthetic_training.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"💾 Saved {len(train_data)} training examples to {train_path}")
    
    # Save validation data
    val_path = "data/training/synthetic_validation.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for example in val_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"💾 Saved {len(val_data)} validation examples to {val_path}")
    
    # Combine with manual examples
    manual_path = "data/training/manual_examples.jsonl"
    combined_path = "data/training/collabry_training.jsonl"
    
    combined_data = []
    
    # Load manual examples
    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            for line in f:
                combined_data.append(json.loads(line))
        print(f"📖 Loaded {len(combined_data)} manual examples")
    
    # Add synthetic examples
    combined_data.extend(train_data)
    random.shuffle(combined_data)
    
    # Save combined dataset
    with open(combined_path, "w", encoding="utf-8") as f:
        for example in combined_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"🎉 Final training dataset: {len(combined_data)} examples saved to {combined_path}")
    print(f"📊 Validation dataset: {len(val_data)} examples")


if __name__ == "__main__":
    asyncio.run(main())
