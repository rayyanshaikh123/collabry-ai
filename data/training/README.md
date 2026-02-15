# Manual Training Examples for Collabry Fine-Tuning

This file contains handcrafted, high-quality training examples for fine-tuning the Collabry study assistant. These examples demonstrate perfect tool calling behavior.

## Format

Each example follows OpenAI's fine-tuning format:
```json
{"messages": [
  {"role": "system", "content": "System prompt..."},
  {"role": "user", "content": "User message"},
  {"role": "assistant", "content": null, "tool_calls": [...]},
  {"role": "tool", "tool_call_id": "...", "content": "Tool result"},
  {"role": "assistant", "content": "Final response to user"}
]}
```

## Examples by Tool Type

### search_sources (20 examples)
### generate_quiz (20 examples)
### generate_flashcards (15 examples)
### generate_mindmap (15 examples)
### generate_study_plan (15 examples)
### summarize_notes (10 examples)
### Multi-tool sequences (15 examples)
### No tool needed (10 examples)

Total: 120 examples

---

Note: The actual JSONL file will be generated from these examples using the convert script.
