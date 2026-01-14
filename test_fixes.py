"""Quick test for Hugging Face fixes"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("TESTING HUGGING FACE FIXES")
print("=" * 70)

# Test 1: LocalLLM generation (should not crash on .text)
print("\n[1] Testing LocalLLM generation...")
try:
    from config import CONFIG
    from core.local_llm import LocalLLM, create_llm
    llm = create_llm(CONFIG)
    response = llm._call("Say hello in 3 words")
    print(f"✅ Success: {response}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Intent classification (should handle dict response)
print("\n[2] Testing Intent Classification...")
try:
    from core.nlp import analyze
    result = analyze("Can you help me study for my math test?")
    print(f"✅ Success:")
    print(f"   Intent: {result['intent']}")
    print(f"   Confidence: {result['intent_proba']}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: Entity extraction
print("\n[3] Testing Entity Extraction...")
try:
    from core.nlp import extract_entities
    entities = extract_entities("I need to study Python and Machine Learning for my exam next week")
    print(f"✅ Success: {entities}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
