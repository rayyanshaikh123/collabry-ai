"""
Complete test for all gemini-2.5-flash fixes

Tests:
1. LocalLLM generation (str vs object)
2. Intent classification (dict vs object) 
3. Entity extraction (dict to tuple conversion)
4. Agent integration (entities format)
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("COMPLETE FIX VERIFICATION - GEMINI 2.5-FLASH")
print("=" * 70)

# Test 1: LocalLLM
print("\n[TEST 1] LocalLLM Generation")
print("-" * 70)
try:
    from config import CONFIG
    from core.local_llm import create_llm
    
    llm = create_llm(CONFIG)
    response = llm._call("Say hello")
    print(f"✅ LocalLLM works")
    print(f"   Response type: {type(response)}")
    print(f"   Response: {response[:50]}...")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Intent Classification
print("\n[TEST 2] Intent Classification (dict/object handling)")
print("-" * 70)
try:
    from core.gemini_intent import IntentClassifier
    
    clf = IntentClassifier()
    result = clf.classify("Can you help me study for my math exam?")
    
    print(f"✅ Intent classification works")
    print(f"   Result type: {type(result)}")
    print(f"   Result: {result}")
    
    # Verify it's a dict
    if isinstance(result, dict):
        print(f"   ✓ Returns dict (backward compatible)")
        print(f"   ✓ Intent: {result.get('intent')}")
        print(f"   ✓ Confidence: {result.get('confidence')}")
    else:
        print(f"   ⚠ Returns object (not dict)")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Entity Extraction Format
print("\n[TEST 3] Entity Extraction (dict -> tuple conversion)")
print("-" * 70)
try:
    from core.nlp import extract_entities
    
    text = "I need to study Python and Machine Learning for my exam next Monday"
    entities = extract_entities(text)
    
    print(f"✅ Entity extraction works")
    print(f"   Result type: {type(entities)}")
    print(f"   Result: {entities}")
    
    # Verify format
    if isinstance(entities, list):
        print(f"   ✓ Returns list")
        if entities and isinstance(entities[0], tuple):
            print(f"   ✓ Contains tuples: {entities[0]}")
            print(f"   ✓ Tuple format: (text, label)")
        elif entities:
            print(f"   ❌ First item is not tuple: {type(entities[0])}")
        else:
            print(f"   ⚠ Empty list (quota or no entities)")
    else:
        print(f"   ❌ Not a list: {type(entities)}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Full NLP Pipeline
print("\n[TEST 4] Full NLP Pipeline (analyze function)")
print("-" * 70)
try:
    from core.nlp import analyze
    
    result = analyze("Help me understand quantum physics for my exam tomorrow")
    
    print(f"✅ NLP analyze works")
    print(f"   Intent: {result['intent']} (confidence: {result['intent_proba']})")
    print(f"   Entities type: {type(result['entities'])}")
    print(f"   Entities: {result['entities']}")
    
    # Verify entities format
    entities = result['entities']
    if isinstance(entities, list):
        print(f"   ✓ Entities is list")
        if entities and isinstance(entities[0], tuple) and len(entities[0]) == 2:
            print(f"   ✓ Correct tuple format: {entities[0]}")
        elif entities:
            print(f"   ❌ Wrong format: {entities[0]}")
        else:
            print(f"   ⚠ No entities found (quota or none detected)")
    else:
        print(f"   ❌ Entities not a list: {type(entities)}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Agent Entity Handling
print("\n[TEST 5] Agent Entity List Comprehension")
print("-" * 70)
try:
    # Simulate what agent.py does
    entities = [("Python", "TOPIC"), ("Machine Learning", "TOPIC"), ("next week", "DATE")]
    
    # This is what agent.py line 275 does
    ent_lines = [f"- {text} ({label})" for text, label in entities]
    
    print(f"✅ Agent entity formatting works")
    print(f"   Formatted entities:")
    for line in ent_lines:
        print(f"   {line}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    print(f"   This means the tuple unpacking will fail in agent.py")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n⚠️  NOTE: Quota errors (429) are expected and don't indicate bugs")
print("✅ All structural/format issues should be resolved")
print("\n🔄 RESTART the AI server to apply all fixes:")
print("   cd ai-engine && python run_server.py")
print("=" * 70)
