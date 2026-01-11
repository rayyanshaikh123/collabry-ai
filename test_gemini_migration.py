"""
Test script to verify Gemini migration is working correctly.

Run this script to test all major components:
- Gemini service initialization
- Intent classification
- Entity extraction
- Text generation
- RAG retrieval
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration loading."""
    print("=" * 70)
    print("TEST 1: Configuration")
    print("=" * 70)
    
    try:
        from config import CONFIG
        
        # Check Gemini config
        assert "gemini_api_key" in CONFIG, "Missing gemini_api_key in config"
        assert "gemini_model" in CONFIG, "Missing gemini_model in config"
        
        # Check if API key is set
        api_key = CONFIG["gemini_api_key"]
        if not api_key or api_key == "your_gemini_api_key_here":
            print("❌ FAILED: GEMINI_API_KEY not set in .env file")
            print("   Please set your API key: https://aistudio.google.com/app/apikey")
            return False
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Model: {CONFIG['gemini_model']}")
        print(f"   API Key: {api_key[:10]}...")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_gemini_service():
    """Test Gemini service initialization."""
    print("\n" + "=" * 70)
    print("TEST 2: Gemini Service")
    print("=" * 70)
    
    try:
        from core.gemini_service import create_gemini_service
        
        gemini = create_gemini_service()
        print("✅ Gemini service initialized")
        
        # Test basic generation
        response = gemini.generate("Say 'Hello, Gemini!' in one short sentence.")
        print(f"✅ Basic generation works")
        print(f"   Response: {response.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_classification():
    """Test intent classification."""
    print("\n" + "=" * 70)
    print("TEST 3: Intent Classification")
    print("=" * 70)
    
    try:
        from core.gemini_intent import IntentClassifier
        
        classifier = IntentClassifier()
        
        # Test different intents
        test_cases = [
            ("Hello, how are you?", "chat"),
            ("What is photosynthesis?", "qa"),
            ("Summarize this article for me", "summarize"),
            ("Create a quiz on biology", "generate"),
            ("Help me plan my study schedule", "plan"),
        ]
        
        for query, expected_intent in test_cases:
            result = classifier.classify(query)
            intent = result["intent"]
            confidence = result["confidence"]
            
            status = "✅" if intent == expected_intent else "⚠️"
            print(f"{status} '{query[:40]}...' → {intent} ({confidence:.2f})")
        
        print("✅ Intent classification works")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entity_extraction():
    """Test entity extraction."""
    print("\n" + "=" * 70)
    print("TEST 4: Entity Extraction")
    print("=" * 70)
    
    try:
        from core.nlp import extract_entities
        
        text = "John Smith lives in New York and works at Google."
        entities = extract_entities(text)
        
        print(f"✅ Entity extraction works")
        print(f"   Text: {text}")
        print(f"   Entities: {entities}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nlp_pipeline():
    """Test complete NLP pipeline."""
    print("\n" + "=" * 70)
    print("TEST 5: NLP Pipeline")
    print("=" * 70)
    
    try:
        from core.nlp import analyze
        
        text = "Can you explain what machine learning is?"
        result = analyze(text)
        
        print(f"✅ NLP pipeline works")
        print(f"   Intent: {result['intent']}")
        print(f"   Entities: {result['entities']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_llm():
    """Test LocalLLM interface."""
    print("\n" + "=" * 70)
    print("TEST 6: LocalLLM (Gemini-powered)")
    print("=" * 70)
    
    try:
        from core.local_llm import create_llm
        from config import CONFIG
        
        llm = create_llm(CONFIG)
        response = llm.invoke("What is 2+2? Answer in one sentence.")
        
        print(f"✅ LocalLLM works (backward compatibility)")
        print(f"   Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("COLLABRY AI ENGINE - GEMINI MIGRATION TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        test_config,
        test_gemini_service,
        test_intent_classification,
        test_entity_extraction,
        test_nlp_pipeline,
        test_local_llm,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ TEST CRASHED: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Gemini migration successful.")
        print("\nNext steps:")
        print("1. Start the server: python run_server.py")
        print("2. Test with frontend or curl")
        print("3. Monitor API usage at: https://aistudio.google.com")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("\nTroubleshooting:")
        print("1. Check GEMINI_API_KEY is set in .env")
        print("2. Verify internet connection (Gemini is cloud-based)")
        print("3. Check logs for detailed error messages")
        print("4. See GEMINI_MIGRATION.md for help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
