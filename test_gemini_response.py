"""Test Gemini API response structure for gemini-2.5-flash"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Test 1: Basic text generation
print("=" * 60)
print("TEST 1: Basic Text Generation")
print("=" * 60)

config = types.GenerateContentConfig(
    temperature=0.7,
    max_output_tokens=100,
    response_mime_type="text/plain"
)

response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="Say hello",
    config=config
)

print(f"Response type: {type(response)}")
print(f"Response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
print(f"\nResponse value: {response}")

# Try to access .text
try:
    print(f"\nresponse.text: {response.text}")
except AttributeError as e:
    print(f"\n❌ AttributeError accessing .text: {e}")
    print("Trying to access response as string...")
    print(f"str(response): {str(response)}")

# Test 2: JSON mode
print("\n" + "=" * 60)
print("TEST 2: JSON Mode Generation")
print("=" * 60)

config_json = types.GenerateContentConfig(
    temperature=0.1,
    max_output_tokens=200,
    response_mime_type="application/json"
)

response_json = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents='Return JSON: {"greeting": "hello", "language": "english"}',
    config=config_json
)

print(f"Response type: {type(response_json)}")
print(f"\nResponse value: {response_json}")

try:
    print(f"\nresponse_json.text: {response_json.text}")
except AttributeError as e:
    print(f"\n❌ AttributeError accessing .text: {e}")
    print(f"str(response_json): {str(response_json)}")
