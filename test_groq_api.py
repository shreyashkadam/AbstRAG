"""Test script to verify Groq API connection"""
import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in environment")
    sys.exit(1)

print(f"Testing Groq API connection...")
print(f"API Key present: {'Yes' if api_key else 'No'} (length: {len(api_key) if api_key else 0})")

try:
    client = Groq(api_key=api_key, timeout=30)
    print("✓ Groq client created successfully")
    
    # Test non-streaming first
    print("\nTesting non-streaming request...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
        stream=False,
    )
    print(f"✓ Non-streaming response received: {response.choices[0].message.content}")
    
    # Test streaming
    print("\nTesting streaming request...")
    stream_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
        stream=True,
    )
    
    chunks = []
    for chunk in stream_response:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
            print(f"  Received chunk: {chunk.choices[0].delta.content[:50]}...")
    
    full_response = "".join(chunks)
    print(f"✓ Streaming response received: {full_response}")
    print("\n✅ All tests passed! Groq API is working correctly.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

