"""
diagnose_llm.py - Quick LLM diagnostic script

Run this to check if Ollama is working:
    python diagnose_llm.py
"""

import urllib.request
import urllib.error
import json
import sys

OLLAMA_URL = "http://localhost:11434"


def check_ollama():
    """Check if Ollama server is responding."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                models = body.get("models", [])
                return True, [m.get("name", "") for m in models]
    except urllib.error.URLError as e:
        return False, [f"Connection refused: {e}"]
    except Exception as e:
        return False, [str(e)]
    return False, ["Unknown error"]


def test_model(model_name: str):
    """Test if a specific model responds to a simple query."""
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say 'OK' and nothing else"}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"  Testing {model_name}... (this may take 30-60s on first run)")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            return True, content[:50]
    except urllib.error.URLError as e:
        return False, f"Connection error: {e}"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("=" * 60)
    print("LLM DIAGNOSTIC")
    print("=" * 60)

    print("\n1. Checking Ollama server...")
    ok, models = check_ollama()

    if not ok:
        print(f"   [FAIL] {models[0]}")
        print("\n   FIX: Start Ollama server:")
        print("   $ ollama serve")
        sys.exit(1)

    print(f"   [OK] Ollama is running")
    print(f"   Available models: {', '.join(models) if models else 'None'}")

    if not models:
        print("\n   [WARN] No models found!")
        print("   FIX: Pull a model:")
        print("   $ ollama pull phi4:latest")
        sys.exit(1)

    # Test each model
    print("\n2. Testing models...")
    for model in models:
        ok, result = test_model(model)
        if ok:
            print(f"   [OK] {model}: Responded in <120s")
            print(f"      Response preview: '{result[:60]}...'")
        else:
            print(f"   [FAIL] {model}: {result}")
            print(f"      This model may need to be downloaded or loaded into memory")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

    print("\nIf models are timing out:")
    print("  1. First run downloads model (3-8GB) — check internet")
    print("  2. First run loads model into RAM — takes 30-120s")
    print("  3. Subsequent runs are faster (model stays in memory)")
    print("\nTo use manual Phase 1 (no LLM):")
    print("  Check '⚡ Use Manual Phase 1' in the Streamlit app")
