"""
Ollama client for local LLM inference. No external APIs.
"""

import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaClient:
    """Simple sync client for Ollama /api/generate."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL

    def generate(self, prompt: str, system: str = None, max_tokens: int = 512) -> str:
        """
        Send prompt to Ollama and return the generated text.
        Optionally set a system message for behavior.
        """
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system:
            body["system"] = system
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read().decode())
                # Newer Ollama versions may return an "error" field instead of "response" on failure
                if "error" in out and out["error"]:
                    return f"[Ollama error from server: {out['error']}]"
                text = out.get("response", "")
                if not text:
                    return "[Ollama returned an empty response. Check model name and logs.]"
                return text.strip()
        except urllib.error.URLError as e:
            return f"[Ollama error: {e}. Is Ollama running at {self.base_url}?]"
        except Exception as e:
            return f"[Error talking to Ollama: {e}]"

    def generate_stream(self, prompt: str, system: str = None, max_tokens: int = 512):
        """
        Stream generated text chunks from Ollama /api/generate.
        Yields plain text pieces as they arrive.
        """
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        if system:
            body["system"] = system
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        out = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if out.get("error"):
                        yield f"[Ollama error from server: {out['error']}]"
                        return
                    chunk = out.get("response", "")
                    if chunk:
                        yield chunk
                    if out.get("done", False):
                        break
        except urllib.error.URLError as e:
            yield f"[Ollama error: {e}. Is Ollama running at {self.base_url}?]"
        except Exception as e:
            yield f"[Error talking to Ollama: {e}]"
