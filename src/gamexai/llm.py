"""Backend-agnostic LLM client: Ollama (local, free) or OpenAI, with graceful
degradation to a deterministic template when no LLM is reachable.

The template path matters for two reasons: (1) the live conference demo never
breaks even with no network/GPU, and (2) it serves as the *faithful baseline*
in the faithfulness experiment.
"""
import json
import os
import urllib.error
import urllib.request

from . import config as C


class LLMClient:
    def __init__(self, backend="auto", model=None, timeout=90):
        self.timeout = timeout
        self.backend = backend if backend != "auto" else self._autodetect()
        self.model = model or (C.OPENAI_MODEL if self.backend == "openai"
                               else C.OLLAMA_MODEL)

    # ---- detection ------------------------------------------------------
    @staticmethod
    def _ollama_up():
        try:
            urllib.request.urlopen(C.OLLAMA_HOST + "/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def _autodetect(self):
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if self._ollama_up():
            return "ollama"
        return "template"

    @property
    def available(self):
        return self.backend in ("ollama", "openai")

    def describe(self):
        return f"{self.backend}:{self.model}" if self.available else "template (no LLM)"

    # ---- generation -----------------------------------------------------
    def chat(self, system, user, temperature=0.2):
        if self.backend == "ollama":
            return self._ollama(system, user, temperature)
        if self.backend == "openai":
            return self._openai(system, user, temperature)
        raise RuntimeError("No LLM backend available; use a template instead.")

    def _ollama(self, system, user, temperature):
        body = json.dumps({
            "model": self.model, "stream": False,
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(C.OLLAMA_HOST + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()

    def _openai(self, system, user, temperature):
        body = json.dumps({
            "model": self.model, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
