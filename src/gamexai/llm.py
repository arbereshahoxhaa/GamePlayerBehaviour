"""Backend-agnostic LLM client with a fallback chain.

Order of preference: OpenAI (if OPENAI_API_KEY is set) -> Ollama (if reachable)
-> deterministic template. Each `chat` call walks the chain and uses the first
backend that succeeds, so an OpenAI hiccup mid-run silently drops to Ollama, and
a fully offline machine still works via the template.

Set the key in a gitignored `.env` (see .env.example) or the environment.
"""
import json
import os
import urllib.request

from . import config as C


class LLMClient:
    def __init__(self, prefer="auto", model=None, timeout=90):
        self.timeout = timeout
        self.openai_model = os.getenv("OPENAI_MODEL", C.OPENAI_MODEL)
        self.ollama_model = model or os.getenv("OLLAMA_MODEL", C.OLLAMA_MODEL)
        self.chain = self._build_chain(prefer)

    # ---- backend discovery ---------------------------------------------
    @staticmethod
    def _ollama_up():
        try:
            urllib.request.urlopen(C.OLLAMA_HOST + "/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def _build_chain(self, prefer):
        chain = []
        if os.getenv("OPENAI_API_KEY"):
            chain.append("openai")
        if self._ollama_up():
            chain.append("ollama")
        if prefer in ("openai", "ollama") and prefer in chain:   # move to front
            chain = [prefer] + [b for b in chain if b != prefer]
        return chain

    @property
    def available(self):
        return bool(self.chain)

    @property
    def primary(self):
        return self.chain[0] if self.chain else "template"

    def describe(self):
        if not self.chain:
            return "template (no LLM)"
        tag = {"openai": self.openai_model, "ollama": self.ollama_model}
        return " -> ".join(f"{b}:{tag[b]}" for b in self.chain) + " -> template"

    # ---- generation -----------------------------------------------------
    def chat(self, system, user, temperature=0.2):
        errors = []
        for backend in self.chain:
            try:
                return self._call(backend, system, user, temperature)
            except Exception as e:                      # fall through to next
                errors.append(f"{backend}: {e}")
        raise RuntimeError("all LLM backends failed: " + "; ".join(errors))

    def _call(self, backend, system, user, temperature):
        return (self._openai if backend == "openai" else self._ollama)(
            system, user, temperature)

    def _ollama(self, system, user, temperature):
        body = json.dumps({
            "model": self.ollama_model, "stream": False,
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
            "model": self.openai_model, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
