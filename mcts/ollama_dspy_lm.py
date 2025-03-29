# ollama_dspy_lm.py

from dspy import LM
from ollama import Client

class OllamaLM(LM):
    def __init__(self, model="llama3", base_url="http://localhost:11434", **kwargs):
        self.client = Client(host=base_url)
        self.model = model
        self.kwargs = kwargs

    def __call__(self, prompt: str, **kwargs) -> str:
        response = self.client.generate(model=self.model, prompt=prompt, stream=False)
        return response["response"]

    def load(self):
        pass  # Optional: preload models

    def reset(self):
        pass  # Optional: clear state if needed
