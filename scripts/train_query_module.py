# scripts/train_query_module.py

import dspy
import json
from mcts.modules.query_generator import QueryGeneratorModule
from mcts.ollama_dspy_lm import OllamaLM
from dspy.teleprompt import LabeledFewShot

# Configure DSPy with Ollama model
dspy.settings.configure(lm=OllamaLM(model="hf.co/ernanhughes/Fin-R1-Q8_0-GGUF:latest"))

# Load dataset
with open("../data/query_train.json") as f:
    data = json.load(f)

trainset = [dspy.Example(question=ex["question"], query=ex["query"]) for ex in data]

# Initialize module + trainer
query_module = QueryGeneratorModule()
tele = LabeledFewShot()

# ✅ Updated compile call (keyword arguments)
tele.compile(student=query_module, trainset=trainset)

# Save the trained module
query_module.save("trained_query_module.json")
