# scripts/train_query_generator.py

import dspy
from mcts_rag.modules.query_generator import QueryGeneratorModule
from data.query_training_examples import training_set

# Load LM (or use a local one via Ollama or LMStudio)
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4"))

# Format dataset
examples = [
    dspy.Example(question=ex["question"], query=ex["query"])
    for ex in training_set
]

# Create and train
module = QueryGeneratorModule()
teleprompter = dspy.Teleprompter()
teleprompter.compile(module, trainset=examples, metric=dspy.exact_match)

# Save trained module
module.save("trained_query_module.json")

