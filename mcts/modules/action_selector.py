import dspy

from mcts.signatures import MCTSNodeDecision


class ActionSelector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(MCTSNodeDecision)

    def forward(self, context, question):
        return self.predict(context=context, question=question).action
