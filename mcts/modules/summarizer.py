import dspy

from mcts.signatures import SummarizeContextAnswer


class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SummarizeContextAnswer)

    def forward(self, context, question):
        return self.predict(context=context, question=question).answer
