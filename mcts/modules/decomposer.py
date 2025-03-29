import dspy

from mcts.signatures import DecomposeQuestion


class QuestionDecomposer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(DecomposeQuestion)

    def forward(self, question):
        return self.predict(question=question).sub_question
