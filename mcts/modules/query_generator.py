import dspy

from mcts.signatures import GenerateSearchQuery


class QueryGeneratorModule(Student):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenerateSearchQuery)

    def forward(self, question):
        return self.predict(question=question).query
