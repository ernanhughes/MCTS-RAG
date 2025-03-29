import dspy

from mcts.modules.action_selector import ActionSelector
from mcts.modules.query_generator import QueryGenerator
from mcts.modules.decomposer import QuestionDecomposer
from mcts.modules.summarizer import Summarizer


class DSPyReasoningProgram(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.select_action = ActionSelector()
        self.decompose = QuestionDecomposer()
        self.query_gen = QueryGenerator()
        self.summarizer = Summarizer()
        self.retriever = retriever

    def forward(self, question, context):
        action = self.select_action.forward(context=context, question=question)
        if action == "A3":
            sub_q = self.decompose.forward(question)
            return self.forward(sub_q, context)
        elif action in ("A4", "A5"):
            query = self.query_gen.forward(question)
            docs = self.retriever.retrieve(query)
            return self.forward(question, docs[0]["text"] if docs else "")
        elif action == "A6":
            return self.summarizer.forward(context=context, question=question)
        return f"[Unimplemented Action: {action}]"
