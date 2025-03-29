import dspy
from mcts.modules.action_selector import ActionSelector

def test_action_selector_returns_action():
    dspy.settings.configure(lm=dspy.MockLM(logprobs=True))  # fast tests
    module = ActionSelector()
    action = module.forward("Some context", "What is the capital of Mali?")
    assert isinstance(action, str)
