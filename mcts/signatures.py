import dspy
from pydantic import Field


class MCTSNodeDecision(dspy.Signature):
    context: str = Field(..., json_schema_extra={"__dspy_field_type": "input"})
    question: str = Field(..., json_schema_extra={"__dspy_field_type": "input"})
    action: str = Field(..., json_schema_extra={"__dspy_field_type": "output"})

class DecomposeQuestion(dspy.Signature):
    question: str = Field(..., json_schema_extra={"__dspy_field_type": "input"})
    sub_question: str = Field(..., json_schema_extra={"__dspy_field_type": "output"})

class GenerateSearchQuery(dspy.Signature):
    question: str = Field(..., json_schema_extra={"__dspy_field_type": "input"})
    query: str = Field(..., json_schema_extra={"__dspy_field_type": "output"})

class SummarizeContextAnswer(dspy.Signature):
    context: str = Field(..., json_schema_extra={"__dspy_field_type": "input"})
    question: str = Field(..., json_schema_extra={"__dspy_field_type": "input"})
    answer: str = Field(..., json_schema_extra={"__dspy_field_type": "output"})
