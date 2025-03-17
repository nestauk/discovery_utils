"""Utils for synthesising Hansard debates"""

from importlib.resources import files
from typing import List

from pydantic import BaseModel
from pydantic import Field

from discovery_utils.utils.io import safe_yaml_load
from discovery_utils.utils.llm.llm_utils import StructuredOutputGenerator


CONFIG_PATH = files("discovery_utils.synthesis.policy").joinpath("prompts_hansard_synthesis.yaml")
CONFIG = safe_yaml_load(str(CONFIG_PATH))


class Debate(BaseModel):
    """Debate content"""

    heading: str
    content: str


class StructuredSummaryOutput(BaseModel):
    """Structured output for summarising a debate"""

    purpose: str = Field(description="The main theme or purpose of the debate.")
    positives: List[str] = Field(
        description="A list of key positive aspects or arguments raised. (indicate who proposed and their party)"
    )
    negatives: List[str] = Field(
        description="A list of key criticisms or issues discussed. (indicate who proposed and their party)"
    )
    next_steps: List[str] = Field(
        description="A list of proposed follow-ups or action points (indicate who proposed and their party)."
    )


class QuoteSummaryOutput(BaseModel):
    """Structured output for summarising a quote"""

    summary: str = Field(description="The summary.")


SummaryGenerator = StructuredOutputGenerator(
    model_dict=CONFIG,
    output_class=StructuredSummaryOutput,
    prompts=CONFIG["debate_summary"],
)


def summarise_debate_with_structure(debate: Debate) -> StructuredSummaryOutput:
    """Summarise a debate with structured output"""
    return SummaryGenerator.generate({"input": debate.content})


QuoteGenerator = StructuredOutputGenerator(
    model_dict=CONFIG,
    output_class=QuoteSummaryOutput,
    prompts=CONFIG["quote_summary"],
)


def summarise_quote(text: str) -> QuoteSummaryOutput:
    """Summarise a quote with structured output"""

    return QuoteGenerator.generate({"input": text})
