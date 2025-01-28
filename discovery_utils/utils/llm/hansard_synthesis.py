"""Utils for synthesising Hansard debates"""

from pydantic import Field
from pydantic import BaseModel

from typing import List
from typing import Union

from importlib.resources import files

from discovery_utils.utils.io import safe_yaml_load
from discovery_utils.utils.llm.llm_utils import StructuredOutputGenerator

CONFIG_PATH = files("discovery_utils.utils.llm").joinpath("prompts_hansard_synthesis.yaml")
CONFIG = safe_yaml_load(open(CONFIG_PATH).read())

class Debate(BaseModel):
    heading: str
    content: str


class StructuredSummaryOutput(BaseModel):
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
    summary: str = Field(description="The summary.")

SummaryGenerator = StructuredOutputGenerator(
    model_dict = CONFIG,
    output_class = StructuredSummaryOutput,
    prompts = CONFIG["debate_summary"],
)

def summarise_debate_with_structure(debate: Debate) -> StructuredSummaryOutput:
    """"""
    return SummaryGenerator.generate(
        input=debate.content,
        output_class=StructuredSummaryOutput,
        messages_config=CONFIG["debate_summary"],
    )


QuoteGenerator = StructuredOutputGenerator(
    model_dict = CONFIG,
    output_class = QuoteSummaryOutput,
    prompts = CONFIG["quote_summary"],
) 

def summarise_quote(text: str) -> QuoteSummaryOutput:
    """"""
       
    return QuoteGenerator.generate(
        input=text,
        output_class=QuoteSummaryOutput,
        messages_config=CONFIG["quote_summary"],
    )
