"""Utils for synthesising GtR research abstracts"""

from importlib.resources import files

from pydantic import BaseModel
from pydantic import Field

from discovery_utils.utils.io import safe_yaml_load
from discovery_utils.utils.llm.llm_utils import StructuredOutputGenerator


CONFIG_PATH = files("discovery_utils.utils.llm").joinpath("prompts_gtr_synthesis.yaml")
CONFIG = safe_yaml_load(str(CONFIG_PATH))


class AbstractSummaryOutput(BaseModel):
    title: str = Field(description="The condensed title of the research abstract.")
    summary: str = Field(description="The summary of the research abstract.")


AbstractGenerator = StructuredOutputGenerator(
    model_dict=CONFIG,
    output_class=AbstractSummaryOutput,
    prompts=CONFIG["abstract_summary"],
)


def summarise_abstract(title: str, text: str) -> AbstractSummaryOutput:
    """Summarise a research abstract."""

    return AbstractGenerator.generate(
        input_dict={"input": text, "title": title},
    )
