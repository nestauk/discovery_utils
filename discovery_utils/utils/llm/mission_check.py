from pydantic import Field
from pydantic import BaseModel

from importlib.resources import files

from discovery_utils.utils.io import safe_yaml_load
from discovery_utils.utils.llm.llm_utils import StructuredOutputGenerator

CONFIG_PATH = files("discovery_utils.utils.llm").joinpath("prompts_mission_check.yaml")
CONFIG = safe_yaml_load(open(CONFIG_PATH).read())

class RelevanceClassifier(BaseModel):
    relevant: bool = Field(description="Whether the text is highly relevant to the defined topic.")

Generator = StructuredOutputGenerator(
    model_dict = CONFIG,
    output_class = RelevanceClassifier,
    prompts = CONFIG["relevance_classifier"],
)

def classify_relevance(input: str, mission: str) -> RelevanceClassifier:
    """Classify the relevance of a text to a given mission.
    
    Uses mission scope definitions in the configuration file
    Args:
        input: The input text to classify.
        mission: The mission to classify the text against.

    Returns:
        RelevanceClassifier: The classification result.
    """
    mission_info = CONFIG["mission_info"][mission]
    return Generator.generate({"input": input, "policy_area": mission_info})