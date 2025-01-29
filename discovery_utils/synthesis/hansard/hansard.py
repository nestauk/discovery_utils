from datetime import datetime
from datetime import timedelta

import pandas as pd

from discovery_utils.getters import hansard


Hansard = hansard.HansardGetter()
debates_df = Hansard.get_debates_parquet()
labelstore_df = Hansard.get_labelstore()
debates_df.date.max()

people_dict = Hansard.get_people_metadata()

import importlib

from src import synthesis_utils


importlib.reload(synthesis_utils)
from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple

import numpy as np

from src import logging

from discovery_utils.utils import keywords
