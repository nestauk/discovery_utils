from discovery_utils.getters import hansard
from datetime import datetime, timedelta
import pandas as pd

Hansard = hansard.HansardGetter()
debates_df = Hansard.get_debates_parquet()
labelstore_df = Hansard.get_labelstore()
debates_df.date.max()

people_dict = Hansard.get_people_metadata()

import importlib
from src import synthesis_utils
importlib.reload(synthesis_utils);
import numpy as np
from typing import Literal, Tuple, List, Dict

from src import logging
from discovery_utils.utils import keywords


