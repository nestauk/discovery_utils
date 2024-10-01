import logging
import os

from pathlib import Path

import dotenv


dotenv.load_dotenv()

# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Define S3 bucket
S3_BUCKET = os.getenv("S3_BUCKET")

# configger logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
