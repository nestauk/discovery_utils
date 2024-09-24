import logging
import os

from pathlib import Path

import dotenv


dotenv.load_dotenv()

# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Define S3 bucket
S3_BUCKET = os.getenv("BUCKET_NAME_RAW")

# configger logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
