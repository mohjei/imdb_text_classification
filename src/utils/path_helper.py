
import logging
from pathlib import Path


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[2].absolute()

DATA_DIR = ROOT_DIR / 'data'

MODEL_DIR = ROOT_DIR / 'saved_model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ASSET_DIR = ROOT_DIR / 'asset'
ASSET_DIR.mkdir(parents=True, exist_ok=True)
