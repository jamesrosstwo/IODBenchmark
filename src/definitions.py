import os
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = Path(SRC_DIR)
ROOT_PATH = SRC_PATH.parent.resolve()
DATA_PATH = ROOT_PATH / "data"
DATA_PATH.mkdir(exist_ok=True)
OUT_PATH = ROOT_PATH / "out"
OUT_PATH.mkdir(exist_ok=True)