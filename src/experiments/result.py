import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

from metrics.continual.base import ContinualMetricResultMatrix


@dataclass
class ExperimentResult:
    """
    Stores the results of a set of metrics over a IODDataset
    """
    name: str
    continual_results: Dict[str, ContinualMetricResultMatrix]
    final_results: Dict[str, List[Any]]

    def save(self, out_path: Path):
        with open(str(out_path), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, save_path: Path):
        with open(str(save_path), "rb") as f:
            return pickle.load(f)