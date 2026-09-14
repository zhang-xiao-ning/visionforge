from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_PATH = PROJECT_ROOT / "datasets"
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

CHECKPOINTS_PATH.mkdir(exist_ok=True)
OUTPUTS_PATH.mkdir(exist_ok=True)