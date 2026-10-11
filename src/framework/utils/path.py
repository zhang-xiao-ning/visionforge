from pathlib import Path

# src/framework/utils/path.py → src/framework/utils → src/framework → src → project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_PATH = PROJECT_ROOT / "datasets"
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
