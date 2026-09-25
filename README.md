# visionforge

![CI](https://github.com/zhang-xiao-ning/visionforge/actions/workflows/ci.yml/badge.svg)

CIFAR-10 image classification with PyTorch.

A small, clean, extensible training framework. Supports multiple models
(MLP / ConvNet / ViT), distributed training, ONNX export, a FastAPI
inference service, and a full local dev environment (lint + type check
+ tests + pre-commit + CI).

---

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (dependency manager)
- PyTorch 2.2
- (Optional) CUDA 12.1 for GPU training
- (Optional) Docker + nvidia-container-toolkit for containerized training

---

## Installation

```bash
git clone https://github.com/zhang-xiao-ning/visionforge.git
cd visionforge
uv sync
```

Install git hooks (runs ruff + mypy before every commit):

```bash
make install-hooks
```

---

## Dataset

CIFAR-10 is expected under `datasets/cifar-10-batches-py/`.

If you don't have it:

```bash
mkdir -p datasets
cd datasets
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
```

Expected structure:

```text
datasets/cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

Each `data_batch_*` should be ~30 MB.

---

## Training

Basic usage:

```bash
make train EXP=mlp EPOCHS=5
make train EXP=deep_convnet EPOCHS=10 LR=0.1
make train EXP=vit EPOCHS=10 BS=256 AMP=1
```

Or directly:

```bash
uv run python src/main.py --experiment vit --epochs 10 --batch-size 256 --amp
```

### Resume from checkpoint

```bash
make train EXP=vit EPOCHS=20 RESUME=checkpoints/vit_20260914_223852.pt
```

### DDP (distributed training)

```bash
# Single process (validation of DDP path)
make train-ddp EXP=mlp EPOCHS=5 NPROC=1

# Multi-GPU (when more GPUs are available)
make train-ddp EXP=vit EPOCHS=10 NPROC=2
```

### Docker training (GPU only)

```bash
docker compose --profile train build train           # one-time
make train-docker EXP=mlp EPOCHS=1
```
### Quick debug (small subset)

```bash
uv run python src/main.py --experiment mlp --epochs 1 --num-train 1000
```

### Full CLI argument list

| Argument | Description | Default |
|---|---|---|
| `--experiment` | Experiment name | `mlp` |
| `--dataset` | Dataset name | `cifar10` |
| `--batch-size` | Batch size | `64` |
| `--epochs` | Number of epochs | `1` |
| `--learning-rate` | Learning rate | per-experiment default |
| `--momentum` | SGD momentum | `0.9` |
| `--no-nesterov` | Disable Nesterov momentum | off |
| `--seed` | Random seed | `42` |
| `--lr-scheduler` | `none` / `step` / `cosine` | `none` |
| `--step-size` | StepLR step size | `10` |
| `--gamma` | StepLR decay factor | `0.1` |
| `--early-stop-patience` | Early stopping patience | `0` (disabled) |
| `--resume` | Resume from checkpoint path | none |
| `--amp` | Enable mixed precision (CUDA only) | off |
| `--num-train` | Number of training samples (default: all) | none |

### Environment variables

| Variable | Effect | Default |
|---|---|---|
| `USE_GPU` | `"true"` / `"false"` — force CPU or auto-detect GPU | `"true"` |

Example:

```bash
USE_GPU=false make train EXP=mlp EPOCHS=1   # force CPU
```

---

## Supported experiments

| Name | Model | Default LR | Notes |
|---|---|---|---|
| `mlp` | Two-layer MLP | `1e-2` | Baseline |
| `shallow_convnet` | 2-layer ConvNet | `1e-2` | |
| `deep_convnet` | 5-layer ConvNet + BN | `0.1` | ~74% on CIFAR-10 |
| `vit` | Small ViT | `3e-4` | patch=4, dim=192, depth=6 |

---

## Inference

### Export to ONNX

```bash
make export EXP=mlp
# exports/mlp.onnx
```

Verifies PyTorch vs ONNX output (max diff < 1e-5) as part of the export.

### FastAPI service

```bash
make serve                            # default ONNX=exports/mlp.onnx
make serve ONNX=exports/deep_convnet.onnx PORT=8001
```

Endpoints:

- `GET /health` → `{"status": "ok", "model": "mlp.onnx"}`
- `POST /predict` (multipart file upload) → top-k predictions

Interactive docs: <http://localhost:8000/docs>

### Docker serving image

```bash
docker compose up -d                  # start api service
docker compose logs -f api
```

---

## Outputs

Every run creates:

```text
outputs/<exp>_<timestamp>.log     # full log
outputs/<exp>_<timestamp>.csv     # epoch, train_loss, val_metric, lr
outputs/<exp>_<timestamp>.json    # config + environment snapshot
outputs/<exp>_<timestamp>/        # TensorBoard event files
checkpoints/<exp>_<timestamp>.pt  # model + optimizer + scheduler
```

View training curves:

```bash
make board     # TensorBoard at http://localhost:6006
```

---

## Project structure

```text
visionforge/
├── src/
│   ├── __init__.py              # public API + __version__
│   ├── py.typed                 # type marker for mypy
│   ├── config.py                # TrainConfig + device + global constants
│   ├── registry.py              # EXPERIMENTS registry
│   ├── main.py                  # CLI entry
│   ├── data/
│   │   ├── transforms.py
│   │   ├── cifar10.py
│   │   └── datasets.py          # dataset registry + loader builder
│   ├── models/
│   │   ├── mlp.py
│   │   ├── shallow_convnet.py
│   │   ├── deep_convnet.py
│   │   └── vit.py
│   ├── training/
│   │   ├── train.py             # generic training loop
│   │   ├── evaluator.py         # generic evaluate()
│   │   └── strategy.py          # SingleDevice / DDP
│   ├── experiment/
│   │   ├── artifacts.py         # RunArtifacts
│   │   └── runner.py            # ExperimentRunner
│   ├── export/
│   │   └── onnx_export.py
│   ├── serving/
│   │   ├── api.py               # FastAPI routes
│   │   ├── inference.py         # ONNX Runtime
│   │   └── schema.py            # Pydantic schemas
│   └── utils/
│       ├── path.py
│       ├── logger.py
│       ├── seed.py
│       └── env.py
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_transforms.py
│   ├── test_data.py
│   ├── test_evaluator.py
│   ├── test_trainer.py
│   ├── test_logger.py
│   ├── test_artifacts.py
│   ├── test_strategy.py
│   └── test_runner.py
├── docker/
│   ├── Dockerfile.serve
│   └── Dockerfile.train
├── scripts/
│   ├── ci.sh
│   └── train_ddp.sh
├── datasets/                    # data (not in git)
├── checkpoints/                 # weights (not in git)
├── outputs/                     # logs/CSV/JSON (not in git)
├── exports/                     # ONNX models (not in git)
├── docs/
│   ├── TODO.md
│   ├── COMPLETED.md
│   └── memo-1.md ~ memo-9.md
├── docker-compose.yml
├── .dockerignore
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
└──  README.md
```

---

## Development

### Common commands

```bash
make help           # list all targets
make install        # uv sync
make install-hooks  # install git pre-commit hooks
make test           # pytest
make lint           # ruff check + ruff format --check + mypy
make format         # ruff check --fix + ruff format
make ci             # full CI pipeline locally
make train          # see "Training"
make train-ddp      # DDP training
make train-docker   # containerized training (GPU)
make board          # TensorBoard
make export         # ONNX export
make serve          # FastAPI service
make clean          # remove caches
```

### Pre-commit

Every `git commit` triggers:

1. `ruff check --fix` (lint + auto-fix)
2. `ruff format`
3. `mypy`

If anything is auto-fixed, the commit is rejected. Re-run:

```bash
git add -A
git commit -m "..."
```

### Adding a new model

1. Create `src/models/<name>.py` with a class inheriting `nn.Module`:

```python
   import torch
   import torch.nn as nn


   class MyModel(nn.Module):
       def __init__(self, num_classes: int = 10) -> None:
           super().__init__()
           ...

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           ...
```

2. Register it in `src/registry.py`:

```python
   EXPERIMENTS = {
       ...
       "my_model": (MyModel, 1e-3),
   }
```

3. Add a shape test in `tests/test_models.py`:

```python
   def test_my_model_output_shape(dummy_batch):
       x, _ = dummy_batch
       model = MyModel()
       out = model(x)
       assert out.shape == (4, 10)
```

4. Run:

```bash
   make test
   make train EXP=my_model EPOCHS=1
  ```

### Adding a new dataset

1. Create `src/data/<name>.py` with `build_datasets()` returning
   `(train_set, val_set, test_set)`.
2. Register it in `src/data/datasets.py`:

```python
   DATASET_REGISTRY = {
       "cifar10": cifar10.build_datasets,
       "my_dataset": my_dataset.build_datasets,
   }
```

3. Run `make train EXP=mlp --dataset=my_dataset`.

### Adding a new training strategy (FSDP, DeepSpeed, ...)

Subclass `TrainingStrategy` in `src/training/strategy.py` and add a
branch to `build_strategy()`. No other file needs to change.

---

## Testing

```bash
make test
# or
uv run pytest -v
```

Tests run on CPU regardless of the machine's GPU (via `USE_GPU=false`
in `tests/conftest.py`). This keeps them fast and deterministic across
macOS / Linux / CI.

---

## Reproducibility

- `set_seed(42)` fixes `random`, `numpy`, `torch`
- Every run writes a JSON snapshot with config + environment info
- CUDA determinism enabled
- Under DDP, `DistributedSampler` reshuffles per epoch via `set_epoch`

---

## License

MIT