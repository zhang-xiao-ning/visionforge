# My Test Project

CIFAR-10 image classification with PyTorch.

A small, clean, extensible training framework that supports multiple
models (MLP / ConvNet / ViT), resumable training, reproducible
experiments, and a full local dev environment (lint + type check +
tests + pre-commit).

---

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (dependency manager)
- PyTorch 2.2
- (Optional) CUDA 12.1 for GPU training

---

## Installation

```bash
git clone https://gitee.com/zhang-xiao-ning/my_test_project.git
cd my_test_project
uv sync
```

Then install git hooks (runs ruff + mypy before every commit):

```bash
make install-hooks
```

---

## Dataset

CIFAR-10 is expected under `datasets/cifar-10-batches-py/`.

If you don't have it yet:

```bash
mkdir -p datasets
# Download cifar-10-python.tar.gz manually and extract here
# Expected structure:
# datasets/cifar-10-batches-py/data_batch_1 ...
```

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

### See all options

```bash
make help
uv run python src/main.py --help
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

### Environment variables

| Variable | Effect | Default |
|---|---|---|
| `USE_GPU` | `"true"` / `"false"` — force CPU or auto-detect GPU | `"true"` |

Example:

```bash
USE_GPU=false make train EXP=mlp EPOCHS=1   # force CPU
```

---

## Supported Experiments

| Name | Model | Default LR | Notes |
|---|---|---|---|
| `mlp` | Two-layer MLP | `1e-2` | Baseline |
| `shallow_convnet` | 2-layer ConvNet | `1e-2` | |
| `deep_convnet` | 5-layer ConvNet + BN | `0.1` | ~74% on CIFAR-10 |
| `vit` | Small ViT | `3e-4` | patch=4, dim=192, depth=6 |

---

## Outputs

Every run creates four files:

```text
outputs/<exp>_<timestamp>.log     # full log
outputs/<exp>_<timestamp>.csv     # epoch, train_loss, val_acc, lr
outputs/<exp>_<timestamp>.json    # config + environment snapshot
outputs/<exp>_<timestamp>/        # TensorBoard event files
checkpoints/<exp>_<timestamp>.pt  # model + optimizer + scheduler
```

View training curves:

```bash
make board     # starts TensorBoard at http://localhost:6006
```

---

## Project Structure

```text
my_test_project/
├── src/
│   ├── __init__.py              # public API + __version__
│   ├── py.typed                 # type marker for mypy
│   ├── config.py                # TrainConfig + device + global constants
│   ├── main.py                  # CLI entry, registers experiments
│   ├── data/
│   │   ├── transforms.py        # data augmentation
│   │   ├── cifar10.py           # CIFAR-10 specific config
│   │   └── datasets.py          # dataset registry + loader builder
│   ├── models/
│   │   ├── mlp.py
│   │   ├── shallow_convnet.py
│   │   ├── deep_convnet.py
│   │   └── vit.py
│   ├── training/
│   │   ├── train.py             # generic training loop
│   │   └── evaluator.py         # generic evaluate()
│   └── utils/
│       ├── path.py              # path constants
│       ├── logger.py            # logging + CSV recorder
│       ├── seed.py              # reproducibility
│       └── env.py               # environment snapshot
├── tests/
│   ├── conftest.py              # shared fixtures
│   ├── test_models.py
│   ├── test_transforms.py
│   ├── test_data.py
│   ├── test_evaluator.py
│   ├── test_trainer.py
│   └── test_logger.py
├── datasets/                    # data (not in git)
├── checkpoints/                 # weights (not in git)
├── outputs/                     # logs/CSV/JSON (not in git)
├── Makefile                     # project shortcuts
├── pyproject.toml               # dependencies + tool config
├── .pre-commit-config.yaml      # pre-commit hooks
├── .gitignore
├── README.md
├── TODO.md
├── COMPLETED.md
├── memo-1.md
├── memo-2.md
├── memo-3.md
├── pack.sh                      # zip the project
└── dump_code.sh                 # export all code as text
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
make train          # train (see above)
make board          # launch TensorBoard
make clean          # remove caches
```

### Pre-commit

Every `git commit` triggers:

1. `ruff check --fix` (lint + auto-fix)
2. `ruff format` (formatting)
3. `mypy` (type check)

If anything is auto-fixed, the commit is rejected. Re-run:

```bash
git add -A
git commit -m "..."
```

To skip hooks (rare):

```bash
git commit --no-verify -m "emergency"
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

2. Register it in `src/main.py`:

   ```python
   EXPERIMENTS: dict[str, tuple[Callable[[], nn.Module], float]] = {
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

No other files need to change.

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

---

## Testing

```bash
make test
# or
uv run pytest -v
```

Tests run on CPU regardless of the machine's GPU. This keeps them fast
and deterministic across macOS / Linux / CI.

---

## Reproducibility

- `set_seed(42)` fixes `random`, `numpy`, `torch`
- Every run writes a JSON snapshot with config + environment info
- CUDA determinism enabled

---

## License

MIT