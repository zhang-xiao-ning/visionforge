# visionforge

![CI](https://github.com/zhang-xiao-ning/visionforge/actions/workflows/ci.yml/badge.svg)

Image classification and image captioning with PyTorch.

A small, clean, extensible training framework. Supports multiple tasks
(classification on CIFAR-10, captioning on Flickr8k), distributed
training, ONNX export, a FastAPI inference service, and a full local
dev environment (lint + type check + tests + pre-commit + CI).

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

### CIFAR-10 (for classification)

Expected under `datasets/cifar-10-batches-py/`.

```bash
bash scripts/download_datasets.sh cifar10
```

### Flickr8k (for captioning)

```bash
bash scripts/download_datasets.sh flickr8k
```

Expected structure:

```text
datasets/
├── cifar-10-batches-py/
│   ├── data_batch_1
│   ├── ...
│   ├── test_batch
│   └── batches.meta
├── Flicker8k_Dataset/          # 8091 images (note the misspelling)
└── Flickr8k_text/
    ├── Flickr8k.token.txt
    ├── Flickr_8k.trainImages.txt
    ├── Flickr_8k.devImages.txt
    └── Flickr_8k.testImages.txt
```

Each `data_batch_*` should be ~30 MB.

---

## Training

Basic usage:

```bash
make train EXP=vit EPOCHS=5
make train EXP=captioning EPOCHS=5

# Or directly
uv run python src/main.py --experiment vit --epochs 5
uv run python src/main.py --experiment captioning --epochs 5 --batch-size 32
```

### Common training options

| Goal | Command |
|---|---|
| AdamW + weight decay | `make train EXP=captioning OPT=adamw WD=0.01` |
| Warmup + cosine | `make train EXP=captioning OPT=adamw WARMUP=100` |
| Large effective batch | `make train EXP=captioning ACCUM=4` |
| Stability | `make train EXP=captioning CLIP=1.0` |
| Mixed precision (CUDA) | `make train EXP=captioning AMP=1` |
| Quick debug | `make train EXP=vit NUM=1000` |

Or directly:

```bash
uv run python src/main.py --experiment captioning --epochs 5 \
    --optimizer adamw --lr-scheduler cosine --warmup-steps 100 \
    --accum-steps 4 --grad-clip 1.0
```

### Resume from checkpoint

```bash
make train EXP=vit EPOCHS=20 RESUME=checkpoints/vit_20260914_223852.pt
```

### DDP (distributed training)

```bash
# Single process (validates the DDP path)
make train-ddp EXP=vit EPOCHS=5 NPROC=1

# Multi-GPU (when more GPUs are available)
make train-ddp EXP=vit EPOCHS=10 NPROC=2
```

### Docker training (GPU only)

```bash
docker compose --profile train build train           # one-time
make train-docker EXP=vit EPOCHS=1
```

### Full CLI argument list

| Argument | Description | Default |
|---|---|---|
| `--experiment` | Experiment name (`vit` / `captioning`) | `vit` |
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
| `--amp` | Mixed precision (CUDA only) | off |
| `--num-train` | Limit training samples | all |
| `--optimizer` | `sgd` / `adamw` | `sgd` |
| `--weight-decay` | Weight decay (AdamW) | `0.0` |
| `--warmup-steps` | Step-level LR warmup | `0` |
| `--accum-steps` | Gradient accumulation steps | `1` |
| `--grad-clip` | Gradient clipping norm | `0.0` |

### Environment variables

| Variable | Effect | Default |
|---|---|---|
| `USE_GPU` | `"true"` / `"false"` — force CPU or auto-detect | `"true"` |

```bash
USE_GPU=false make train EXP=vit EPOCHS=1   # force CPU
```

---

## Supported experiments

| Name | Data | Model | Default LR | Notes |
|---|---|---|---|---|
| `vit` | CIFAR-10 | Small ViT | `3e-4` | patch=4, dim=192, depth=6 |
| `captioning` | Flickr8k | ViT encoder + Transformer decoder | `1e-3` | perplexity ~50 |

---

## Inference

### Export classification model to ONNX

```bash
make export EXP=vit
# exports/vit.onnx
```

Verifies PyTorch vs ONNX output (max diff < 1e-5) as part of the export.

### Generate a caption

```bash
make sample CKPT=checkpoints/captioning_xxx.pt IMG=datasets/Flicker8k_Dataset/xxx.jpg
```

Or directly:

```bash
uv run python scripts/sample_caption.py \
    --checkpoint checkpoints/captioning_xxx.pt \
    --image datasets/Flicker8k_Dataset/xxx.jpg
```

**Self-contained checkpoints**: `sample_caption.py` reconstructs the
tokenizer and model from metadata stored in the checkpoint. No extra
arguments needed.

### FastAPI service

```bash
make serve                                  # default ONNX=exports/vit.onnx
make serve ONNX=exports/vit.onnx PORT=8001
```

Endpoints:

- `GET /health` → `{"status": "ok", "model": "vit.onnx"}`
- `POST /predict` (multipart file upload) → top-k predictions

Interactive docs: <http://localhost:8000/docs>

### Docker serving image

```bash
docker compose up -d
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
checkpoints/<exp>_<timestamp>.pt  # model + optimizer + scheduler + model_init + extras
```

The checkpoint is **self-contained**: it stores everything needed to
rebuild the model and resume training.

```python
ckpt = torch.load("checkpoints/captioning_xxx.pt", weights_only=False)
# ckpt["model_init"]  -> {"vocab_size": 50257, "pad_id": 50256}
# ckpt["extras"]      -> {"tokenizer_name": "tiktoken"}
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
│   ├── __init__.py
│   ├── py.typed
│   ├── runtime.py              # constants + get_device()
│   ├── registry.py             # EXPERIMENTS (name -> data/model/task/lr)
│   ├── cli.py                  # argparse -> TrainConfig
│   ├── main.py                 # entry point
│   ├── data/
│   │   ├── tokenizers/         # Tokenizer protocol + adapters + factory
│   │   │   ├── base.py
│   │   │   ├── char.py
│   │   │   └── tiktoken_bpe.py
│   │   ├── transforms.py
│   │   ├── bundle.py           # DataBundle + DataContext
│   │   ├── cifar10.py
│   │   ├── flickr8k.py
│   │   └── datasets.py         # registry: name -> build_bundle
│   ├── models/
│   │   ├── vit.py
│   │   └── captioning.py
│   ├── tasks/
│   │   ├── base.py             # Task protocol
│   │   ├── classification.py
│   │   └── captioning.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluator.py
│   │   └── strategy.py         # SingleDevice / DDP
│   ├── experiment/
│   │   ├── config.py           # TrainConfig
│   │   ├── artifacts.py        # RunArtifacts
│   │   └── runner.py           # ExperimentRunner
│   ├── export/
│   │   └── onnx_export.py
│   ├── serving/
│   │   ├── api.py
│   │   ├── inference.py
│   │   └── schema.py
│   └── utils/
│       ├── path.py
│       ├── logger.py
│       ├── seed.py
│       └── env.py
├── tests/
│   ├── conftest.py             # global fixtures
│   ├── contracts/              # protocol contracts
│   │   ├── conftest.py
│   │   ├── test_classification_contract.py
│   │   └── test_tokenizer_contract.py
│   ├── unit/                   # fast, no IO
│   │   ├── conftest.py
│   │   └── ... (14 files)
│   └── integration/            # need real data + env flag
│       ├── conftest.py
│       ├── test_captioning_integration.py
│       ├── test_data_integration.py
│       ├── test_integration.py
│       └── test_regression.py
├── docker/
│   ├── Dockerfile.serve
│   └── Dockerfile.train
├── scripts/
│   ├── ci.sh
│   ├── download_datasets.sh
│   ├── sample_caption.py
│   └── train_ddp.sh
├── docs/
│   ├── TODO.md
│   ├── COMPLETED.md
│   ├── architecture-notes.md
│   └── memo-1.md ~ memo-10.md
├── datasets/                   # (not in git)
├── checkpoints/                # (not in git)
├── outputs/                    # (not in git)
├── exports/                    # (not in git)
├── docker-compose.yml
├── .dockerignore
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
└── uv.lock
```

---

## Development

### Common commands

```bash
make help           # list all targets
make install        # uv sync
make install-hooks  # install git pre-commit hooks
make test           # unit + contracts (fast)
make lint           # ruff check + ruff format --check + mypy
make format         # ruff check --fix + ruff format
make check          # format + lint + test
make ci             # full CI pipeline locally
make train          # see "Training"
make train-ddp      # DDP training
make train-docker   # containerized training (GPU)
make board          # TensorBoard
make export         # ONNX export
make serve          # FastAPI service
make sample         # generate caption
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

### Testing layers

- **`tests/unit/`** — fast, no real data, run by default
- **`tests/contracts/`** — verify every implementation of a protocol
  (all tokenizers, all classification models)
- **`tests/integration/`** — need real data on disk + `RUN_INTEGRATION=1`

```bash
make test                                         # unit + contracts (fast)
make test-integration                             # requires real data
make test-regression                              # accuracy baseline
uv run pytest -m "" -v                            # everything
```

Markers are defined in `pyproject.toml`; the default `addopts` excludes
`integration` and `regression`.

---

## Extending

### Adding a new model

1. Create `src/models/<name>.py` with an `nn.Module` subclass.

2. If the model's constructor needs data-derived parameters
   (e.g. `num_classes`, `vocab_size`), implement `from_data`:

   ```python
   @classmethod
   def from_data(cls, bundle: "DataBundle") -> "MyModel":
       return cls(**bundle.model_init)
   ```

   For fixed-shape models, no `from_data` is required — declare a
   `build_model` in the dataset bundle, or add a small wrapper.

3. Register it in `src/registry.py`:

   ```python
   EXPERIMENTS["my_model"] = {
       "data": "cifar10",
       "model": MyModel,
       "task": ClassificationTask,
       "lr": 1e-3,
       "category": "classification",
   }
   ```

4. Contract tests (in `tests/contracts/`) automatically pick it up
   via `EXPERIMENTS` iteration.

### Adding a new dataset

1. Create `src/data/<name>.py` with a `build_bundle(ctx: DataContext)
   -> DataBundle` function.

2. Register it in `src/data/datasets.py`:

   ```python
   DATASET_REGISTRY = {
       "cifar10": cifar10.build_bundle,
       "flickr8k": flickr8k.build_bundle,
       "my_dataset": my_dataset.build_bundle,
   }
   ```

   For classification-style datasets, also add to `DATASET_INFO`:

   ```python
   DATASET_INFO = {
       "cifar10": DatasetInfo(input_shape=(3, 32, 32), num_classes=10),
       "my_dataset": DatasetInfo(input_shape=(1, 28, 28), num_classes=10),
   }
   ```

3. Reference it in `src/registry.py`:

   ```python
   EXPERIMENTS["my_exp"] = {
       "data": "my_dataset",
       "model": MyModel,
       "task": ClassificationTask,
       "lr": 1e-3,
       "category": "classification",
   }
   ```

### Adding a new task

Subclass `Task` and implement three things:

```python
from tasks.base import Task

class MyTask(Task):
    primary_metric = "loss"
    higher_is_better = False

    @classmethod
    def from_data(cls, bundle: "DataBundle") -> "MyTask":
        return cls()

    def train_step(self, model, batch, device, dtype) -> torch.Tensor:
        """Compute loss for one batch. Do NOT call backward."""

    def eval_step(self, model, batch, device, dtype) -> dict[str, float]:
        """Compute metrics for one batch."""
```

Then register it in `src/registry.py`. `train.py` does not need to
change.

### Adding a new training strategy (FSDP, DeepSpeed, ...)

Subclass `TrainingStrategy` in `src/training/strategy.py` and add a
branch to `build_strategy()`. No other file needs to change.

---

## Reproducibility

- `set_seed(42)` fixes `random`, `numpy`, `torch`
- Every run writes a JSON snapshot with config + environment info
- CUDA determinism enabled
- Under DDP, `DistributedSampler` reshuffles per epoch via `set_epoch`

---

## License

MIT