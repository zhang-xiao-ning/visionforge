# visionforge

![CI](https://github.com/zhang-xiao-ning/visionforge/actions/workflows/ci.yml/badge.svg)

A small, clean, extensible PyTorch training framework.

**Design goal** — adding a new model, task, dataset, or metric should
mean writing one file. The training loop, evaluation pipeline,
checkpointing, logging, distributed training, and serving are already
handled by the framework.

**What's inside** — a pluggable experiment registry, metrics as
first-class objects with their own data sources, multi-stage training
with parameter-group freezing, DDP, AMP, gradient accumulation, ONNX
export with numerical verification, a FastAPI inference service, and a
full local dev environment (ruff + mypy + pytest + pre-commit + CI).

**Reference experiments** — CIFAR-10 classification and Flickr8k image
captioning, both intended as worked examples of the extension points
above.

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

### YAML config

```bash
make train CFG=configs/vit-baseline.yaml
```

YAML format:

```yaml
experiment: vit          # required; must exist in the registry
description: baseline    # optional, human-readable
learning_rate: 3e-4
batch_size: 128
epochs: 10

# multi-stage (optional)
stages:
  - name: align
    freeze: [vision, llm_base]
    epochs: 1
  - name: instruct
    freeze: [vision]
    epochs: 2
```

Priority chain:

```text
TrainConfig default < Experiment.config < YAML < CLI < stage.overrides
```

Example configs live in `configs/`.

### Common training options

| Goal | Command |
|---|---|
| AdamW + weight decay | `make train EXP=captioning OPT=adamw WD=0.01` |
| Warmup + cosine | `make train EXP=captioning OPT=adamw WARMUP=100` |
| Large effective batch | `make train EXP=captioning ACCUM=4` |
| Stability | `make train EXP=captioning CLIP=1.0` |
| Mixed precision (CUDA) | `make train EXP=captioning AMP=1` |
| Quick debug | `make train EXP=vit NUM=1000` |

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
| `--config` | Path to a YAML config file | none |
| `--experiment` | Experiment name (`vit` / `captioning`) | `vit` |
| `--batch-size` | Batch size | per-experiment default |
| `--epochs` | Number of epochs | per-experiment default |
| `--learning-rate` | Learning rate | per-experiment default |
| `--momentum` | SGD momentum | per-experiment default |
| `--no-nesterov` | Disable Nesterov momentum | off |
| `--seed` | Random seed | per-experiment default |
| `--lr-scheduler` | `none` / `step` / `cosine` | per-experiment default |
| `--step-size` | StepLR step size | per-experiment default |
| `--gamma` | StepLR decay factor | per-experiment default |
| `--early-stop-patience` | Early stopping patience | `0` (disabled) |
| `--resume` | Resume from checkpoint path | none |
| `--amp` | Mixed precision (CUDA only) | off |
| `--num-train` | Limit training samples | all |
| `--optimizer` | `sgd` / `adamw` | per-experiment default |
| `--weight-decay` | Weight decay (AdamW) | per-experiment default |
| `--warmup-steps` | Step-level LR warmup | per-experiment default |
| `--accum-steps` | Gradient accumulation steps | per-experiment default |
| `--grad-clip` | Gradient clipping norm | per-experiment default |

### Environment variables

| Variable | Effect | Default |
|---|---|---|
| `USE_GPU` | `"true"` / `"false"` — force CPU or auto-detect | `"true"` |

```bash
USE_GPU=false make train EXP=vit EPOCHS=1   # force CPU
```

---

## Reference experiments

| Name | Data | Model | Metrics | Primary | Notes |
|---|---|---|---|---|---|
| `vit` | CIFAR-10 | Small ViT | `acc`, `loss` | `acc` | patch=4, dim=192, depth=6 |
| `captioning` | Flickr8k | ViT encoder + Transformer decoder | `perplexity`, `bleu4` | `perplexity` | perplexity ~50 |

---

## Evaluation

Metrics are first-class objects (`src/evaluation/`). Each metric
declares:

- `name` — key in CSV / log
- `higher_is_better` — for best-model selection
- `run_every_n_epochs` — `1` (every epoch), `N` (every N), `None` (test only)
- `train_loader` / `test_loader` — which data to iterate
- `evaluate(model, loader, device, dtype) -> float` — the scoring itself

Evaluation data may differ in **granularity** from training data. For
example, captioning evaluation uses image-level batches (one image + all
5 references), while training uses pair-level batches (one image + one
caption). This is handled by `EvalBundle` (separate from `DataBundle`).

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

**Self-contained checkpoints**: `sample_caption.py` reconstructs the
tokenizer and model from metadata stored in the checkpoint.

### FastAPI service

```bash
make serve                                  # default ONNX=exports/mlp.onnx
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

Checkpoints are **self-contained**: they store everything needed to
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
│   ├── runtime.py              # constants + get_device() + USE_CUDA
│   ├── registry.py             # EXPERIMENTS (name -> Experiment)
│   ├── cli.py                  # argparse + YAML -> run parameters
│   ├── main.py                 # entry point
│   ├── data/
│   │   ├── tokenizers/         # Tokenizer protocol + adapters + factory
│   │   │   ├── base.py
│   │   │   ├── char.py
│   │   │   └── tiktoken_bpe.py
│   │   ├── transforms.py
│   │   ├── bundle.py           # DataBundle + EvalBundle + DataContext
│   │   ├── cifar10.py
│   │   ├── flickr8k.py         # pair-level + image-level datasets
│   │   └── datasets.py         # registry: name -> (DataBundle, EvalBundle|None)
│   ├── models/
│   │   ├── base.py             # Model ABC
│   │   ├── vit.py
│   │   ├── mlp.py
│   │   ├── mamba.py
│   │   └── captioning.py
│   ├── tasks/
│   │   ├── base.py             # Task ABC (train_step only)
│   │   ├── classification.py
│   │   └── captioning.py
│   ├── evaluation/             # first-class metrics
│   │   ├── base.py             # Metric ABC + train_loader/test_loader
│   │   ├── metrics.py          # Accuracy / CrossEntropy / Perplexity
│   │   └── bleu.py             # corpus_bleu + BLEU4
│   ├── training/
│   │   ├── train.py
│   │   ├── hooks.py
│   │   ├── strategy.py         # SingleDevice / DDP
│   │   └── tracker.py
│   ├── experiment/
│   │   ├── spec.py             # TrainConfig + Stage + Experiment
│   │   ├── loader.py           # YAML load + validate + coerce
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
│   │   ├── test_model_contract.py
│   │   └── test_tokenizer_contract.py
│   ├── unit/                   # fast, no IO
│   └── integration/            # need real data + env flag
├── configs/                    # example YAML experiment configs
├── docker/
│   ├── Dockerfile.serve
│   └── Dockerfile.train
├── scripts/
│   ├── ci.sh
│   ├── download_datasets.sh
│   ├── sample_caption.py
│   └── train_ddp.sh
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
  (all tokenizers, all classification models, all registered models)
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

The framework is designed to be extended. Each of the following takes
one file and no changes to the training loop:

### Adding a new model

1. Create `src/models/<name>.py` with a `Model` subclass.

2. If the model's constructor needs data-derived parameters
   (e.g. `num_classes`, `vocab_size`), override `from_data`:

   ```python
   from models.base import Model

   class MyModel(Model):
       @classmethod
       def from_data(cls, bundle):
           return cls(**bundle.model_init)
   ```

   The default `from_data` already does this. Override only if you need
   `bundle.extras`.

3. Register it in `src/registry.py`:

   ```python
   EXPERIMENTS["my_model"] = Experiment(
        model=MyModel,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=1e-3),
        category="classification",
      )
   ```

4. Contract tests (in `tests/contracts/`) automatically pick it up.

### Adding a new dataset

1. Create `src/data/<name>.py` with a `build_bundle(ctx) -> (DataBundle, EvalBundle | None)`
   function.

2. Register it in `src/data/datasets.py`:

   ```python
   DATASET_REGISTRY = {
       "cifar10": cifar10.build_bundle,
       "flickr8k": flickr8k.build_bundle,
       "my_dataset": my_dataset.build_bundle,
   }
   ```

   Return `(DataBundle(...), None)` if no separate evaluation loader is
   needed.

3. Reference it from `src/registry.py`.

### Adding a new task

Subclass `Task` and implement one method:

   ```python
    from tasks.base import Task

    class MyTask(Task):
        def train_step(self, model, batch, device, dtype) -> torch.Tensor:
            """Compute loss for one batch. Do NOT call backward."""
   ```

Then register it in `src/registry.py`. `train.py` does not need to change.

### Adding a new metric

1. Create `src/evaluation/<name>.py` with a `Metric` subclass:

   ```python
   from evaluation.base import Metric

   class MyMetric(Metric):
       name = "mymetric"
       higher_is_better = True
       run_every_n_epochs = 1        # 1 = every epoch, N = every N, None = test only

       @classmethod
       def from_data(cls, data, eval_data=None):
           return cls()

       def evaluate(self, model, loader, device, dtype) -> float:
           ...
   ```

2. Add it to the experiment's `metrics` list in `src/registry.py`:

   ```python
   metrics=[Perplexity, BLEU4],
   ```

3. If the metric needs a different data source than the default
   (`data.loader_val` / `data.loader_test`), override `train_loader` /
   `test_loader`. `BLEU4` is the reference: it uses `eval_data.loader`
   (image-level batches with all references).

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

Code: MIT (see [LICENSE](LICENSE)).

Datasets are **not** covered by this license:
- CIFAR-10: released by the University of Toronto for research use.
- Flickr8k: images are copyrighted by their original Flickr uploaders;
  the dataset is distributed for research purposes only.
  This repository does not redistribute any dataset files.
