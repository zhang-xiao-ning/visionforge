# ============================================================
# visionforge
# ============================================================

.PHONY: help install install-hooks test test-integration test-regression \
        lint format check train train-docker train-ddp \
        board export serve sample ci clean

# ---- Defaults ----
EXP     ?= vit
EPOCHS  ?= 1
BS      ?= 64
LR      ?=
RESUME  ?=
AMP     ?=
CKPT    ?=
OUT     ?=
OPSET   ?= 17
ONNX    ?= exports/vit.onnx
PORT    ?= 8000
NPROC   ?= 1
IMG     ?=
MAXTOK  ?=
NUM     ?=

# ---- Training extras ----
OPT     ?=
WD      ?=
WARMUP  ?=
ACCUM   ?=
CLIP    ?=

# ---- Help ----
help:
	@echo "Available targets:"
	@echo "  make install              uv sync"
	@echo "  make install-hooks        Install git pre-commit hooks"
	@echo "  make test                 unit + contracts (fast)"
	@echo "  make test-integration     integration tests (needs real data + RUN_INTEGRATION=1)"
	@echo "  make test-regression      regression tests (needs RUN_REGRESSION=1)"
	@echo "  make lint                 ruff + mypy"
	@echo "  make format               ruff --fix + format"
	@echo "  make check                format + lint + test"
	@echo "  make train                run training"
	@echo "      EXP=vit EPOCHS=5 BS=128 LR=3e-4 RESUME=... AMP=1"
	@echo "      OPT=adamw WD=0.01 WARMUP=100 ACCUM=4 CLIP=1.0"
	@echo "  make train-docker         training inside Docker (GPU)"
	@echo "  make train-ddp            DDP training (NPROC=1,2,...)"
	@echo "  make board                TensorBoard on outputs/"
	@echo "  make export               export checkpoint to ONNX"
	@echo "      EXP=vit [CKPT=path] [OUT=path] [OPSET=17]"
	@echo "      ... NUM=128 (limit training samples)"
	@echo "  make serve                FastAPI inference server"
	@echo "      ONNX=exports/vit.onnx PORT=8000"
	@echo "  make sample               generate caption for an image"
	@echo "      CKPT=path/to.pt IMG=path/to.jpg [MAXTOK=32]"
	@echo "  make ci                   run scripts/ci.sh"
	@echo "  make clean                remove caches"

# ---- Install ----
install:
	uv sync

install-hooks:
	uv run pre-commit install
	@echo "Pre-commit hook installed."

# ---- Tests ----
test:
	uv run pytest -v

test-integration:
	RUN_INTEGRATION=1 uv run pytest -m integration -v

test-regression:
	RUN_REGRESSION=1 uv run pytest -m regression -v

# ---- Quality ----
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

format:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

check: format lint test

ci:
	bash scripts/ci.sh

# ---- Training ----
TRAIN_ARGS = \
	--experiment $(EXP) \
	--epochs $(EPOCHS) \
	--batch-size $(BS) \
    $(if $(NUM),--num-train $(NUM),) \
    $(if $(LR),--learning-rate $(LR),) \
	$(if $(RESUME),--resume $(RESUME),) \
	$(if $(AMP),--amp,) \
	$(if $(OPT),--optimizer $(OPT),) \
	$(if $(WD),--weight-decay $(WD),) \
	$(if $(WARMUP),--warmup-steps $(WARMUP),) \
	$(if $(ACCUM),--accum-steps $(ACCUM),) \
	$(if $(CLIP),--grad-clip $(CLIP),)

train:
	uv run python src/main.py $(TRAIN_ARGS)

train-docker:
	docker compose --profile train run --rm train $(TRAIN_ARGS)

train-ddp:
	NPROC=$(NPROC) bash scripts/train_ddp.sh $(TRAIN_ARGS)

# ---- Export / Serve ----
export:
	uv run python src/export/onnx_export.py \
		--experiment $(EXP) \
		$(if $(CKPT),--checkpoint $(CKPT),) \
		$(if $(OUT),--output $(OUT),) \
		--opset $(OPSET)

sample:
	uv run python scripts/sample_caption.py \
		--checkpoint $(CKPT) \
		--image $(IMG) \
		$(if $(MAXTOK),--max-new-tokens $(MAXTOK),)

serve:
	ONNX_PATH=$(ONNX) uv run uvicorn serving.api:app --host 0.0.0.0 --port $(PORT) --reload --app-dir src

board:
	uv run tensorboard --logdir outputs/

# ---- Clean ----
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/