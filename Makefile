# ============================================================
# visionforge
# ============================================================

.PHONY: help install install-hooks test test-integration test-regression lint format check train train-docker train-ddp board export serve ci clean

EXP     ?= mlp
EPOCHS  ?= 1
BS      ?= 64
LR      ?=
RESUME  ?=
AMP     ?=
CKPT    ?=
OUT     ?=
OPSET   ?= 17
ONNX    ?= exports/mlp.onnx
PORT    ?= 8000
NPROC   ?= 1

help:
	@echo "Available targets:"
	@echo "  make install              uv sync"
	@echo "  make install-hooks        Install git pre-commit hooks"
	@echo "  make test                 uv run pytest"
	@echo "  make test-integration     Run integration tests (requires real CIFAR-10)"
	@echo "  make test-regression      Run regression tests (accuracy baseline)"
	@echo "  make lint                 uv run ruff + mypy"
	@echo "  make format               uv run ruff format"
	@echo "  make check                Run format + lint + test (dev workflow)"
	@echo "  make train                uv run train"
	@echo "      EXP=mlp EPOCHS=5 BS=128 LR=0.01 RESUME=path/to.pt"
	@echo "  make train-docker         Train inside Docker (GPU)"
	@echo "      EXP=mlp EPOCHS=5"
	@echo "  make train-ddp            DDP training (NPROC=1,2,...)"
	@echo "      EXP=mlp EPOCHS=5 NPROC=1"
	@echo "  make board                Launch TensorBoard on outputs/"
	@echo "  make export               Export checkpoint to ONNX"
	@echo "      EXP=mlp [CKPT=path] [OUT=path] [OPSET=17]"
	@echo "  make serve                Start FastAPI inference server"
	@echo "      ONNX=exports/mlp.onnx PORT=8000"
	@echo "  make ci                   Run full CI pipeline locally"
	@echo "  make clean                Remove caches"

install:
	uv sync

install-hooks:
	uv run pre-commit install
	@echo "Pre-commit hook installed."

test:
	uv run pytest -v

test-integration:
	RUN_INTEGRATION=1 uv run pytest tests/test_integration.py -v

test-regression:
	RUN_REGRESSION=1 uv run pytest tests/test_regression.py -v

ci:
	bash scripts/ci.sh

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

format:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

check: format lint test

train:
	uv run python src/main.py \
		--experiment $(EXP) \
		--epochs $(EPOCHS) \
		--batch-size $(BS) \
		$(if $(LR),--learning-rate $(LR) ,)$(if $(RESUME),--resume $(RESUME),)$(if $(AMP),--amp,)

train-docker:
	docker compose --profile train run --rm train \
		--experiment $(EXP) \
		--epochs $(EPOCHS) \
		$(if $(BS),--batch-size $(BS),) \
		$(if $(LR),--learning-rate $(LR),) \
		$(if $(RESUME),--resume $(RESUME),)

train-ddp:
	NPROC=$(NPROC) bash scripts/train_ddp.sh \
		--experiment $(EXP) \
		--epochs $(EPOCHS) \
		--batch-size $(BS) \
		$(if $(LR),--learning-rate $(LR),) \
		$(if $(RESUME),--resume $(RESUME),)

export:
	uv run python src/export/onnx_export.py \
		--experiment $(EXP) \
		$(if $(CKPT),--checkpoint $(CKPT),) \
		$(if $(OUT),--output $(OUT),) \
		--opset $(OPSET)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

board:
	uv run tensorboard --logdir outputs/

serve:
	ONNX_PATH=$(ONNX) uv run uvicorn serving.api:app --host 0.0.0.0 --port $(PORT) --reload --app-dir src