# ============================================================
# My Test Project
# ============================================================

.PHONY: help install install-hooks test lint format train board export clean

EXP     ?= mlp
EPOCHS  ?= 1
BS      ?= 64
LR      ?=
RESUME  ?=
AMP     ?=
CKPT    ?=
OUT     ?=
OPSET   ?= 17

help:
	@echo "Available targets:"
	@echo "  make install              uv sync"
	@echo "  make install-hooks        Install git pre-commit hooks"
	@echo "  make test                 uv run pytest"
	@echo "  make lint                 uv run ruff + mypy"
	@echo "  make format               uv run ruff format"
	@echo "  make train                uv run train"
	@echo "      EXP=mlp EPOCHS=5 BS=128 LR=0.01 RESUME=path/to.pt"
	@echo "  make clean                Remove caches"
	@echo "  make board                Launch TensorBoard on outputs/"
	@echo "  make export               Export checkpoint to ONNX"
	@echo "      EXP=mlp [CKPT=path] [OUT=path] [OPSET=17]"

install:
	uv sync

install-hooks:
	uv run pre-commit install
	@echo "Pre-commit hook installed."

test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

format:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

train:
	uv run python src/main.py \
		--experiment $(EXP) \
		--epochs $(EPOCHS) \
		--batch-size $(BS) \
		$(if $(LR),--learning-rate $(LR) ,)$(if $(RESUME),--resume $(RESUME),)$(if $(AMP),--amp,)

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