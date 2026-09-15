# ============================================================
# My Test Project
# ============================================================

.PHONY: help install test lint format train clean

EXP     ?= mlp
EPOCHS  ?= 1
BS      ?= 64
LR      ?=
RESUME  ?=

help:
	@echo "Available targets:"
	@echo "  make install              uv sync"
	@echo "  make test                 uv run pytest"
	@echo "  make lint                 uv run ruff + mypy"
	@echo "  make format               uv run ruff format"
	@echo "  make train                uv run train"
	@echo "      EXP=mlp EPOCHS=5 BS=128 LR=0.01 RESUME=path/to.pt"
	@echo "  make clean                Remove caches"

install:
	uv sync

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
		$(if $(LR),--learning-rate $(LR) ,)$(if $(RESUME),--resume $(RESUME),)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/