# COMPLETED

已完成的工作记录。按时间倒序。

---
## v0.1.0 — 2026-09-18

### 阶段 7：CI（第 26 步）

- [x] **第 26 步**：CI 自动化
  - `scripts/ci.sh`：平台无关的 CI 脚本
  - `make ci`：本地跑完整 CI 流程
  - `.github/workflows/ci.yml`：GitHub Actions
  - 从 Gitee Go 切到 GitHub Actions（免费、无额度限制）
  - 首次 push 成功，耗时 5m35s（冷缓存）

### 项目更名

- [x] `my_test_project` → `visionforge`
  - GitHub / Gitee 仓库改名
  - `pyproject.toml` / `docker-compose.yml` / `README.md` 同步
  - 本地目录和 WSL 目录改名
  - 由于 import 用的是"脚本式"，代码零改动

## v0.1.0 — 2026-09-16

### 阶段 4：DevOps 闭环（第 23-25 步）

- [x] **第 23 步**：ONNX 导出
  - `src/export/onnx_export.py`
  - PyTorch vs ONNX 输出验证（`max_diff < 1e-5`）
  - `make export EXP=mlp` / `make export EXP=deep_convnet`
  - 注意：`vit` 导出留待以后（Mac 上没有 vit checkpoint）
- [x] **第 24 步**：FastAPI 推理服务
  - `src/serving/api.py`（`/health` + `/predict`）
  - `src/serving/inference.py`（ONNX Runtime 推理）
  - `src/serving/schema.py`（Pydantic schema）
  - `make serve`
  - 浏览器 `http://localhost:8000/docs` 自动文档
- [x] **第 25 步**：Docker 部署
  - `docker/Dockerfile.serve`（CPU torch + 清华源 + `numpy<2`）
  - `docker-compose.yml`
  - `.dockerignore`
  - 在 WSL2 上构建运行
  - **完整闭环打通**：Mac → SSH 隧道 → 阿里云中转 → WSL2 → Docker → ONNX 推理

---

## v0.1.0 — 2026-09-15

### 阶段 1：从 CS231N 作业到项目骨架（第 1–10 步）

- [x] **第 1 步**：修 import，统一风格，能跑起来
- [x] **第 2 步**：删教学代码，模型抽成 `nn.Module` 类
- [x] **第 3 步**：训练循环工程化（每 epoch 评估、保存最优、test 评估）
- [x] **第 4 步**：配置管理（`TrainConfig` + argparse）
- [x] **第 5 步**：日志与输出（`logging` + CSV + JSON）
- [x] **第 6 步**：数据模块化（`transforms.py` + 注册表）
- [x] **第 7 步**：可复现性（seed + config 快照 + env 快照）
- [x] **第 8 步**：pytest 冒烟测试
- [x] **第 9 步**：学习率调度 + early stopping + 恢复训练
- [x] **第 10 步**：接入 ViT（注册表加一行）

### 阶段 2：项目外观（第 11–13 步）

- [x] **第 11 步**：`pyproject.toml` + `README.md` + `.gitignore`
  - Python 版本全对齐 3.12（`requires-python` / `ruff` / `mypy`）
  - 使用 `[dependency-groups]`（PEP 735，uv 推荐）
  - 清理 `.idea/`、`__pycache__/`、`*.egg-info/` 出 git
  - 保留 `uv.lock` 在 git 中
- [x] **第 12 步**：`Makefile`
  - `make install` / `test` / `lint` / `format` / `train` / `clean` / `help`
  - 用 `uv run` 保证环境一致
- [x] **第 13 步**：暴露公共 API
  - `src/__init__.py` 定义 `__all__` 和 `__version__`
  - `src/py.typed` 标记类型

### 阶段 3：训练增强（第 14–15 步）

- [x] **第 14 步**：数据增强
  - 训练集：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`
  - 验证/测试集：只归一化
  - 效果：deep_convnet 从 70.08% → 73.82%
- [x] **第 15 步**：混合精度 AMP
  - `torch.cuda.amp.autocast` + `GradScaler`
  - 只在 CUDA 上生效，MPS / CPU 自动退化
  - `--amp` 命令行开关

### 阶段 4：可观测（第 16 步）

- [x] **第 16 步**：TensorBoard
  - `SummaryWriter` 记录 loss / val_acc / lr
  - `outputs/<exp>_<timestamp>/` 下生成 event 文件
  - `make board` 一键启动

### 阶段 5：本地开发环境（第 17–19 步）

- [x] **第 17 步**：ruff 代码风格清理
  - 删除坏文件 `src/utils/init.py`
  - import 排序、`%` → f-string、格式化
  - `ruff check` 全过
- [x] **第 18 步**：pre-commit 本地 hook
  - `.pre-commit-config.yaml` 用 local 模式（不联网）
  - ruff + ruff-format + mypy
- [x] **第 19 步**：`make install-hooks`
  - 新人 clone 后一句命令装 hook

### 阶段 6：工程质量（第 20–22 步）

- [x] **第 20 步**：类型注解补齐
  - 打开 `disallow_untyped_defs = true`
  - 30 个类型错误全部修复
  - `mypy src/` 全过
- [x] **第 21 步**：补单元测试
  - `test_evaluator.py`（3）
  - `test_trainer.py`（5）
  - `test_logger.py`（4）
  - 测试与硬件解耦（`USE_GPU` 环境变量）
  - 总计 22 passed, 1 skipped
- [x] **第 22 步**：README 完善
  - CLI 参数表
  - 环境变量表
  - 完整目录结构
  - "如何加新模型" / "如何加新数据集"

---

## 项目结构

```text
visionforge/
├── src/
│   ├── __init__.py              # 公共 API + 版本号
│   ├── py.typed
│   ├── config.py                # TrainConfig + device
│   ├── main.py                  # CLI 入口
│   ├── data/
│   │   ├── transforms.py
│   │   ├── cifar10.py
│   │   └── datasets.py          # 注册表
│   ├── models/
│   │   ├── mlp.py
│   │   ├── shallow_convnet.py
│   │   ├── deep_convnet.py
│   │   └── vit.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluator.py
│   ├── export/
│   │   └── onnx_export.py       # ONNX 导出
│   ├── serving/
│   │   ├── api.py               # FastAPI 路由
│   │   ├── inference.py         # ONNX 推理
│   │   └── schema.py            # Pydantic schema
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
│   └── test_logger.py
├── docker/
│   └── Dockerfile.serve         # 推理镜像
├── docker-compose.yml
├── .dockerignore
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
├── README.md
├── TODO.md
├── COMPLETED.md
├── memo-1.md ~ memo-5.md
├── .gitignore
├── pack.sh
└── dump_code.sh