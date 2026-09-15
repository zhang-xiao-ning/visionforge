# COMPLETED

已完成的工作记录。按时间倒序。

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

---

## 项目结构

```text
my_test_project/
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
│   └── utils/
│       ├── path.py
│       ├── logger.py
│       ├── seed.py
│       └── env.py
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_transforms.py
│   └── test_data.py
├── Makefile
├── pyproject.toml
├── README.md
├── TODO.md
├── COMPLETED.md
├── .gitignore
├── pack.sh
└── dump_code.sh