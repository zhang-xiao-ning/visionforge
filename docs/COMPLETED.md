# COMPLETED

已完成的工作记录。按时间倒序。

---

## v0.1.1 — 2026-09-26

### 阶段 12：Captioning（第 33-40 步）

- [x] **第 33 步**：Task 抽象
  - `tasks/base.py`：`Task` ABC，含 `primary_metric` + `higher_is_better`
  - `tasks/classification.py`：从 `train.py` 抽出分类逻辑
  - `train.py` / `evaluator.py` 变成任务无关
- [x] **第 34 步**：字符级 Tokenizer
  - `data/tokenizer.py`：`CharTokenizer`（后被迁移）
- [x] **第 35 步**：Flickr8k Dataset
  - `data/flickr8k.py`：`Flickr8kDataset` + `build_captioning_loaders`
  - 自回归 shift：`input_ids[:-1]` / `target_ids[1:]`
  - target padding 用 `-100`（PyTorch ignore_index 标准）
- [x] **第 36 步**：Captioning 模型
  - `models/captioning.py`：ViT Encoder + Transformer Decoder
  - Weight tying：`lm_head.weight = token_embed.weight`
- [x] **第 37 步**：CaptioningTask
  - `tasks/captioning.py`：cross-entropy + perplexity
  - `primary_metric = "perplexity"`, `higher_is_better = False`
- [x] **第 38 步**：Tokenizer 协议讨论
  - 决策：能力驱动（鸭子类型），不引入 ABC 基类
- [x] **第 39 步**：Tokenizer 协议化 + setup 机制
  - `data/tokenizers/`：`Tokenizer` Protocol + `CharTokenizer` + `TiktokenTokenizer` + 工厂
  - `models/base.py`：`SetupContext` + `ExperimentBundle`
  - `CaptioningModel.setup` classmethod（鸭子类型）
  - `runner.py` 加 `hasattr(model_cls, "setup")` 分支（带 TODO）
- [x] **第 40 步**：训练跑通 + 采样
  - 训练：perplexity 50257 → 50.22（5 epoch）
  - 采样：第一次生成 caption
  - **两个 NaN bug 修复**：`tie_weights` 默认值 + `tgt_key_padding_mask`

**关键产物**：
- `data/tokenizers/`：协议化，切中文只需加 1 文件 + 1 行注册
- `models/captioning.py`：多模态模型
- 训练结果：Flickr8k 上 perplexity 50.22

## v0.1.1 — 2026-09-23

### 阶段 10：工程打磨（第 29-31 步）

- [x] **第 29 步**：版本号单一来源
  - `src/__init__.py` 用 `importlib.metadata` 读版本
  - 只改 `../pyproject.toml`，`__version__` 自动同步
  - 加 `../tests/test_version.py`（3 个测试）
- [x] **第 30 步**：`num_train` 参数 + 集成测试
  - CLI 加 `--num-train`
  - `ExperimentRunner` 透传
  - 集成测试（`../tests/test_integration.py`）：训练 + 恢复训练
  - `make test-integration`：1 分钟
- [x] **第 31 步**：回归测试
  - `../tests/test_regression.py`：`mlp` + `deep_convnet` 的精度基线
  - 阈值宽松（防崩不防微调）
  - `make test-regression`：50 秒
- [x] **第 32 步**：框架边界
  - `src/framework/__init__.py`：re-export 框架公开 API
  - `main.py` 改用 `from framework import ...`
  - 不改行为，不深层解耦（`ExperimentRunner` 仍 import 应用层）
  - 未来拆 repo 从这层切

**测试金字塔**：
- 单元：41 passed（3 秒）
- 集成：2 passed（1 分钟）
- 回归：2 passed（50 秒）

## v0.1.0 — 2026-09-22

### 阶段 9：架构重构 + DDP（第 28 步）

- [x] **Commit 1**：`TrainingStrategy` 抽象（SingleDevice + DDP）
- [x] **Commit 2**：`RunArtifacts` 抽象（路径 / logger / recorder / writer）
- [x] **Commit 3**：`ExperimentRunner` 编排 + `main.py` 精简（200 → 90 行）
- [x] **Commit 4**：`strategy` 接入 `datasets.py` / `train.py`（向后兼容）
- [x] **Commit 5**：DDP 脚本 + Makefile + 4070 验证
  - `../scripts/train_ddp.sh`
  - `make train-ddp EXP=mlp EPOCHS=1 NPROC=1`
  - 4070 上跑通，可复现性验证通过
- [x] **Commit 6**：文档更新（memo-8 + COMPLETED + TODO）

**重构效果**：
- 加 FSDP / DeepSpeed 只需加一个 `TrainingStrategy` 子类
- `main.py` / `runner.py` / `datasets.py` / `train.py` 不动
- 数值逐位一致（重构不改变行为）

## v0.1.0 — 2026-09-20

### 阶段 8：训练镜像（第 27 步）

- [x] **第 27 步**：训练 Docker 镜像
  - `../docker/Dockerfile.train`（CUDA 12.1 + torch 2.2.2+cu121）
  - `../docker-compose.yml` 加 `train` service（`profiles: [train]`）
  - GPU 直通：`deploy.resources.reservations.devices`
  - 卷挂载：`datasets:ro` / `checkpoints` / `outputs`
  - 在 4070 WSL2 上验证：`using device: cuda`，1 epoch ~6 秒
  - 可复现：两次运行数值完全一致


## v0.1.0 — 2026-09-18

### 阶段 7：CI（第 26 步）

- [x] **第 26 步**：CI 自动化
  - `../scripts/ci.sh`：平台无关的 CI 脚本
  - `make ci`：本地跑完整 CI 流程
  - `../.github/workflows/ci.yml`：GitHub Actions
  - 从 Gitee Go 切到 GitHub Actions（免费、无额度限制）
  - 首次 push 成功，耗时 5m35s（冷缓存）
  - **CI 时长从 5m30s 优化到 45s**（CPU torch + 删 lock）

### 项目更名

- [x] `my_test_project` → `visionforge`
  - GitHub / Gitee 仓库改名
  - `../pyproject.toml` / `../docker-compose.yml` / `../README.md` 同步
  - 本地目录和 WSL 目录改名
  - 由于 import 用的是"脚本式"，代码零改动

## v0.1.0 — 2026-09-16


### 阶段 6：DevOps 闭环（第 23-25 步）

- [x] **第 23 步**：ONNX 导出
  - `../src/export/onnx_export.py`
  - PyTorch vs ONNX 输出验证（`max_diff < 1e-5`）
  - `make export EXP=mlp` / `make export EXP=deep_convnet`
  - 注意：`vit` 导出留待以后（Mac 上没有 vit checkpoint）
- [x] **第 24 步**：FastAPI 推理服务
  - `../src/serving/api.py`（`/health` + `/predict`）
  - `../src/serving/inference.py`（ONNX Runtime 推理）
  - `../src/serving/schema.py`（Pydantic schema）
  - `make serve`
  - 浏览器 `http://localhost:8000/docs` 自动文档
- [x] **第 25 步**：Docker 部署
  - `../docker/Dockerfile.serve`（CPU torch + 清华源 + `numpy<2`）
  - `../docker-compose.yml`
  - `../.dockerignore`
  - 在 WSL2 上构建运行
  - **完整闭环打通**：Mac → SSH 隧道 → 阿里云中转 → WSL2 → Docker → ONNX 推理

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

### 阶段 5：本地开发环境（第 17–19 步）

- [x] **第 17 步**：ruff 代码风格清理
  - 删除坏文件 `src/utils/init.py`
  - import 排序、`%` → f-string、格式化
  - `ruff check` 全过
- [x] **第 18 步**：pre-commit 本地 hook
  - `../.pre-commit-config.yaml` 用 local 模式（不联网）
  - ruff + ruff-format + mypy
- [x] **第 19 步**：`make install-hooks`
  - 新人 clone 后一句命令装 hook

### 阶段 4：可观测（第 16 步）

- [x] **第 16 步**：TensorBoard
  - `SummaryWriter` 记录 loss / val_acc / lr
  - `outputs/<exp>_<timestamp>/` 下生成 event 文件
  - `make board` 一键启动
  - 
### 阶段 3：训练增强（第 14–15 步）

- [x] **第 14 步**：数据增强
  - 训练集：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`
  - 验证/测试集：只归一化
  - 效果：deep_convnet 从 70.08% → 73.82%
- [x] **第 15 步**：混合精度 AMP
  - `torch.cuda.amp.autocast` + `GradScaler`
  - 只在 CUDA 上生效，MPS / CPU 自动退化
  - `--amp` 命令行开关

### 阶段 2：项目外观（第 11–13 步）

- [x] **第 11 步**：`../pyproject.toml` + `../README.md` + `../.gitignore`
  - Python 版本全对齐 3.12（`requires-python` / `ruff` / `mypy`）
  - 使用 `[dependency-groups]`（PEP 735，uv 推荐）
  - 清理 `../.idea`、`__pycache__/`、`*.egg-info/` 出 git
  - 保留 `../uv.lock` 在 git 中
- [x] **第 12 步**：`../Makefile`
  - `make install` / `test` / `lint` / `format` / `train` / `clean` / `help`
  - 用 `uv run` 保证环境一致
- [x] **第 13 步**：暴露公共 API
  - `src/__init__.py` 定义 `__all__` 和 `__version__`
  - `../src/py.typed` 标记类型

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

---