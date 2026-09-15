# Memo 2: 从 CS231N 作业到企业级 PyTorch 项目

> 记录一次完整的项目改造过程
> 时间：2026-09-14 ~ 2026-09-15
> 版本：v0.1.0

---

## 一、项目概述

**起点**：CS231N Assignment 2 的 PyTorch 代码，所有逻辑混在 `main.py` 和 `layers.py` 两个文件里。

**终点**：一个可复用、可配置、可复现、可扩展的 CIFAR-10 图像分类训练框架。

**核心原则**：

> 任何一个变化，都只需要改一个地方。
> - 改超参 → `config.py`
> - 加模型 → `models/xxx.py` + 注册表一行
> - 加数据集 → `data/xxx.py` + 注册表一行
> - 换训练策略 → `training/train.py`

---

## 二、目录结构

```text
my_test_project/
├── src/
│   ├── __init__.py              # 公开 API + 版本号
│   ├── py.typed                 # mypy 类型标记
│   ├── config.py                # TrainConfig + device + 全局常量
│   ├── main.py                  # CLI 入口，组装一切
│   ├── data/
│   │   ├── transforms.py        # 数据增强
│   │   ├── cifar10.py           # CIFAR-10 具体配置
│   │   └── datasets.py          # 数据集注册表 + loader 构建
│   ├── models/
│   │   ├── mlp.py
│   │   ├── shallow_convnet.py
│   │   ├── deep_convnet.py
│   │   └── vit.py
│   ├── training/
│   │   ├── train.py             # 通用训练循环
│   │   └── evaluator.py         # 通用评估
│   └── utils/
│       ├── path.py              # 路径常量
│       ├── logger.py            # 日志 + CSV
│       ├── seed.py              # 随机种子
│       └── env.py               # 环境快照
├── tests/
│   ├── conftest.py              # 共享 fixture
│   ├── test_models.py
│   ├── test_transforms.py
│   └── test_data.py
├── datasets/                    # 数据（不进 git）
├── checkpoints/                 # 权重（不进 git）
├── outputs/                     # 日志/CSV/JSON（不进 git）
├── Makefile                     # 常用命令
├── pyproject.toml               # 依赖 + 工具配置
├── README.md
├── .gitignore
├── pack.sh                      # 打包脚本
└── dump_code.sh                 # 导出所有代码为文本
```

---

## 三、用法

### 1. 安装

```bash
git clone <repo>
cd my_test_project
uv sync
```

### 2. 跑测试

```bash
make test
# 或
uv run pytest -v
```

### 3. 训练

```bash
make train EXP=mlp EPOCHS=5
make train EXP=deep_convnet EPOCHS=10 LR=0.1
make train EXP=vit EPOCHS=10 BS=256 AMP=1
```

或直接：

```bash
uv run python src/main.py --experiment vit --epochs 10 --batch-size 256 --amp
```

### 4. 断点续训

```bash
make train EXP=vit EPOCHS=20 RESUME=checkpoints/vit_20260914_223852.pt
```

### 5. 查看帮助

```bash
make help
uv run python src/main.py --help
```

### 6. 命令行参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--experiment` | 实验名 | `mlp` |
| `--dataset` | 数据集名 | `cifar10` |
| `--batch-size` | batch size | `64` |
| `--epochs` | 训练轮数 | `1` |
| `--learning-rate` | 学习率 | 每个实验的默认值 |
| `--momentum` | SGD 动量 | `0.9` |
| `--no-nesterov` | 关闭 Nesterov | 关闭 |
| `--seed` | 随机种子 | `42` |
| `--lr-scheduler` | `none` / `step` / `cosine` | `none` |
| `--step-size` | StepLR 步长 | `10` |
| `--gamma` | StepLR 衰减率 | `0.1` |
| `--early-stop-patience` | 早停耐心值 | `0`（不启用） |
| `--resume` | 从 checkpoint 恢复 | 无 |
| `--amp` | 启用混合精度 | 关闭 |

### 7. 产出文件

每次运行生成：

```text
outputs/<exp>_<timestamp>.log     # 完整日志
outputs/<exp>_<timestamp>.csv     # epoch, train_loss, val_acc, lr
outputs/<exp>_<timestamp>.json    # config + environment 快照
checkpoints/<exp>_<timestamp>.pt  # model + optimizer + scheduler + epoch + best_acc
```

---

## 四、15 步知识点总结

### 第 1 步：能跑起来

**目标**：修 import，统一风格

**知识点**：
- import 风格有两种：脚本式（`from models.xxx`）和包式（`from src.models.xxx`）
- 脚本式需要 `src` 在 `sys.path` 里，PyCharm 用 **Sources Root** 实现
- 删掉无用文件（比如 `utils/init.py` 这种没改完的）

**教训**：先让它跑，再让它好。

---

### 第 2 步：删教学代码，抽模型

**目标**：模型变成 Module 类，`main.py` 里不再写 `nn.Conv2d`

**知识点**：
- CS231N 的 `functional.py`、`random_weight`、`zero_weight` 是教学产物，企业不用
- 企业里 99% 用 `nn.Module` + `optim`
- `nn.Flatten()` 替代手写 `flatten()`

**教训**：教学代码 ≠ 企业代码。CS231N 让你理解原理，企业用抽象。

---

### 第 3 步：训练循环工程化

**目标**：每个 epoch 评估、保存最优、test 评估

**知识点**：
- 训练循环的骨架：`forward → loss → backward → step`
- best_state 保存：`{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}`
- 每个 epoch 结束评估一次，而不是每 N 步
- 用最优权重做 test

**教训**：训练循环是一个**函数**，不是一段脚本。

---

### 第 4 步：配置管理

**目标**：`TrainConfig` + argparse

**知识点**：
- `@dataclass` 管理配置
- `argparse` 管理命令行
- **超参集中**，不散落在代码里
- 配置分三层：
  - 全项目统一 → `config.py` 模块级常量
  - 实验级 → `TrainConfig` 字段
  - 一次调用 → 函数参数

**教训**：参数化是所有工程化的第一步。

---

### 第 5 步：日志与输出

**目标**：`logging` + CSV + JSON

**知识点**：
- `logging` 替代 `print`，同时输出终端和文件
- `CSVRecorder` 每个 epoch 追加一行，方便画曲线
- 文件带时间戳，多次运行不覆盖
- `FileHandler` + `StreamHandler` 双通道

**教训**：`print` 是调试工具，`logging` 是工程工具。

---

### 第 6 步：数据模块化

**目标**：注册表模式

**知识点**：
- `DATASET_REGISTRY = {"cifar10": cifar10.build_datasets}`
- `transforms.py` 独立，方便改数据增强
- `build_loaders(name=..., batch_size=...)` 参数化
- `range(num_train, total)` 不写死 50000

**教训**：**注册表模式**是加东西不改老代码的核心技巧。

---

### 第 7 步：可复现性

**目标**：seed + config 快照 + env 快照

**知识点**：
- `set_seed(42)` 固定 `random` / `numpy` / `torch`
- `torch.backends.cudnn.deterministic = True`
- JSON 快照：config + dataset + batch_size + env
- `get_env_info()` 记录 Python / torch / cuda 版本

**教训**：**可复现 = seed + 配置 + 环境**。没有这三样，实验等于白跑。

---

### 第 8 步：测试

**目标**：pytest 冒烟测试

**知识点**：
- `conftest.py` 共享 fixture
- `dummy_batch` fixture
- 只测**形状**和**接口**，不测精度
- `pytest.ini_options.pythonpath = ["src"]` 解决 import
- `@pytest.mark.skip` 跳过需要真实数据的测试

**教训**：测试的意义是**改动不怕**。

---

### 第 9 步：调度 + 早停 + 恢复

**目标**：`StepLR` / `CosineAnnealingLR` + early stopping + resume

**知识点**：
- `scheduler.step()` 放在 epoch 末（PyTorch 官方推荐）
- early stopping：patience 个 epoch 没提升就停
- **恢复训练保存完整状态**：model + optimizer + scheduler + epoch + best_acc
- resume 后 CSV 追加到旧文件（`Path(resume_path).stem` 反推）

**教训**：
- 断点续训是长期训练的必备功能
- **CSV 要连续，日志要独立** —— 这是 resume 的正确哲学

---

### 第 10 步：加 ViT

**目标**：注册表加一行就行

**知识点**：
- `PatchEmbedding`：用 `nn.Conv2d` 做 patch 切分
- `[CLS]` token + 位置编码
- `nn.TransformerEncoderLayer` + `nn.TransformerEncoder`
- `norm_first=True` 对应 Pre-LN（ViT 原版）
- 小 ViT：`embed_dim=192, depth=6, heads=6, patch=4`

**教训**：**注册表模式的威力**：加模型只需改两处（模型文件 + 注册表一行）。

---

### 第 11 步：项目外观

**目标**：`pyproject.toml` + `README.md` + `.gitignore`

**知识点**：
- `pyproject.toml`：依赖 + 工具配置 + 构建配置
- **`[dependency-groups]`**（PEP 735，uv 推荐）vs `[project.optional-dependencies]`（pip 传统）
- `.gitignore` 只对未追踪文件生效，已追踪的要 `git rm --cached`
- **Python 版本必须全对齐**：`requires-python` / `ruff target-version` / `mypy python_version` / CI / Docker
- `uv.lock` **要提交**（应用项目）
- `.idea/` / `__pycache__/` / `*.egg-info/` 要排除

**教训**：
- **uv 项目不要用 pip**，`uv sync` 一步到位
- 项目的外观决定别人 5 分钟能不能看懂

---

### 第 12 步：Makefile

**目标**：常用命令集中

**知识点**：
- `?=` 默认值
- `$(if $(VAR),then,else)` 条件参数
- `.PHONY` 声明伪目标
- **缩进必须是 Tab**
- `uv run` 保证环境一致

**教训**：Makefile 是**项目级快捷键**。

---

### 第 13 步：暴露公共 API

**目标**：`import src` 就能拿到模型

**知识点**：
- `src/__init__.py` 定义 `__all__` 和 `__version__`
- `py.typed` 空文件，告诉 mypy 有类型
- 版本号与 `pyproject.toml` 对齐

**教训**：**公开 API 是包的契约**。改 `__all__` = 改契约。

---

### 第 14 步：数据增强

**目标**：训练集增强，验证/测试集不增强

**知识点**：
- 训练：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`
- 验证/测试：只 `ToTensor()` + `Normalize()`
- 同一个 CIFAR-10 训练集，**用不同 transform 产生 train_set 和 val_set**
- 数据增强让 loss 更抖，但泛化更好

**效果**：
- deep_convnet 从 70.08% → **73.82%**（+3.7 个点）

**教训**：**数据 > 模型 > 超参**。数据增强是最高性价比的改进。

---

### 第 15 步：混合精度 AMP

**目标**：GPU 训练 2x 加速

**知识点**：
- `torch.cuda.amp.autocast`：自动选 float16 / float32
- `GradScaler`：梯度放大防 underflow
- 三步走：`scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`
- **只在 CUDA 上生效**，MPS / CPU 自动退化
- `amp_enabled = use_amp and device.type == "cuda"`

**教训**：AMP 是 GPU 训练的标配。但**先让代码正确，再考虑速度**。

---

## 五、企业级项目必备的 10 个点

| # | 点 | 一句话 |
|---|---|---|
| 1 | **关注点分离** | 每个模块只做一件事 |
| 2 | **配置管理** | 所有超参集中 |
| 3 | **可复现性** | seed + config + env |
| 4 | **日志与监控** | `logging` 替代 `print` |
| 5 | **Checkpoint 管理** | model + optimizer + scheduler |
| 6 | **测试** | 改动不怕 |
| 7 | **注册表模式** | 加东西不改老代码 |
| 8 | **命令行接口** | 不改代码跑不同实验 |
| 9 | **环境快照** | 追溯历史 |
| 10 | **项目结构约定** | 新人 5 分钟看懂 |

---

## 六、还需要学习的内容

### 已完成（里程碑 1）

```text
✅ 项目骨架
✅ 配置管理
✅ 训练循环
✅ 日志系统
✅ 可复现性
✅ 测试
✅ 注册表模式
✅ 数据增强
✅ 混合精度
```

### 待学习（里程碑 2）

**DevOps 阶段**：

- CI/CD（Jenkins / GitHub Actions）
- Docker
- 分布式训练（DDP）
- 模型导出（ONNX / TorchScript）
- 推理服务（FastAPI）

**工程质量阶段**：

- ruff / mypy / pre-commit
- 类型注解
- 代码审查

**可观测阶段**：

- TensorBoard / WandB
- MLflow
- 实时监控

**算法进阶**：

- 学习率预热（warmup）
- MixUp / CutMix
- RandAugment
- 知识蒸馏
- 量化

**架构进阶**：

- Hydra / OmegaConf（层级配置）
- Lightning（训练框架）
- Registry 装饰器自动注册
- 插件系统

---

## 七、项目哲学

### 1. 先跑，再对，再快

- 第 1 步让它跑
- 第 2-9 步让它对
- 第 14-15 步让它快

### 2. Rule of Three（三次法则）

> 同一件事做第三次的时候，才值得抽象。

不要为了"将来可能用到"去抽象。等用到第三次，自然知道该怎么抽象。

### 3. 配置 vs 硬编码

**任何会变的东西都应该参数化**。学习率、batch size、seed、数据路径、模型结构……

### 4. 可复现 > 精度

准确率 99% 但复现不了的实验，是没用的实验。

### 5. 模块间通过参数通信，不通过全局变量

```python
# ❌
def train():
    for x, y in loader_train:  # 全局

# ✅
def train(model, optimizer, loader_train, loader_val, ...):
```

### 6. 教学代码 ≠ 企业代码

CS231N 的 `functional.py`、手写梯度更新、`random_weight` 是**理解原理**用的。企业用 `nn.Module` + `optim` + `nn.init`。

---

## 八、一句话总结

> **企业级项目的本质不是用了多少高级工具，而是：任何一个变化，都只需要改一个地方。**

- 改超参 → `config.py`
- 加模型 → `models/xxx.py` + 注册表一行
- 加数据集 → `data/xxx.py` + 注册表一行
- 换训练策略 → `training/train.py`
- 换日志后端 → `utils/logger.py`

这就是从 CS231N 作业变成企业级项目的**唯一标准**。

---

## 九、下一步

**里程碑 2** 按优先级：

1. TensorBoard 可视化
2. ruff + mypy + pre-commit（代码质量）
3. Docker
4. CI
5. 模型导出
6. 分布式训练

**方向选择**：

- **A. 继续视觉线**：ResNet、EfficientNet、目标检测、语义分割
- **B. 换任务线**：NLP、Transformer 语言模型、多模态

**不管选哪条，现在的骨架都能复用** —— 这就是这 15 步最大的价值。

---

*Last updated: 2026-09-15*