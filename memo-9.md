# Memo 9: 工程打磨（测试金字塔）

> 从"能测试"到"能防崩"
> 时间：2026-09-23
> 版本：v0.1.1

---

## 一、本阶段做了什么（第 29-31 步）

### 第 29 步：版本号单一来源

**问题**：两处版本号手动同步

- `pyproject.toml`：`version = "0.1.1"`
- `src/__init__.py`：`__version__ = "0.1.1"`

**改法**：`__init__.py` 从 `importlib.metadata` 读

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("visionforge")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
```

**收益**：只改 `pyproject.toml`，`__version__` 自动同步。

**验证**：
```bash
uv run python -c "import src; print(src.__version__)"
# 0.1.1
```

---

### 第 30 步：`num_train` 参数 + 集成测试

**目标**：加一个真正有用的功能，同时作为集成测试的基础。

**功能**：从 CLI 限定训练样本数

```bash
uv run python src/main.py --experiment mlp --epochs 1 --num-train 1000
```

**改动**：
- `ExperimentRunner.__init__` 加 `num_train` 参数
- `runner._build_loaders` 透传给 `build_loaders`
- `main.py::parse_args` 加 `--num-train`
- README 加参数说明

**集成测试**（`tests/test_integration.py`）：

- `test_train_one_epoch_on_real_data`：128 张图跑 1 epoch
- `test_runner_saves_and_resumes`：跑 + 恢复训练
- **用 `RUN_INTEGRATION=1` 控制**，CI 上 skip
- `num_train=128` 让测试秒级完成

**踩坑**：
- `ExperimentRunner.__init__` 忘了 `self.num_train = num_train`
- 测试跑成 60 秒（49000 张图）

---

### 第 31 步：回归测试

**目标**：重构后，精度不能崩（防大错，不防微调）。

**设计**：

```python
BASELINES = [
    ("mlp", 1, 1000, 0.15),
    ("deep_convnet", 1, 1000, 0.15),
]
```

**阈值故意宽松**：抓"崩了"（0.36 → 0.10），不抓"微调"（0.36 → 0.34）。环境差异（MPS / CUDA / CPU）不应触发。

**用 `RUN_REGRESSION=1` 控制**，CI 上 skip。

**踩坑**：测试用了 `TrainConfig` 默认 lr（1e-2），而不是 `EXPERIMENTS` 的（`deep_convnet` 是 0.1）。

**修法**：从 `registry.EXPERIMENTS` 查默认 lr。

---

## 二、测试金字塔

| 层级 | 数量 | 何时跑 | 时间 |
|---|---|---|---|
| 单元测试 | 41 | 每次 commit + CI | 3 秒 |
| 集成测试 | 2 | `make test-integration` | 1 分钟 |
| 回归测试 | 2 | `make test-regression` | 50 秒 |

**金字塔原则**：

- 底层多、快、常跑
- 上层少、慢、手动跑

---

## 三、Makefile 命令

```bash
make test               # 单元测试
make test-integration   # 集成测试
make test-regression    # 回归测试
make ci                 # 完整 CI（单元测试 + lint + mypy）
```

---

## 四、核心收获

### 1. 单一来源原则

- 版本号只在 `pyproject.toml` 定义
- 其它地方**读**，不**写**

**任何"两处同步"的东西，早晚不一致。**

### 2. 测试的层次

| 层 | 测什么 | 保护什么 |
|---|---|---|
| 单元 | 组件逻辑 | 单文件改动 |
| 集成 | 端到端流程 | 跨模块接口 |
| 回归 | 精度不崩 | 重构安全性 |

**三层缺一不可**。

### 3. 阈值设计哲学

**回归测试的阈值是"承诺"，不是"现状快照"。**

- 定太严 → 环境差异就红，浪费时间
- 定太松 → 抓不住真崩

**推荐**：基线 × 0.5 左右（比如 0.36 → 0.15）。

### 4. 测试写法 = 生产写法的镜像

**回归测试踩的坑**：测试用了 `TrainConfig()` 默认 lr，而真实运行用 `EXPERIMENTS` 查表。

**教训**：测试调用的接口应该和生产**完全一致**，否则测的不是真东西。

---

## 五、项目现状

### 里程碑

| # | 里程碑 | 状态 |
|---|---|---|
| 1-8 | 骨架 / 外观 / 增强 / 可观测 / 本地环境 / 工程质量 / DevOps / CI | ✅ |
| 9 | 训练镜像 | ✅ |
| 10 | 重构 + DDP | ✅ |
| 11 | **工程打磨** | ✅（今天） |

### 测试

```text
41 passed, 5 skipped (CI)
2 passed (make test-integration)
2 passed (make test-regression)
```

### 质量维度

| 维度 | 工具 | 状态 |
|---|---|---|
| 代码风格 | ruff | ✅ |
| 类型安全 | mypy (strict) | ✅ |
| 提交规范 | pre-commit | ✅ |
| 单元测试 | pytest (41) | ✅ |
| 集成测试 | pytest (2) | ✅ |
| 回归测试 | pytest (2) | ✅ |
| 远端 CI | GitHub Actions (40s) | ✅ |
| 版本号 | `importlib.metadata` | ✅ 单一来源 |

---

## 六、下一步

### 方向

- **A. 多模态启动**：Captioning / CLIP
- **B. 图像算法**：ResNet / warmup / MixUp
- **C. 工程二期**：CHANGELOG / 覆盖率 / mkdocs
- **D. 收工**

### 判断

- 工程已经**足够扎实**，继续做 C 边际效益递减
- **A 是原始目标**，架构已经支持
- B 是"锦上添花"

**推荐 A。**

---

## 七、一句话

> **本阶段把"能测"变成"知道什么时候会崩"。**

- 单元测试 → 组件不坏
- 集成测试 → 流程能跑
- 回归测试 → 精度不崩

**重构时的底气，来自这三层防线。**

---

*Last updated: 2026-09-23*