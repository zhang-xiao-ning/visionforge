# Memo 4: 工程质量 + 类型安全

> 从"本地环境成型"到"重构不怕"
> 时间：2026-09-15
> 版本：v0.1.0

---

## 一、本阶段做了什么（第 20-22 步）

### 第 20 步：类型注解补齐（B1）

- 打开 `disallow_untyped_defs = true`
- 30 个"缺少类型注解"错误 → 全部修完
- `mypy src/` → `Success: no issues found in 20 source files`

**改动覆盖：**

| 文件 | 函数数 |
|---|---|
| `src/data/transforms.py` | 2 |
| `src/data/cifar10.py` | 1 |
| `src/data/datasets.py` | 1 |
| `src/utils/logger.py` | 3 |
| `src/utils/seed.py` | 1 |
| `src/utils/env.py` | 2 |
| `src/training/evaluator.py` | 1 |
| `src/training/train.py` | 2 |
| `src/models/*.py` | 各 2 |
| `src/main.py` | 5 |

**关键类型：**

```python
def train(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    loader_train: DataLoader,
    loader_val: DataLoader,
    epochs: int = 1,
    scheduler: lr_scheduler.LRScheduler | None = None,
    ...
    writer: SummaryWriter | None = None,
) -> dict[str, float | int]:
```

**知识点：**

- `disallow_untyped_defs` 让 mypy 强制所有函数有类型
- `self` 不用标注
- 无返回值 → `-> None`
- 可选 → `X | None = None`（Python 3.10+）
- 容器 → `dict[str, int]` / `tuple[A, B, C]`

**学到的：**

- **严格模式和宽松模式的差距是 30 个错误**
- 类型注解的价值：**重构不怕**，IDE 补全更准
- 一次性补齐比每天加一点更高效

---

### 第 21 步：补单元测试（B3）

新增三个测试文件：

| 文件 | 测试内容 |
|---|---|
| `tests/test_evaluator.py` | `evaluate` 返回 float、范围、恢复 train 模式 |
| `tests/test_trainer.py` | 训练 1 epoch、权重更新、scheduler 衰减、early stopping |
| `tests/test_logger.py` | CSV 表头、写入、append 模式、无 lr |

**总量：**

```text
22 passed, 1 skipped
```

**知识点：**

- `tmp_path` fixture：pytest 自动给的临时目录
- `conftest.py` 里的共享 fixture
- 测试用 `_DummyClassifier`：一个极简的 `flatten + linear` 模型
- 断言要**具体**：`assert not torch.equal(w_before, w_after)`
- 跳过需要真实数据的测试：`@pytest.mark.skip`

**学到的：**

- 测试的核心不是"覆盖率"，而是**保护关键逻辑不被改坏**
- 每个测试只测一件事

---

### 第 22 步：测试与硬件解耦

发现的问题：Mac 用 MPS，家里用 CUDA，测试里 `evaluate` 把数据搬到 `device`，但模型留在 CPU，导致 `Placeholder storage has not been allocated on MPS device`。

**方案：**

1. `src/config.py`：`USE_GPU` 从环境变量读

   ```python
   USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
   ```

2. `tests/conftest.py`：在 import 前强制 CPU

   ```python
   import os
   os.environ["USE_GPU"] = "false"
   ```

**效果：**

| 场景 | device |
|---|---|
| Mac 测试 | CPU |
| 4070 测试 | CPU |
| Mac 训练 | MPS |
| 4070 训练 | CUDA |
| 纯 CPU 机器训练 | CPU（自动 fallback） |

**知识点：**

- 环境变量控制模块行为
- **必须在 import 之前设置**（import 是执行一次的）
- 测试和硬件彻底解耦，跨平台一致

**学到的：**

- **测试用 CPU，训练用 GPU** —— 这是最佳实践
- 测试慢会让人不想跑测试

---

### 第 23 步：README 完善（B2）

重写 `README.md`，新增：

- 一句话介绍
- 环境要求 + 安装
- 数据集准备
- 训练命令 + resume
- 完整 CLI 参数表
- 环境变量表（`USE_GPU`）
- 实验列表（4 个模型）
- 产出文件说明
- 完整目录结构
- 开发命令（Makefile）
- pre-commit 说明
- **如何加新模型**（3 步）
- **如何加新数据集**（3 步）
- 测试说明
- 可复现性说明

**学到的：**

- README 是**项目说明书**，不是"文档"
- 新人看 README 能 5 分钟上手 = README 合格
- "如何加新模型"这一节最关键，直接体现**架构可扩展性**

---

## 二、本阶段核心收获

### 1. 严格类型检查的价值

| 之前 | 之后 |
|---|---|
| 类型错误在运行时才暴露 | 写代码时 IDE 就提示 |
| 重构时心惊胆战 | `mypy` 告诉你哪里坏了 |
| 参数类型靠猜 | 签名即文档 |

**开启严格模式后，30 个错误一次性修完，比每天修 1 个有效率。**

### 2. 测试的设计原则

- **快**：22 个测试 2 秒跑完
- **稳**：CPU-only，跨平台一致
- **准**：只测关键逻辑，不测实现细节
- **独立**：每个测试不依赖其他测试

### 3. 设备解耦的工程意义

```python
# 之前
if torch.cuda.is_available():
    device = cuda

# 之后
USE_GPU = os.environ.get("USE_GPU", "true") == "true"
```

**从"自动检测"到"环境变量 + 自动检测"**，多了灵活性：

- 本地调试：`USE_GPU=false`
- 生产训练：默认自动
- 测试：代码里强制 false

### 4. README 的核心价值

不是"项目介绍"，而是：

> **让一个陌生人 5 分钟内能跑起来、改起来。**

关键章节：

1. 安装（`uv sync`）
2. 训练（`make train ...`）
3. **如何加新模型**（3 步）
4. **如何加新数据集**（3 步）

---

## 三、项目现状

### 质量维度

| 维度 | 工具 | 状态 |
|---|---|---|
| 代码风格 | ruff | ✅ 强制 |
| 类型安全 | mypy（严格） | ✅ 强制 |
| 提交规范 | pre-commit | ✅ 拦截 |
| 测试 | pytest | ✅ 22 passed |
| 可观测 | TensorBoard | ✅ 集成 |
| 可复现 | seed + snapshot | ✅ |
| 文档 | README | ✅ 完整 |

### 测试覆盖

```text
22 passed, 1 skipped
```

| 文件 | 测试数 |
|---|---|
| test_models.py | 5 |
| test_trainer.py | 5 |
| test_logger.py | 4 |
| test_evaluator.py | 3 |
| test_transforms.py | 3 |
| test_data.py | 2 (+1 skipped) |

### 目录结构

```text
my_test_project/
├── src/
│   ├── __init__.py
│   ├── py.typed
│   ├── config.py
│   ├── main.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
├── README.md
├── TODO.md
├── COMPLETED.md
├── memo-1.md ~ memo-4.md
├── .gitignore
├── pack.sh
└── dump_code.sh
```

---

## 四、下一步

### 已完成里程碑

| # | 里程碑 | 状态 |
|---|---|---|
| 1 | 核心骨架 | ✅（1-10） |
| 2 | 项目外观 | ✅（11-13） |
| 3 | 训练增强 | ✅（14-15） |
| 4 | 可观测 | ✅（16） |
| 5 | 本地开发环境 | ✅（17-19） |
| 6 | 工程质量 | ✅（20-23） |

### 待做

**DevOps 方向：**

- CI（Jenkins / Gitee Go）
- Docker
- 分布式训练 DDP
- 模型导出 ONNX
- FastAPI 推理

**算法方向：**

- ResNet / EfficientNet
- 换数据集
- 学习率 warmup
- MixUp / CutMix

**工程质量剩余：**

- 版本号单一来源（`importlib.metadata`）
- `make docker` 等新命令

---

## 五、一句话

> **本阶段让项目从"能安心开发"变成"能安心重构"。**

关键区别：

- 之前：改代码怕破坏逻辑，靠人工 review
- 现在：**mypy 检查类型 + pytest 保护逻辑 + pre-commit 拦截**

**重构时只要 `make test` 全过，就放心。**

---

## 六、附录：本阶段遇到的坑

### 坑 1：`disallow_untyped_defs` 一开报 30 个错

**症状**：打开开关后 `mypy` 满屏 `missing type annotation`。

**解决**：逐个文件加类型。文件顺序按"从易到难"：`transforms` → `cifar10` → `datasets` → ... → `main`。

### 坑 2：MPS device 报错

**症状**：`RuntimeError: Placeholder storage has not been allocated on MPS device!`

**原因**：模型在 CPU，数据被 `evaluate` 搬到 MPS。

**解决**：环境变量 + conftest 强制 CPU。

### 坑 3：`from config import device` 是值拷贝

**症状**：`monkeypatch.setattr(config, "device", ...)` 不生效。

**原因**：`from config import device` 导入的是 `device` 值，不是引用。

**解决**：改用环境变量。

### 坑 4：`git diff --stat` 停在 `:` 不动

**原因**：git 用 `less` 分页。

**解决**：按 `q` 退出。

### 坑 5：zsh 不认 `#` 注释

**症状**：`command not found: #`。

**解决**：一行行敲，或 `setopt interactive_comments`。

---

*Last updated: 2026-09-15*