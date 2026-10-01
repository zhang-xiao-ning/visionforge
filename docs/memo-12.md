# Memo 12：训练范式接口

> 让 LLaVA 的"冻结 / 多阶段 / 预训练加载"三个需求成为声明，而不是改代码
> 时间：2026-10-01
> 版本：v0.1.1

---

## 一、本阶段做了什么（第 41-44 步）

### 第 41 步：修 bug + `USE_CUDA` 抽取

**代码审查发现的三个 bug**：

| # | 位置 | 症状 |
|---|---|---|
| B1 | `tests/integration/test_regression.py` | 传了不存在的参数 `dataset_name`，`RUN_REGRESSION=1` 时 `TypeError`，CI skip 掉所以没暴露 |
| B2 | `src/models/captioning.py` | `forward` 里 `tgt_mask = self._causal_mask(...)` 写了两次 |
| B3 | `src/data/flickr8k.py` | `build_bundle` 没传 `num_workers`，GPU 下 DataLoader 单进程 |

**顺带抽取常量**：

```python
# src/runtime.py
device = get_device()
USE_CUDA = device.type == "cuda"
```

`cifar10.py` 和 `flickr8k.py` 里的 `use_cuda = device.type == "cuda"` 统一改为 `USE_CUDA`。

**关键决策**：`USE_CUDA` 语义是"我们打算用 CUDA"，**MPS 不算**。Mac 上 DataLoader 多进程有已知问题，保持单进程。

---

### 第 42 步：`Model` 基类

**产物**：`src/models/base.py`

```python
class Model(nn.Module):
    @classmethod
    def from_data(cls, bundle): return cls(**bundle.model_init)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"all": list(self.parameters())}

    def initialize(self, checkpoint: Path) -> None:
        pass
```

**三个默认方法**：

| 方法 | 默认 | 覆盖时机 |
|---|---|---|
| `from_data` | `cls(**bundle.model_init)` | 需要 `bundle.extras` 时 |
| `param_groups` | 单组 `{"all": [...]}` | LoRA / 冻结 / 多组 lr |
| `initialize` | noop | 加载预训练权重 |

**`ViT` / `CaptioningModel` 的改动**：

- 继承从 `nn.Module` 改为 `Model`
- 删掉各自的 `from_data`（现在继承默认）
- 删掉文件顶部 `if TYPE_CHECKING: from data.bundle import DataBundle`

**契约测试**：`tests/contracts/test_model_contract.py`（5 个测试）

- `issubclass(model_cls, Model)` 遍历所有 experiment
- `param_groups()` 默认覆盖所有参数
- `initialize()` 默认 noop 且不改变权重

**关键决策**：**ABC + 默认实现**（选项 B），不是 ABC 强制（A），也不是鸭子类型（C）。和 `TrainHooks` 的"默认 noop"哲学一致。

---

### 第 43 步：`Task.stages` + `Stage`

**产物**：`src/tasks/base.py` 加 `Stage` dataclass，`Task` 加类属性 `stages`

```python
@dataclass
class Stage:
    name: str
    epochs: int
    freeze: list[str] = field(default_factory=list)

class Task(ABC):
    primary_metric: str = "loss"
    higher_is_better: bool = True
    stages: list[Stage] | None = None   # None = 单阶段
```

**`freeze` 里的名字**必须匹配 `Model.param_groups()` 的 key。

**向后兼容**：`stages = None` 表示"用 `TrainConfig.epochs`，走旧逻辑"。`ClassificationTask` / `CaptioningTask` 不改，自动是 `None`。

**契约测试**：`tests/unit/test_task_stages.py`（4 个测试）

- `Stage` 默认值
- `freeze` 默认不共享（`field(default_factory=list)` 正确）
- 两个 Task 的 `stages` 默认都是 `None`

---

### 第 44 步：Runner 消费 stages

**核心改动**：`ExperimentRunner.run()` 分派

```python
def run(self):
    self._log_header()
    if self.task.stages is None:
        result = self._run_single_stage()   # 原逻辑，行为零变化
    else:
        result = self._run_multi_stage()    # 新增路径
    self._tracker.restore(self.model)
    test_acc = self._evaluate_test()
    self._save_checkpoint(result)
    return {"test_acc": test_acc, **result}
```

**多阶段流程**：

```text
for stage in task.stages:
    1. _apply_freeze(stage.freeze)         # 设 requires_grad
    2. self.optimizer = _build_optimizer() # 只含可训练参数
    3. _build_scheduler(..., stage.epochs) # 每 stage 重建
    4. self._tracker = _build_tracker()    # 每 stage 重建
    5. hooks = self._build_hooks()
    6. train(..., stage_cfg, start_epoch=1)
```

**关键决策**（P6 讨论已定）：

| # | 决策 |
|---|---|
| Q2 | `param_groups()` 返回**所有**命名组，由 stage 决定冻谁 |
| Q3 | `initialize()` 由 **application 调用**（不是 framework / runner） |
| Q4 | 多阶段循环在 **Runner**，`train()` 只跑 N epoch |
| Q5 | 每 stage **重建 optimizer**（不同 stage 训不同参数，state 无复用） |
| Q6 | `param_groups()` **不**带 lr（lr 是配置的事） |

**`_apply_freeze`**：

```python
def _apply_freeze(self, freeze: list[str]) -> None:
    model = _unwrap_model(self.model)
    groups = model.param_groups()
    unknown = set(freeze) - set(groups.keys())
    if unknown:
        raise ValueError(f"Unknown freeze groups: {sorted(unknown)}. Available: {sorted(groups.keys())}")
    for name, params in groups.items():
        should_train = name not in freeze
        for p in params:
            p.requires_grad = should_train
```

**`_build_optimizer` 改动**：从 `self.model.parameters()` 改为 `[p for p in self.model.parameters() if p.requires_grad]`。单阶段下等价（所有参数都可训）。

**多阶段不支持 resume**：明确报 `NotImplementedError`。理由——每 stage 是 fresh start，resume 单个 stage 的语义未定。

**`cfg.epochs` 在多阶段下被忽略**：用 `dataclasses.replace(cfg, epochs=stage.epochs)` 派生每 stage 的 config。

**契约测试**：`tests/unit/test_runner_multi_stage.py`（6 个测试）

- 多阶段跑完所有 stage
- `cfg.epochs=999` 不泄漏到多阶段
- 多阶段 resume 抛 `NotImplementedError`
- `_apply_freeze` 未知组名抛 `ValueError`
- `_apply_freeze` 正确设置 `requires_grad`
- `_apply_freeze([])` 全部解冻

---

## 二、核心收获

### 1. 三个接口的对称性

阶段 14 后，`Model` / `Task` / `Tokenzier` 三个抽象都是同一模式：

```text
接口 + 默认实现 → 子类只覆盖需要的部分
```

| 抽象 | 基类形态 | 默认实现 |
|---|---|---|
| `Tokenizer` | `Protocol` + `@runtime_checkable` | 无（每个方法都要写） |
| `Task` | `ABC` + `@abstractmethod` | `primary_metric` / `higher_is_better` / `stages` |
| `Model` | `nn.Module` 子类 | `from_data` / `param_groups` / `initialize` |

**`Tokenizer` 走 Protocol 是因为它没有"默认行为"**——每个 tokenizer 的 encode/decode 完全不同。

**`Task` / `Model` 走 ABC + 默认是因为它们有明确的通用默认**——绝大多数 Task 走 CE loss + accuracy/perplexity，绝大多数 Model 全参数训练。

**判据**："默认实现是否对 80% 的子类正确？"

- 是 → 默认实现（`Task` / `Model`）
- 否 → Protocol，每个子类都写（`Tokenizer`）

### 2. "接口早，实现晚"的落地

`param_groups` / `stages` / `initialize` 三个接口在第 42-43 步立好，**没有任何具体实现**。`ViT` / `CaptioningModel` 都用默认。

LLaVA 接入时：

- `param_groups` 覆盖为 4 组
- `stages` 覆盖为 2 阶段
- `initialize` 覆盖为加载 CLIP + Llama

**接口不改，只加实现**。这就是"接口早做"的收益。

### 3. 多阶段实现的三个独立决策

多阶段是**三个决策的乘积**：

| 决策 | 选择 |
|---|---|
| 循环在哪 | Runner |
| optimizer 复用吗 | 重建 |
| tracker 复用吗 | 重建 |

每个决策都独立可改。选"重建"的理由是**每 stage 训不同参数集**——state 无复用价值，重建干净。

---

## 三、踩的坑

### 坑 1：改继承时漏改 class 声明

第 42 步，我给的指令里说"改 import + 继承"，容易被理解为"检查 import 有没有加"，但**真正要改的是 `class ViT(nn.Module):` → `class ViT(Model):`**。

**症状**：`issubclass(ViT, Model)` 返回 `False`，且 `ViT` 没有继承来的 `param_groups` / `initialize` / `from_data`。

**教训**：单行修改（改 class 声明）必须显式列出"改之前 → 改之后"。

### 坑 2：`monkeypatch.setitem` 不接收字符串路径

第 44 步，测试里写了：

```python
monkeypatch.setitem("experiment.runner.EXPERIMENTS", "fake_exp", fake_exp)
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 字符串
```

`monkeypatch.setitem` 接收 **dict 对象**，不是路径。路径只对 `setattr` 有效。

**修法**：

```python
from registry import EXPERIMENTS
monkeypatch.setitem(EXPERIMENTS, "fake_exp", fake_exp)
```

**教训**：`monkeypatch.setattr` 和 `monkeypatch.setitem` API 不对称。

### 坑 3：`Task` 没有默认 `from_data`

第 44 步的测试 `_MultiStageTask` 忘了写 `from_data`，导致 `TypeError: type object '_MultiStageTask' has no attribute 'from_data'`。

**根本原因**：`Model` 有默认 `from_data`（第 42 步加的），但 `Task` 没有。**不对称**。

`ClassificationTask` / `CaptioningTask` 都手写了 `from_data`，所以没暴露。但每个新 Task 都要写这段样板代码。

**待做**：给 `Task` 加默认 `from_data`（`return cls()`），对齐 `Model`。见"下一步"。

---

## 四、现在架构

```text
src/
├── models/
│   ├── base.py              ← 新增：Model 基类
│   ├── vit.py               ← 继承 Model，删 from_data
│   └── captioning.py        ← 继承 Model，删 from_data
├── tasks/
│   └── base.py              ← 加 Stage dataclass + Task.stages
├── runtime.py               ← 加 USE_CUDA
├── data/
│   ├── cifar10.py           ← 用 USE_CUDA
│   └── flickr8k.py          ← 用 USE_CUDA + num_workers
└── experiment/
    └── runner.py            ← run() 分派单/多阶段；新增 _run_multi_stage / _apply_freeze / _build_tracker
```

**扩展性提升**：

| 维度 | 之前 | 之后 | 提升 |
|---|---|---|---|
| 训练范式扩展性 | 4/10 | **8/10** | +4 |
| 多阶段 | 写死在 Task 代码里 | 声明式 `Task.stages` | — |
| 冻结 | 无 | 声明式 `Stage.freeze` | — |
| 预训练加载 | 无接口 | `Model.initialize()` | — |
| 参数分组 | 无 | `Model.param_groups()` | — |

**"加东西"成本**：

| 动作 | 改动文件数 |
|---|---|
| 加新模型（继承 Model） | 2（`models/xxx.py` + `registry.py`） |
| 加新 task | 2（`tasks/xxx.py` + `registry.py`） |
| 加 LLaVA 冻结 | 0（覆盖 `param_groups` + YAML 声明 freeze） |
| 加 LLaVA 多阶段 | 0（覆盖 `stages`） |
| 加 FSDP | 1（`strategy.py` 加子类） |

**测试**：

```text
123 passed, 5 deselected
```

新增 15 个测试：
- `test_model_contract.py`：5 个（契约）
- `test_task_stages.py`：4 个（Stage 语义）
- `test_runner_multi_stage.py`：6 个（多阶段行为）

**Commit 序列**：

```text
87cb99e feat: add Model base class with default from_data/param_groups/initialize
f66eed7 feat: add Task.stages and Stage dataclass
e6b8314 feat: multi-stage training in ExperimentRunner
```

---

## 五、下一步

### 短期补丁（半天）

**补 `Task.from_data` 默认实现**，对齐 `Model`：

```python
class Task(ABC):
    @classmethod
    def from_data(cls, bundle: "DataBundle") -> Self:
        return cls()
```

然后 `ClassificationTask` / `CaptioningTask` 删掉各自的 `from_data`。

**触发条件**：LLaVA 的 `CaptioningTaskLLaVA` 接入前，避免样板代码。

### 阶段 15：评估层（半天）

**目标**：Captioning 从"只有 perplexity"到"BLEU / CIDEr / METEOR / ROUGE"。

**架构应对**：

- `Task.eval_step` 已返回 `dict[str, float]`，结构上支持
- `primary_metric` 是类属性——可能要改成从配置声明
- 依赖：`pycocoevalcap` 或 `nltk`（在 `tests/` 或 `scripts/`，不进 `src/`）

### 阶段 16：LLaVA-A 推理（1 天）

**目标**：脚本级加载现有 LLaVA 权重，跑通对话。

**约束**：

- `transformers` 只出现在 `scripts/`，不进 `src/`
- **不改 framework**
- 4bit 量化（4070 12G 显存）

### 阶段 17：LLaVA-B LoRA 微调（2-3 天）

**目标**：冻结 + LoRA。**这一步会真正撞到 `DataBundle` 强类型化问题。**

### 阶段 18-19：YAML workflow + LLaVA-C 从 0 训

（见 architecture-notes）

---

## 六、一句话

> **本阶段把 LLaVA 的三个需求（冻结 / 多阶段 / 预训练加载）从"接入时改 framework"变成"接入时声明"。**

- `Model.param_groups()` → 声明"我有几组参数"
- `Task.stages` → 声明"我要分几阶段训"
- `Model.initialize()` → 声明"我从哪里加载权重"

**框架不动，只加声明。这就是"接口早，实现晚"的收益。**

---

*Last updated: 2026-10-01*

---