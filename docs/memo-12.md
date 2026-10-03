# Memo 12：训练范式接口 + 评估层解耦

> 把 LLaVA 的三个需求（冻结/多阶段/预训练加载）从"接入时改代码"变成"接入时声明"；把评估从 Task 剥离为独立一等公民
> 时间：2026-10-01 ~ 2026-10-03
> 版本：v0.1.1

---

## 一、本阶段做了什么（第 41-57 步）

### 阶段 14：训练范式接口（41-50）

#### 第 41 步：修 bug + `USE_CUDA`

**代码审查发现 3 个 bug**：

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

`cifar10.py` / `flickr8k.py` 统一用 `USE_CUDA`。语义是"我们打算用 CUDA"，**MPS 不算**——Mac 上 DataLoader 多进程有已知问题。

#### 第 42 步：`Model` 基类

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

**`ViT` / `CaptioningModel`**：继承 `Model`，删各自的 `from_data`。

**契约测试**：`tests/contracts/test_model_contract.py`（5 个测试）。

#### 第 43 步：`Task.stages` + `Stage`（后被重构）

**产物**：`Task` 加类属性 `stages: list[Stage] | None = None`。

**后被重构**：`Stage` 从 `Task` 移到 `src/experiment/spec.py`，理由——训练范式是 experiment 的配置，不是 task 的代码。

#### 第 44 步：Runner 多阶段（后被重构）

**产物**：`ExperimentRunner.run()` 分派单/多阶段。

**关键决策**（保留）：

| # | 决策 |
|---|---|
| Q2 | `param_groups()` 返回**所有**命名组，由 stage 决定冻谁 |
| Q3 | `initialize()` 由 **application 调用**（不是 framework / runner） |
| Q4 | 多阶段循环在 **Runner**，`train()` 只跑 N epoch |
| Q5 | 每 stage **重建 optimizer**（不同 stage 训不同参数，state 无复用） |
| Q6 | `param_groups()` **不**带 lr（lr 是配置的事） |

**后被重构**：stages 从 `Task` 移到 `Experiment` + Runner 构造参数。

#### 第 45 步：`Task.from_data` 默认

**对齐 `Model`**：

```python
class Task(ABC):
    @classmethod
    def from_data(cls, bundle: "DataBundle") -> Self:
        return cls()
```

`ClassificationTask` / `CaptioningTask` 删掉各自的 `from_data`。

**测试**：`test_task_from_data_inherited` 用 `"from_data" not in cls.__dict__` 检查——**classmethod 不能用 `is` 比较**（每次访问返回新的 bound method）。

#### 第 46-47 步：`spec` 重构

**新建 `src/experiment/spec.py`**，定义三个 dataclass：

```python
@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 64
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    # ... 共 16 个字段
    # 移除：experiment / seed / amp

@dataclass
class Stage:
    name: str
    freeze: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)

@dataclass
class Experiment:
    model: type[nn.Module]
    task: type[Task]
    data: str
    metrics: list[Metric]
    primary_metric: str
    seed: int = 42
    amp: bool = False
    config: TrainConfig = field(default_factory=TrainConfig)
    category: str = ""
```

**`registry.py`**：dict → `Experiment` 对象。

**删 `src/experiment/config.py`**（搬到 spec.py）。

**`ExperimentRunner` 签名变化**：

```python
ExperimentRunner(
    experiment_name: str,
    config: TrainConfig,
    stages: list[Stage] | None = None,
    amp: bool = False,
    ...
)
```

**`batch_size` 从 Runner 参数移入 `TrainConfig`。**

#### 第 48-50 步：YAML 支持

**新建 `src/experiment/loader.py`**：

```python
def load_yaml(path) -> dict: ...                # 读 + 校验根结构
def split_overrides(data) -> dict: ...          # 剥离 experiment/description/stages
def validate_override_keys(overrides): ...      # 白名单校验
def coerce_overrides(overrides) -> dict: ...    # 类型转换
def parse_stages(raw) -> list[Stage]: ...       # stage dict → Stage
```

**YAML 形态**：

```yaml
experiment: vit
description: baseline
learning_rate: 3e-4
batch_size: 128
stages:
  - name: align
    freeze: [vision]
    epochs: 1
```

**优先级链条**：

```text
TrainConfig 默认 < Experiment.config < YAML < CLI < stage.overrides
```

**`cli.py` 改动**：

- 加 `--config`
- 所有参数 default 改 `None`——区分"用户没传"vs"用户传了"
- `build_run(args)` 返回 `(experiment_name, config, seed, amp, num_train, stages)`

**YAML 1.1 的坑**：`1e-4` 是字符串，不是 float。`coerce_overrides` 按 `TrainConfig` 的字段类型强制转换。

---

### 阶段 15：评估层解耦（51-57）

#### 第 51-53 步：Metric 从 Task 剥离

**问题**：`Task` 混了 loss / metric / 训练范式。

**新结构**：

| 抽象 | 职责 |
|---|---|
| **Task** | loss 定义（`train_step`） |
| **Metric** | 评估（`evaluate`） |
| **Experiment** | 声明用哪些 metric + primary |

**`Task` 简化**：

```python
class Task(ABC):
    @classmethod
    def from_data(cls, bundle): return cls()

    @abstractmethod
    def train_step(self, model, batch, device, dtype) -> torch.Tensor: ...
```

**删掉**：`eval_step` / `primary_metric` / `higher_is_better`。

**`Metric` 新形态**：

```python
class Metric(ABC):
    name: str
    higher_is_better: bool

    @classmethod
    def from_data(cls, bundle): return cls()

    @abstractmethod
    def evaluate(self, model, loader, device, dtype) -> float: ...
```

**内置 metric**：`Accuracy` / `CrossEntropy` / `Perplexity`。

**`Experiment` 加字段**：

```python
metrics: list[type[Metric]]      # 类，不是实例
primary_metric: str
```

**`registry.py`**：

```python
"vit": Experiment(
    model=ViT,
    task=ClassificationTask,
    data="cifar10",
    metrics=[Accuracy, CrossEntropy],
    primary_metric="acc",
    ...
)
```

**删 `src/training/evaluator.py`**——`evaluate()` 函数不再需要。

#### 第 54 步：`run_every_n_epochs`

**Metric 加一个字段**：

```python
run_every_n_epochs: int | None = None
```

| 值 | 含义 |
|---|---|
| `1` | 每 epoch 跑（默认） |
| `N > 1` | 每 N epoch 跑 |
| `None` | 训中不跑，只训后跑 |

**Runner 过滤**：

```python
active = [
    m for m in metrics
    if m.run_every_n_epochs is not None and ctx.epoch % m.run_every_n_epochs == 0
]
```

**校验**：`primary_metric` 必须 `run_every_n_epochs == 1`（否则早停/best model 无数据）。

#### 第 55 步：Metric → ABC

**从 Protocol 改成 ABC**——和 `Task` 对称。

理由：需要给 `from_data` 默认实现。Protocol 无法给默认。

#### 第 56 步：`EvalBundle` + Metric loader 分发

**问题**：训练和评估需要**不同的条目粒度**。

| | 训练 / Perplexity | BLEU / CIDEr |
|---|---|---|
| 粒度 | (image, caption) 对 | image + 5 references |
| 数量（test） | 5000 | 1000 |

**两个 Bundle**：

```python
@dataclass
class DataBundle:
    """训练 + 训中评估。条目粒度 = 训练粒度。"""
    loader_train: DataLoader
    loader_val: DataLoader
    loader_test: DataLoader
    model_init: dict[str, Any]
    extras: dict[str, Any]

@dataclass
class EvalBundle:
    """训后评估。条目粒度 = 评估粒度。"""
    loader: DataLoader
    extras: dict[str, Any] = field(default_factory=dict)
```

**`build_data` 返回 tuple**：

```python
def build_data(name, ctx) -> tuple[DataBundle, EvalBundle | None]:
    return DATASET_REGISTRY[name](ctx)
```

**`Metric` 加两个方法**：

```python
def train_loader(self, data, eval_data) -> DataLoader:
    """训中评估用。默认 data.loader_val。"""
    return data.loader_val

def test_loader(self, data, eval_data) -> DataLoader:
    """训后评估用。默认 data.loader_test。"""
    return data.loader_test
```

**Runner 不做分发**——每个 metric 自己声明用哪个 loader。

**Flickr8k 新增**：

- `Flickr8kImageDataset`：一张图 + 全部 references
- `make_eval_collate_fn()`
- `_build_eval_bundle()`：构造 image 级 loader

#### 第 57 步：BLEU4

**`src/evaluation/bleu.py`**：

```python
def corpus_bleu(predictions, references, max_n=4) -> float: ...

class BLEU4(Metric):
    name = "bleu4"
    higher_is_better = True
    run_every_n_epochs = None       # 训中不跑

    def __init__(self, tokenizer, max_new_tokens=32): ...

    @classmethod
    def from_data(cls, data, eval_data=None):
        tokenizer = build_tokenizer(data.extras["tokenizer_name"])
        return cls(tokenizer=tokenizer)

    def test_loader(self, data, eval_data):
        assert eval_data is not None
        return eval_data.loader

    def evaluate(self, model, loader, device, dtype) -> float:
        # 遍历 image 级 batch
        # model.generate + decode
        # corpus_bleu(predictions, references)
```

**BLEU 数学要点**：

- **Corpus 级**，不是"每句算再平均"——全局 n-gram 统计
- Brevity penalty：`gen_len < ref_len` 时惩罚
- 多参考：取"最接近生成长度的参考"计算 ref_len

---

## 二、核心收获

### 1. 三个角色完全对称

| | Model | Task | Metric |
|---|---|---|---|
| 基类 | `nn.Module` 子类 | `ABC` | `ABC` |
| `from_data` | 默认 `cls(**model_init)` | 默认 `cls()` | 默认 `cls()` |
| 抽象方法 | `forward` | `train_step` | `evaluate` |
| 注册表存 | 类 | 类 | 类 |
| 构造时机 | bundle 之后 | bundle 之后 | bundle 之后 |

**"接口 + 默认实现"贯穿三者。**

### 2. "接口早，实现晚"的落地

`param_groups` / `stages` / `initialize` 三个接口在第 42-47 步立好，**没有任何具体实现**。LLaVA 接入时覆盖即可。

### 3. 评估层解耦

**三个维度**：

| 维度 | 字段 | 位置 |
|---|---|---|
| 执行时机 | `run_every_n_epochs` | Metric |
| 数据源 | `train_loader` / `test_loader` | Metric |
| 数据粒度 | `DataBundle` vs `EvalBundle` | Dataset |

**Runner 零 if。** 加新 metric 只加 metric 类 + registry 一行。

### 4. YAML 的三层优先级

```text
TrainConfig 默认 < Experiment.config < YAML < CLI < stage.overrides
```

**YAML 一个文件一套配置。** 名字用 `experiment:` 字段（不是文件名），`description` 人类可读。

---

## 三、踩的坑

### 坑 1：改继承时漏改 class 声明

第 42 步，`class ViT(nn.Module):` → `class ViT(Model):` 容易漏。**症状**：`issubclass(ViT, Model)` 为 `False`，`ViT` 没有继承来的方法。

### 坑 2：`monkeypatch.setitem` 不接收字符串路径

`monkeypatch.setitem("a.b.C", ...)` 会报错——它只接收 dict 对象。`monkeypatch.setattr` 才支持字符串路径。

### 坑 3：`classmethod` 不能用 `is` 比较

```python
assert ClassificationTask.from_data is Task.from_data   # ❌ 永远 False
assert "from_data" not in ClassificationTask.__dict__   # ✅
```

描述符协议每次返回新的 bound method。

### 坑 4：`Task` 没有默认 `from_data`

第 44 步的 `_MultiStageTask` 忘了写 `from_data`，`AttributeError`。**根本原因**：`Model` 有默认，`Task` 没有——不对称。第 45 步补齐。

### 坑 5：YAML 1.1 把 `1e-4` 解析成字符串

只有 `1.0e-4` 或 `0.0001` 是 float。解法：`coerce_overrides` 按 `TrainConfig` 字段类型强制转换。

### 坑 6：`build_data` 返回 tuple 后大量测试挂

`test_runner*.py` 里的 `_fake_data_bundle` 要改成返回 `(DataBundle, None)`。

### 坑 7：BLEU 测试用例设计

"per-sentence 平均 ≠ corpus BLEU"的测试，第一组数据恰好两个算法都得 0.5。换一组（`["a b c d", "x y z"]` vs `[["a b c d"], ["a b c"]]`）。

---

## 四、现在架构

```text
src/
├── evaluation/                 ← 新目录
│   ├── base.py                 ← Metric ABC + train_loader / test_loader
│   ├── metrics.py              ← Accuracy / CrossEntropy / Perplexity
│   └── bleu.py                 ← corpus_bleu + BLEU4
├── models/
│   └── base.py                 ← Model 基类
├── experiment/
│   ├── spec.py                 ← TrainConfig + Stage + Experiment
│   ├── loader.py               ← YAML 加载 + 校验 + coerce
│   ├── runner.py               ← 单/多阶段 + 训中/训后评估
│   └── artifacts.py
├── data/
│   ├── bundle.py               ← DataBundle + EvalBundle
│   ├── datasets.py             ← build_data 返回 tuple
│   └── flickr8k.py             ← Flickr8kDataset + Flickr8kImageDataset
├── tasks/
│   └── base.py                 ← 只剩 train_step
└── training/
    ├── train.py                ← 训练循环
    ├── hooks.py
    ├── strategy.py
    └── tracker.py
```

**删**：`src/experiment/config.py`、`src/training/evaluator.py`。

**"加东西"成本**：

| 动作 | 改动文件数 |
|---|---|
| 加新模型 | 2（`models/xxx.py` + `registry.py`） |
| 加新 task | 2（`tasks/xxx.py` + `registry.py`） |
| 加新 metric | 2（`evaluation/xxx.py` + `registry.py`） |
| 加 LLaVA 冻结 | 0（覆盖 `param_groups` + YAML freeze） |
| 加 LLaVA 多阶段 | 0（YAML `stages:`） |
| 加 FSDP | 1（`strategy.py` 加子类） |

**测试**：

```text
179 passed, 3 deselected
```

---

## 五、下一步

### 阶段 16：LLaVA-A 推理（1 天）

- 脚本级（`scripts/llava_infer.py`）
- 不改 framework
- 4bit 量化（4070 12G）
- `transformers` 只在 `scripts/`

### 阶段 17：LLaVA-B LoRA 微调（2-3 天）

**这一步会真正撞到 `DataBundle` 强类型化问题**。

### 阶段 18-19：YAML workflow + LLaVA-C 从 0 训

（见 architecture-notes）

---

## 六、一句话

> **本阶段把"训练范式"和"评估"从 Task 剥离，成为一等公民的声明式配置。**

- `Model.param_groups()` → 声明"我有几组参数"
- `Stage.freeze` → 声明"这一阶段冻谁"
- `Experiment.metrics` → 声明"用哪些评估"
- `Metric.run_every_n_epochs` → 声明"训中跑不跑"

**框架不动，只加声明。这就是"接口早，实现晚"的收益。**

---

*Last updated: 2026-10-03*