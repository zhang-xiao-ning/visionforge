# Memo 11: 数据先行架构 + 训练层能力

> 从"模型负责装配"到"数据先行 + 各自适配"
> 时间：2026-09-28
> 版本：v0.1.1

---

## 一、本阶段做了什么

### 1. 训练层 4 个能力

`train.py` 之前只支持 SGD + 固定 lr。加 4 个能力，让"能训 transformer"：

| 能力 | 解决的问题 |
|---|---|
| **优化器抽象** | SGD 写死 → 支持 SGD / AdamW |
| **学习率调度** | 只有 epoch 级 → 加 step 级 warmup + cosine |
| **梯度累积** | 大模型显存不够 → 等效大 batch |
| **梯度裁剪** | Transformer 训练不稳定 → 限制梯度范数 |

**配置**：`TrainConfig` 加 5 个字段，默认值 = 现有行为，**向后兼容**。

```python
@dataclass
class TrainConfig:
    # ... 原有字段
    optimizer: str = "sgd"          # "sgd" / "adamw"
    weight_decay: float = 0.0
    warmup_steps: int = 0
    accum_steps: int = 1
    grad_clip: float = 0.0
```

**验证**：4070 上 AdamW + cosine + warmup 训练成功（captioning 小规模）。

### 2. 数据先行架构（from_data）

**旧问题**：`CaptioningModel.setup(ctx)` 负责建 model + task + loaders。**模型管了 task 和 data**——责任错位。

**新架构**：

```text
build_data(name, ctx) → DataBundle
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
       model.from_data   task.from_data    loaders
              │               │
              └───────┬───────┘
                      ↓
                  Runner.run()
```

**核心契约**：

```python
@dataclass
class DataBundle:
    loader_train: DataLoader
    loader_val: DataLoader
    loader_test: DataLoader
    model_init: dict[str, Any]   # 构造模型所需 kwargs
    extras: dict[str, Any]        # 运行时依赖（tokenizer_name 等）
```

**每个角色只做一件事**：

| 角色 | 职责 |
|---|---|
| `build_data` | 建数据集 → `DataBundle` |
| `model.from_data(bundle)` | 从 bundle 拿 `model_init` 构造自己 |
| `task.from_data(bundle)` | 从 bundle 拿 task 相关参数 |
| `Runner` | 只串联，**无 if** |

**注册表声明组合**：

```python
EXPERIMENTS = {
    "vit": {
        "data": "cifar10",
        "model": ViT,
        "task": ClassificationTask,
        "lr": 3e-4,
        "category": "classification",
    },
    "captioning": {
        "data": "flickr8k",
        "model": CaptioningModel,
        "task": CaptioningTask,
        "lr": 1e-3,
        "category": "captioning",
    },
}
```

### 3. Checkpoint 自包含

**旧**：`sample_caption.py` 要 `--tokenizer tiktoken` 外部指定。

**新**：checkpoint 保存重建信息：

```python
torch.save({
    "model": ...,
    "optimizer": ...,
    "scheduler": ...,
    "epoch": ..., "best_acc": ...,
    "config": dataclasses.asdict(cfg),
    "model_init": bundle.model_init,   # ← 新增
    "extras": bundle.extras,            # ← 新增
}, path)
```

**采样**：

```python
ckpt = torch.load(path)
tokenizer_name = ckpt["extras"]["tokenizer_name"]   # "tiktoken"
model = CaptioningModel(**ckpt["model_init"])        # vocab_size, pad_id
```

**checkpoint 是自包含交付物**——拿到 `.pt` 就能重建模型，不需要外部参数。

---

## 二、改动清单

| 文件 | 操作 |
|---|---|
| `data/bundle.py` | **新建**：`DataBundle` + `DataContext` |
| `data/datasets.py` | 重写：`build_data(name, ctx)` |
| `data/cifar10.py` | 加 `build_bundle(ctx)` |
| `data/flickr8k.py` | 加 `build_bundle(ctx)` |
| `models/base.py` | **删除**（被 `DataBundle` 取代） |
| `models/vit.py` | 加 `from_data` |
| `models/captioning.py` | 加 `from_data`（去 `setup`） |
| `tasks/classification.py` | 加 `from_data` |
| `tasks/captioning.py` | 加 `from_data` |
| `registry.py` | tuple → dict（含 `category`） |
| `experiment/runner.py` | 无 if；用 `from_data` |
| `experiment/artifacts.py` | 保存 snapshot 加 `model_init` / `extras` |
| `cli.py` | 删 `--dataset`；加 5 个训练参数 |
| `scripts/sample_caption.py` | 从 checkpoint 读配置 |
| `Makefile` | 加 `NUM` / `OPT` / `WD` / `WARMUP` / `ACCUM` / `CLIP` |
| `README.md` | 重写 |
| 测试 | 多个文件跟着改 |

---

## 三、关键决策

### 决策 1：为什么用 `dict` 而不是强类型

`DataBundle.model_init` / `extras` 都是 `dict[str, Any]`。

**另一个方案**：每类任务一种 dataclass（`LMBundle` / `ClassificationBundle`）。

**选 dict 的理由**：

- 当前 2 类任务，dict 够用
- 强类型要每类任务定义一种——**增加样板**
- mypy 对 dict 检查弱，但**真正的错误在 `from_data` 内会立刻爆**
- **等第 3 类任务接入时再评估**（Rule of Three）

### 决策 2：`category` 字段

注册表加 `category`，让契约测试自动过滤：

```python
_CLASSIFICATION_EXPERIMENTS = [
    name for name, exp in EXPERIMENTS.items()
    if exp["category"] == "classification"
]
```

**好处**：加新模型自动被契约测试覆盖，**不用手改测试文件**。

### 决策 3：`DATASET_INFO` 与 `DATASET_REGISTRY` 分开

- `DATASET_REGISTRY`：`name → build_fn`（运行时建 bundle）
- `DATASET_INFO`：`name → DatasetInfo`（静态元信息，供测试用）

**原因**：`model_init` 在 bundle 里，但**测试仍需要"每个 dataset 的形状"**来生成 dummy batch。

### 决策 4：`TrainConfig` 默认值 = 向后兼容

新字段全部有默认值：

- `optimizer: str = "sgd"`
- `accum_steps: int = 1`
- `grad_clip: float = 0.0`
- `warmup_steps: int = 0`
- `weight_decay: float = 0.0`

**效果**：不传新参数 → 行为和之前**完全一致**。

---

## 四、踩的坑

### 坑 1：`cli.py` 解包错误

注册表从 tuple 变 dict 后：

```python
# 旧（错）
_, default_lr = EXPERIMENTS[args.experiment]

# 新（对）
default_lr = EXPERIMENTS[args.experiment]["lr"]
```

**症状**：`lr = dict` → `optim.SGD` 里 `dict < float` 报 TypeError。

### 坑 2：Checkpoint 里有类对象

`dataclasses.asdict(cfg)` 里的字段**必须全是基础类型**。任何字段塞了"类"或"整个 dict"，JSON 序列化就崩。

**诊断方法**：临时加 `default=_json_default` 打印不可序列化的值。

### 坑 3：Makefile 缩进

- **变量续行**（`TRAIN_ARGS = \`）：**空格**
- **target recipe**（`@echo ...`）：**Tab**

**混用** → `commands commence before first target`。

---

## 五、现在架构

```text
build_data(name, ctx) → DataBundle
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
       model.from_data   task.from_data    loaders
              │               │
              └───────┬───────┘
                      ↓
                  Runner.run()
```

**"加东西"的成本**：

| 动作 | 改动文件数 |
|---|---|
| 加模型 | **2**（`models/xxx.py` + `registry.py`） |
| 加 dataset | **2**（`data/xxx.py` + `datasets.py`） |
| 加 task | **2**（`tasks/xxx.py` + `registry.py`） |
| 换 dataset | **0**（改注册表或 CLI，不动代码） |

---

## 六、下一步

| 方向 | 时间 |
|---|---|
| 评估层（BLEU / CIDEr） | 半天 |
| LLaVA 推理脚本 | 1 天 |
| Logger 级别系统 | 1~2h |
| from_data 优化（DatasetInfo 合并） | 1h |

**长期**：接生成式 / LM → 撞 framework 缺口 → 重构 Trainer。

---

## 七、一句话

> **本阶段把"模型负责装配"变成"数据先行 + 各自适配"。**

- 模型不管 task
- Task 不管 model
- Runner 无 if
- Checkpoint 自包含