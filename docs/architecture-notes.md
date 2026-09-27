# Architecture Notes

设计决策档案。记录"为什么这么设计"，而不是"做了什么"。
每条决策包含：问题 / 讨论 / 结论 / 待定项。

---

## 2026-09-27：实验装配中心该在哪里

### 背景

`ExperimentRunner` 在加入 captioning 时，出现了一个 if 分支：

```python
if hasattr(model_cls, "setup"):
    bundle = model_cls.setup(ctx)     # captioning
else:
    ...                                # classification（默认路径）
```

`setup` 由模型类实现，负责构造：

```python
@classmethod
def setup(cls, ctx):
    tokenizer = build_tokenizer(...)      # 数据
    loaders = build_captioning_loaders(...)  # 数据
    model = cls(vocab_size=..., pad_id=...)  # 模型自己
    task = CaptioningTask()               # 任务  ← 模型管了不该管的
    return ExperimentBundle(...)
```

**问题**：模型承担了"装配中心"的职责，包括构造 task 和 loader。这不对。

### 讨论

**核心洞察**：依赖链是**数据先行**。

```text
数据集 → DataBundle
       ↓
       ├──→ 模型构造参数（num_classes / vocab_size / pad_id）
       ├──→ Task 构造参数
       └──→ Loaders
```

**装配不该由模型负责，应该"数据先行 + 各自适配"**。

### 方向（未定案）

**`from_data` 模式**：

```python
# 1. 建数据（dataset 负责）
bundle = build_data(name, ctx)   # → 某种 DataBundle

# 2. 模型适配数据
model = model_cls.from_data(bundle)

# 3. 任务适配数据
task = task_cls.from_data(bundle)

# 4. Runner 只串联，无 if
```

**注册表声明组合**：

```python
EXPERIMENTS = {
    "vit": {
        "data": "cifar10",
        "model": ViT,
        "task": ClassificationTask,
        "lr": 1e-3,
    },
    "captioning": {
        "data": "flickr8k",
        "model": CaptioningModel,
        "task": CaptioningTask,
        "lr": 1e-3,
    },
}
```

### DataBundle 的类型安全（三个方案）

| 方案 | 说明 | 评价 |
|---|---|---|
| **1. 约定 key** | `extras: dict[str, Any]`，靠命名约定 | 弱，运行时才能发现拼写错误 |
| **2. 强类型 Bundle** | 每类任务一种 dataclass（`LMBundle` / `ClassificationBundle`） | **倾向**，mypy 能检查 |
| **3. BaseBundle + 继承** | 共享 loader 部分，各自扩展 | 可能过度设计 |

**方案 2 示例**：

```python
@dataclass
class LMBundle:
    loader_train: DataLoader
    loader_val: DataLoader
    loader_test: DataLoader
    tokenizer: Tokenizer     # 强类型
```

**模型 / 任务针对特定 Bundle 类型实现 `from_data`，mypy 检查类型**。

### 待定 / 不做

**不现在实现**，原因：Rule of Three。当前只有 2 类任务（分类 + captioning）。第 3 类（生成式 / 语言模型）接入时才知道真实需求。

**做 CS236 / CS224N 时留意**：

1. **生成式任务的数据形态**：是 `(x, y)`？还是 `(noise, condition)`？
2. **生成式任务的评估指标**：FID / IS / 采样图？——影响 `Task.eval_step`
3. **生成式任务的特殊训练步骤**：DDPM 随机 t、VAE 重参数化？——影响 `Task.train_step`
4. **语言模型的数据形态**：`(input_ids, labels)`？packed sequence？——影响 `DataBundle`

**等 3 类任务全部就位，按 `from_data` 方向重构**。

---

## 2026-09-27：Tokenizer 差异是否破坏架构

### 背景

未来支持中文（CS224N）时，中文 / 英文 tokenizer 不同。会不会破坏"数据先行 + `from_data`"？

### 结论

**不会。** 因为：

> **tokenizer 是数据集的属性，不是模型的属性。**

- 中文数据集 → 中文 tokenizer
- 英文数据集 → 英文 tokenizer
- 数据准备时就知道用哪个 tokenizer

**换语言 = 换数据集 = 换一个实验**。

### 示例

```python
# data/chinese_corpus.py
def build_data(ctx) -> LMBundle:
    tokenizer = build_tokenizer("sentencepiece-zh")
    loaders = build_lm_loaders(..., tokenizer=tokenizer)
    return LMBundle(*loaders, tokenizer=tokenizer)

# data/english_corpus.py
def build_data(ctx) -> LMBundle:
    tokenizer = build_tokenizer("tiktoken")
    loaders = build_lm_loaders(..., tokenizer=tokenizer)
    return LMBundle(*loaders, tokenizer=tokenizer)

# 同一个 GPT 类
class GPT(nn.Module):
    @classmethod
    def from_data(cls, bundle: LMBundle) -> GPT:
        return cls(vocab_size=bundle.tokenizer.vocab_size, ...)

# 注册表
EXPERIMENTS = {
    "gpt-zh": {"data": "chinese_corpus", "model": GPT, ...},
    "gpt-en": {"data": "english_corpus", "model": GPT, ...},
}
```

**同一个 `GPT` 类，两个实验各自实例化。模型层零改动。**

### 关键结论

> **架构的抽象粒度是"数据集"，不是"语言"。**
> **"语言差异"被隔离在数据层。**

### 未来可能相关但不在本设计范围

**"先英文预训练，再中文微调"**：

- 这是"工作流"层的事（实验之间传 checkpoint / tokenizer）
- 不是"单次实验"层的事
- **不要让它污染当前的实验设计**

---

## 通用原则（本档案中反复出现的）

### 1. 数据先行

> 数据是装配的起点。模型和任务各自从数据里取需要的参数。

### 2. 责任单一

> 模型不管 task 构造，task 不管模型构造，runner 不管具体任务。

### 3. Rule of Three

> 3 个同类场景才抽象。2 个场景用现有方案 workaround。
> 第 3 类任务接入时是重构的最好时机。

### 4. 别让"未来可能"污染"当前设计"

> 转移学习、分布式、多模态融合——都不是单次实验层的事，别塞进 runner。