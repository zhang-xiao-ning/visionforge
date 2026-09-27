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


---

## 2026-09-27：Captioning 向 LLaVA 演进的分析

### 背景

当前 captioning 的架构是 ViT Encoder + Transformer Decoder（Show-and-Tell 的现代版），
从零训练，参数 ~0.5M，Flickr8k 数据，4070 上 20 分钟跑完。

问题：**这是教学级范式。企业级的 captioning / 多模态是什么？改成那种范式要多少工作量？**

### 企业级范式：LLaVA（非 BLIP-2）

| 维度 | 当前（教学级） | BLIP-2 | **LLaVA** |
|---|---|---|---|
| 图像编码器 | 从零训的小 ViT | 冻结 CLIP | **冻结 CLIP / SigLIP** |
| 中间层 | 无（直接 cross-attn） | Q-Former | **简单 MLP** |
| 文本解码器 | 从零训的小 Transformer | 冻结 LLM | **冻结 LLM**（或 LoRA） |
| 参数量 | ~0.5M | 7B+ | 7B ~ 13B |
| 训练阶段 | 单阶段 | 两阶段 | **两阶段（对齐 + 指令微调）** |

**结论**：
- BLIP-2 是技术探索（Q-Former 复杂，复现难）
- **LLaVA 是 2024 年事实标准**（简单、社区主流、易复现）
- 企业做多模态基本从 LLaVA 起步

### 当前 captioning 完整度

| 维度 | 完整度 | 说明 |
|---|---|---|
| 教学范式 | **100%** | 从零训练 captioning 跑通 |
| 企业范式 | **~20%** | 基础设施齐，模型是玩具 |
| 部署范式 | **~40%** | ONNX / FastAPI / Docker 有，但服务小模型 |

**Framework 组件 readiness**：

| 组件 | 直接上 LLaVA？ |
|---|---|
| 数据加载框架 | ⚠️ 需新 Dataset，但架构可用 |
| 训练循环 | ⚠️ **需加"冻结参数"支持** |
| Task 抽象 | ✅ captioning loss 和 LLaVA loss 都是 CE |
| Tokenizer 协议 | ✅ 加 HF adapter 即可 |
| Checkpoint / resume | ✅ 通用 |
| 日志 / TensorBoard | ✅ 通用 |
| ONNX 导出 | ❌ 7B 模型 ONNX 很难 |
| Serving | ❌ 要换 vLLM / llama.cpp |
| 模型架构 | ❌ 要重写成 LLaVA |

### 从当前到 LLaVA 的改动清单

**框架层（加能力，OCP 友好）**：

| 改动 | 工作量 |
|---|---|
| `train.py` 支持冻结参数 | 半天 |
| `train.py` 支持两阶段训练 | 半天 |
| `Task` 加 `loss_fn` 参数化 | 半天 |
| Tokenizer 加 HF 适配器 | 2 小时 |

**小计：1~2 天**。都是"加能力"，**不改现有接口**。

**模型层（重写）**：

| 新文件 | 工作量 |
|---|---|
| `models/llava.py`（包装 CLIP + MLP + LLM） | 半天 |
| `data/llava_instruct.py`（对话格式 Dataset） | 半天 |

**小计：1 天**。

**部署层（大改）**：

| 改动 | 工作量 |
|---|---|
| ONNX 导出（7B） | **可能放弃** |
| Serving 换 vLLM | 1~2 天 |
| Docker 镜像（10GB+） | 半天 |

**小计：2~4 天**。**服务性能和 0.5M 模型完全不同**。

### OCP 分析（关键）

**OCP 友好**：
- 加 LLaVA 模型 → `models/llava.py` 新建 + 注册表一行
- 加 LLaVA 数据 → `data/llava_instruct.py` 新建 + 注册表一行
- 加 HF Tokenizer → `data/tokenizers/hf.py` + 工厂一行

**破坏 OCP 的 4 处**：
- **参数冻结**：`train.py` 假设"所有参数都训练"
- **多阶段训练**：`train()` 假设"单阶段"
- **Serving 换引擎**：ONNX/FastAPI 是给小模型的
- **导出格式**：ONNX 对 7B 不友好，可能换 safetensors

**这是"新能力的引入"，不是"扩展点"**——`train.py` 要加能力，不是纯加文件。

### 三档渐进方案（推荐）

| 层级 | 工作量 | 学到什么 | Framework 缺口 |
|---|---|---|---|
| **A. 复现 LLaVA 推理** | 1 天 | 加载现有权重，跑对话 | 无（只是推理脚本） |
| **B. LoRA 微调现有 LLaVA** | 2~3 天 | 冻结 + LoRA | 参数冻结 |
| **C. 从 0 训小 LLaVA** | 1 周 | 完整架构 | 多阶段训练 + 大数据加载 |

**推荐路径：A → B → C**

**理由**：每一层暴露不同的 framework 缺口。先跑通 A，撞到需求再改 framework。

### 决策

**现在不做。** 先跑通 LLaVA 推理（阶段 A）。

**理由**：
1. 当前 framework 为 LLaVA 准备了 80%
2. 不改框架也能跑推理
3. **真实需求在阶段 B/C 才浮现**
4. Rule of Three：现在 2 类任务，等"完整版 captioning"作为第 3 类再动框架

**触发条件**：CS236 学完后，尝试 LLaVA 推理，撞到 framework 缺口时再改。

### 与"装配中心"决策的关系

如果 LLaVA 接入需要"参数冻结 + 多阶段训练"，**这些能力也应该挂在任务层**：
- `Task` 声明 `trainable_parameters`（用于冻结）
- `Task` 声明 `stages`（多阶段）
- **Runner 只调 `task.train_step`**，不感知阶段/冻结

**这符合"数据先行 + `from_data`"的方向**——能力从任务/数据声明，runner 只编排。

### 相关硬件需求

| 阶段 | 硬件 |
|---|---|
| A（推理） | 4070 够（4bit 量化 7B） |
| B（LoRA 微调） | 4070 够（QLoRA） |
| C（从 0 训） | **不够**（要 A100） |

**未来可能需要的硬件讨论**：云 GPU 租用成本。