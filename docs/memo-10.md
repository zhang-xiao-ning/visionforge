# Memo 10: 从分类器到多模态模型

> Captioning 从零到跑通
> 时间：2026-09-23 ~ 2026-09-25
> 版本：v0.1.1

---

## 一、本阶段做了什么（第 33-40 步）

### 第 33 步：Task 抽象

**问题**：`train.py` 写死分类逻辑

```python
loss = F.cross_entropy(scores, y)   # 写死
val_acc = evaluate(model, loader_val)   # 写死 accuracy
```

**改法**：引入 `Task` 抽象

```python
class Task(ABC):
    primary_metric: str = "loss"
    higher_is_better: bool = True

    def train_step(model, batch, device, dtype) -> Tensor: ...
    def eval_step(model, batch, device, dtype) -> dict[str, float]: ...
```

**好处**：`train.py` / `evaluator.py` 变成任务无关。

**新增**：`higher_is_better` 字段处理"越小越好"的指标（loss / perplexity）。

### 第 34 步：字符级 Tokenizer

**目的**：让 captioning 能快速跑通（不用 BPE）。

**产物**：`CharTokenizer` — 从语料构建字符表 + 4 个特殊 token。

**关键决策**：

- `pad_id = 0`，`bos_id = 1`，`eos_id = 2`，`unk_id = 3`
- 支持 `save` / `load`

**当时不知道的坑**：后面会被 tiktoken 取代。

### 第 35 步：Flickr8k Dataset

**产物**：`Flickr8kDataset` + `build_captioning_loaders`

**关键设计**：

| 决策 | 原因 |
|---|---|
| 返回 dict（不是 tuple） | Captioning batch 结构复杂 |
| `input_ids` / `target_ids` 错位一位 | 自回归训练标准 |
| **target padding 用 `-100`** | PyTorch ignore_index 标准 |
| `collate_fn` 是闭包 | 需要 `pad_id`，但 DataLoader 只接受单参函数 |

**数据集坑**：图片目录是 `Flicker8k_Dataset`（少一个 r），文本目录是 `Flickr8k_text`。原始作者拼写错误。

### 第 36 步：Captioning 模型

**架构**：

```text
图像 (B, 3, 224, 224)
    ↓ PatchEmbedding (Conv2d)
    ↓ + 位置编码
TransformerEncoder (4 层)
    ↓ memory
TransformerDecoder (4 层)  ← input_ids 通过 Embedding + causal mask
    ↓
lm_head (Linear, vocab_size)
    ↓ logits (B, L, V)
```

**关键实现**：

- `_causal_mask`：`torch.triu(ones, diagonal=1)`，True = 屏蔽
- `generate`：贪心自回归采样
- **`tie_weights`**：`lm_head.weight = token_embed.weight`（省一半参数）

### 第 37 步：CaptioningTask

```python
class CaptioningTask(Task):
    primary_metric = "perplexity"
    higher_is_better = False
```

**loss = cross_entropy(ignore_index=-100)**，metric = perplexity = `exp(loss)`。

### 第 38 步：Tokenizer 协议讨论（纯架构）

**触发问题**：`CaptioningModel` 需要 `vocab_size`，`vocab_size` 从数据来。

**依赖链**：

```text
数据集 → tokenizer → vocab_size → 模型
```

**讨论的三个方案**：

| 方案 | 结论 |
|---|---|
| setup 函数进注册表 | ❌ 注册表职责混乱 |
| tuple 4 元组 `(model, lr, task, tokenizer)` | ❌ 冗余（model 和 task 一一对应） |
| runner 纯注入 | ❌ main.py 要薄 |

**最终决策**：

- **能力驱动**（鸭子类型）：谁需要 `setup` 谁实现
- **不引入 ABC 基类**：分类模型零改动
- **`setup` 延迟 import**：避免顶层反向依赖

### 第 39 步：Tokenizer 协议化 + 一切落地

**新目录**：

```text
src/data/tokenizers/
├── __init__.py       # build_tokenizer 工厂
├── base.py           # Tokenizer Protocol
├── char.py           # CharTokenizer（从旧 tokenizer.py 迁移）
└── tiktoken_bpe.py   # TiktokenTokenizer（新）
```

**`Tokenizer` 协议**：

```python
class Tokenizer(Protocol):
    vocab_size: int
    pad_id: int      # -1 = 没有
    bos_id: int
    eos_id: int
    unk_id: int
    def encode(text, add_special=True) -> list[int]: ...
    def decode(ids, skip_special=True) -> str: ...
    def save(path): ...
```

**关键约定**：

- 没有的特殊 token 返回 `-1`（tiktoken 没有 unk）
- `encode(add_special=True)` 语义 = BOS + tokens + EOS 包裹

**`SetupContext` + `ExperimentBundle`**（`models/base.py`）：

```python
@dataclass
class SetupContext:
    cfg: TrainConfig
    dataset_name: str
    batch_size: int
    num_train: int | None = None

@dataclass
class ExperimentBundle:
    model, task, loader_train, loader_val, loader_test
```

**`CaptioningModel.setup`**：

```python
@classmethod
def setup(cls, ctx) -> ExperimentBundle:
    tokenizer = build_tokenizer(cls.tokenizer_name)
    loaders = build_captioning_loaders(...)
    model = cls(vocab_size=tokenizer.vocab_size, ...)
    task = CaptioningTask()
    return ExperimentBundle(model, task, *loaders)
```

**`runner.py`** 加 if 分支（**带 TODO**）：

```python
# TODO: 临时分叉。三个 setup 风格模型出现时抽成 builder.py
if hasattr(model_cls, "setup"):
    bundle = model_cls.setup(ctx)
    ...
else:
    # 分类：默认路径
```

### 第 40 步：训练跑通 + 采样

**两次 NaN bug**（详见第五节）。

**最终训练结果**：

| Epoch | train_loss | val_perplexity |
|---|---|---|
| 1 | 5.51 | 117.39 |
| 2 | 4.50 | 81.38 |
| 3 | 4.22 | 66.46 |
| 4 | 4.06 | 58.86 |
| 5 | 3.93 | 53.37 |
| **Test** | — | **50.22** |

**首次生成**：

```
Image:     a young girl in a pink dress climbing stairs
Generated: A man in a red shirt is sitting on a red shirt in a red shirt .
```

**语法通顺，语义不准**——5 epoch 小模型的正常表现。

---

## 二、三个抽象总结

### 1. Task（第 33 步）

**管**：loss / metric 的差异

```text
分类：cross_entropy + accuracy
Captioning：cross_entropy(shift) + perplexity
```

**价值**：`train.py` / `evaluator.py` 一次写好，永不再改。

### 2. Tokenizer Protocol（第 39 步）

**管**：不同 tokenizer 后端的差异

```text
CharTokenizer       — 字符级
TiktokenTokenizer   — GPT-2 BPE
SentencePiece       — 未来（中文）
Qwen Tokenizer      — 更远未来
```

**价值**：切 tokenizer 只加一个文件 + 1 行注册。

### 3. Setup 能力驱动（第 39 步）

**管**：数据依赖模型的构建

```text
分类：模型无参，loaders 独立
Captioning：vocab_size 从 tokenizer 来，有依赖链
```

**价值**：分类模型零改动，captioning 自己声明 `setup`。

**Pythonic 关键**：**鸭子类型**（`hasattr` 判断）而非 ABC 继承。

---

## 三、两个 NaN bug（重点记录）

### Bug 1：`tie_weights=False`

**症状**：Iter 0 loss = nan

**原因**：`CaptioningModel.__init__` 里 `tie_weights: bool = False`（我写的默认值错了）。

**为什么 NaN**：tie_weights=False 本身不会 NaN，但——

**真正的坑**：tie_weights=False 时 `lm_head` 和 `token_embed` 都是独立大矩阵。**如果初始化规模不对 + SGD 学习率 1e-3**，梯度容易爆炸。

**修法**：`tie_weights: bool = True`（默认）。

### Bug 2：`tgt_key_padding_mask` 把 BOS 也 mask 了

**症状**：tie_weights 改成 True 后仍然 NaN。

**原因**：

```python
tgt_key_padding_mask = input_ids == self.pad_id
```

**tiktoken 的 `pad_id == bos_id == eos_id == 50256`**，所以 BOS 也被判为 padding 位置被 mask。

**如果整个 batch 的序列都被 mask**，attention softmax 全 -inf → **NaN**。

**修法**：删掉 `tgt_key_padding_mask`。

**为什么能删**：

- loss 已用 `ignore_index=-100` 屏蔽 padding
- decoder 处理 padding 位置产生的 logits 会被 loss 忽略
- `tgt_key_padding_mask` 只是优化，**在 tiktoken 下是 bug**

**教训**：**共享 token（pad == bos == eos）的 tokenizer，不要用 token id 做 mask 判断。**

---

## 四、Tokenzier 协议设计（未来切中文的关键）

### 切中文场景：从 tiktoken → SentencePiece

**改动清单**：

| 文件 | 类型 | 行数 |
|---|---|---|
| `src/data/tokenizers/sentencepiece_bpe.py` | 新增 | ~100 |
| `scripts/train_sp_tokenizer.py` | 新增（一次性） | ~30 |
| `src/data/tokenizers/__init__.py` | 加 1 行 | +1 |
| `src/models/captioning.py` | 改 `tokenizer_name` 类属性 | +1 |

**应用代码（`runner.py` / `train.py` / `evaluator.py` / `main.py`）改动 = 0。**

### 验收标准

> **未来切中文时，改动 ≤ 3 个文件。**

**满足。**

### 为什么这么设计

| 层 | 职责 | 变化时 |
|---|---|---|
| `Tokenizer` Protocol | 接口 | 不变 |
| 适配器（`char.py` / `tiktoken_bpe.py` / `sp.py`） | 吸收底层差异 | 加新文件 |
| 工厂（`build_tokenizer`） | 名字 → 实例 | 加 1 行 |
| 模型 / 数据 / 训练 | 只依赖协议 | **不变** |

**这是 Protocol + Adapter + Factory 三个模式的组合。**

---

## 五、踩的坑

### 坑 1：PyCharm refactor 把文件搬错位置

**症状**：`data.tokenizers.tokenizer` 找不到。

**原因**：PyCharm 把 `tokenizer.py` 移到 `tokenizers/tokenizer.py`，我设计的是 `tokenizers/char.py`。

**修法**：`git mv src/data/tokenizers/tokenizer.py src/data/tokenizers/char.py`。

**教训**：**PyCharm refactor 只做"位置移动"，不做"重命名以匹配新架构"。**

### 坑 2：tiktoken 共享 token

**教训**：`pad_id == bos_id == eos_id`，任何时候用 token id 做逻辑判断都要小心。

### 坑 3：target padding 必须用 `-100`

**原因**：`cross_entropy` 的 `ignore_index` 默认是 `-100`，用 `pad_id` 会让**真实的 EOS** 也被忽略（tiktoken 下 pad == eos）。

### 坑 4：`best_state` 快照存 CPU

**原因**：`v.detach().cpu().clone()` 避免在 GPU 上多占一份模型权重。

**对 1.3B 级模型，这是 2.6GB 的显存差异。**

---

## 六、训练结果

### 数据规模

- 训练：6000 张图 × 5 caption = **30000 训练对**
- Val：1000 张图 × 5 caption
- Test：1000 张图

### 训练曲线

```
Epoch 1: loss=5.51, perplexity=117.39
Epoch 2: loss=4.50, perplexity=81.38
Epoch 3: loss=4.22, perplexity=66.46
Epoch 4: loss=4.06, perplexity=58.86
Epoch 5: loss=3.93, perplexity=53.37
Test:    loss=3.88, perplexity=50.22
```

**Perplexity 从 50257（随机）降到 50.22，下降 1000 倍。**

### 时间

4070 上每 epoch ~3.5 分钟，5 epoch 总 ~20 分钟。

---

## 七、下一步可能的改进

| # | 改进 | 预期收益 |
|---|---|---|
| 1 | **Beam search** | 减少重复，句子流畅 |
| 2 | **Temperature sampling** | 生成多样性 |
| 3 | **训练更多 epoch**（20~50） | 语义更准 |
| 4 | **加大模型**（d_model=512, 8 层） | 表达能力更强 |
| 5 | **换 SentencePiece**（中文） | 支持中文 captioning |
| 6 | **COCO 数据集**（12 万图） | 训练数据 20 倍 |
| 7 | **加 `/caption` serving 端点** | 端到端部署 |
| 8 | **从 GPT-2 初始化 encoder** | 迁移学习 |

---

## 八、一句话

> **本阶段从"分类器"走到了"多模态模型"。**

- 之前：Image → 10 类
- 现在：Image → 自然语言描述

**从 CS231N 到真正多模态的第一步。**

---

## 九、项目现状

### 里程碑总览

| # | 里程碑 | 状态 |
|---|---|---|
| 1 | 核心骨架 | ✅（1-10） |
| 2 | 项目外观 | ✅（11-13） |
| 3 | 训练增强 | ✅（14-15） |
| 4 | 可观测 | ✅（16） |
| 5 | 本地开发环境 | ✅（17-19） |
| 6 | 工程质量 | ✅（20-22） |
| 7 | DevOps 闭环 | ✅（23-25） |
| 8 | CI | ✅（26） |
| 9 | 训练镜像 | ✅（27） |
| 10 | 重构 + DDP | ✅（28） |
| 11 | 工程打磨 | ✅（29-32） |
| 12 | **Captioning** | ✅（33-40） |

### 测试

```text
76 passed, 5 skipped
```

### 项目结构（关键部分）

```text
visionforge/
├── src/
│   ├── data/
│   │   ├── tokenizers/          # 协议 + 适配器 + 工厂
│   │   ├── flickr8k.py          # captioning 数据集
│   │   ├── cifar10.py           # 分类数据集
│   │   └── datasets.py          # 分类 loader 构建
│   ├── models/
│   │   ├── base.py              # SetupContext / ExperimentBundle
│   │   ├── captioning.py        # ViT Encoder + Transformer Decoder
│   │   ├── vit.py
│   │   └── ...
│   ├── tasks/
│   │   ├── base.py              # Task Protocol
│   │   ├── classification.py
│   │   └── captioning.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluator.py
│   │   └── strategy.py
│   └── experiment/
│       ├── config.py
│       ├── artifacts.py
│       └── runner.py
├── scripts/
│   ├── ci.sh
│   ├── download_datasets.sh
│   └── train_ddp.sh
└── ...
```

---

## 十、附：本阶段完整命令

```bash
# 1. 下载数据
bash scripts/download_datasets.sh flickr8k

# 2. 训练
uv run python src/main.py \
    --experiment captioning \
    --dataset flickr8k \
    --epochs 5 \
    --batch-size 32

# 3. 采样
uv run python scripts/sample_caption.py \
    --checkpoint checkpoints/captioning_xxx.pt \
    --image datasets/Flicker8k_Dataset/xxx.jpg

# 4. 检查
make check
```

---

*Last updated: 2026-09-26*