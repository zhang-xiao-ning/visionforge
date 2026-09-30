# Architecture Notes

**定位**：战略方向 + 架构应对。

- **不写**已落地的设计（README 里有）
- **不写**具体 feature（TODO 里有）
- **只写**：未来往哪走 → 为了这个方向，架构要做哪些调整

**逻辑链**：战略方向 → 架构应对 → 触发条件。

> ⚠️ **本文档已废弃 "Rule of Three"（三次法则）原则。**
> 替代原则：**接口早，实现晚**——接口/契约/协议早做，具体实现/具体类型/具体形状晚做。
> 判据：**"这个抽象的名字能写出来吗？"** 能 → 接口，早做；不能 → 实现，晚做。
> 详见"通用原则 3"。

**与其它文档的分工**：

| 文档 | 内容 |
|---|---|
| `README.md` | 怎么用（当前能力） |
| `TODO.md` | 待做的 feature |
| `COMPLETED.md` | 已完成的里程碑 |
| `memo-*.md` | 开发日志（历史） |
| **`architecture-notes.md`** | **往哪走（本文档）** |

---

## 战略方向总览

| # | 方向 | 状态 | 触发时机 |
|---|---|---|---|
| 1 | **多模态（LLaVA）** | 🎯 主方向 | CS236 学完后 |
| 2 | **生成式（Diffusion / VAE）** | 📘 学习期 | CS236 课程中 |
| 3 | **语言模型（LM）** | 📘 学习期 | CS224N 课程中 |
| 4 | **大模型训练基础设施** | 🔮 远期 | 模型放大到撞硬件时 |

---

## 方向 1：多模态（LLaVA）

### 战略

当前 captioning 是**教学级范式**（从零训练的小 ViT + Transformer，0.5M 参数）。

企业级是 **LLaVA**：

```text
冻结 CLIP/SigLIP → 简单 MLP 投影 → 冻结 LLM（+ LoRA）
```

**目标**：从"能训 captioning"到"能训 LLaVA 级多模态模型"。

### 架构应对

| 能力 | 现状 | 要做 | 层次 |
|---|---|---|---|
| **LLM 接入** | ❌ 无 | `models/llava.py` 包装 HF `transformers` | 实现 |
| **多模态数据** | ❌ 无 | `data/llava_instruct.py`（对话格式） | 实现 |
| **参数分组 / 冻结** | ❌ `train.py` 假设全参数训练 | `Model.param_groups()` 接口 | **接口** |
| **多阶段训练** | ❌ `train()` 假设单阶段 | `Task.stages` 接口 | **接口** |
| **预训练权重加载** | ❌ 无 | `Model.initialize()` 接口 | **接口** |
| **Task 抽象** | ✅ CE loss 通用 | 加 `loss_fn` 参数化 | 实现 |
| **Serving** | ⚠️ ONNX / FastAPI | 换 vLLM / TGI | 实现 |
| **导出** | ⚠️ ONNX | 7B 模型可能要放弃 ONNX | 实现 |

### 关键洞察：LLaVA 破的是"训练层"，不是"数据/模型/任务层"

| 层 | 加 LLaVA 时 | 破 OCP？ |
|---|---|---|
| 数据层 | 加 `data/llava_instruct.py` | ❌ 纯加 |
| 模型层 | 加 `models/llava.py` | ❌ 纯加 |
| 任务层 | 加 `tasks/llava.py` | ❌ 纯加 |
| **训练层** | **加三个能力** | ✅ **要改现有接口** |

**"接口早，实现晚"的应用**：

- 三个**接口**（`param_groups` / `stages` / `initialize`）**现在就可以定**
- 三个**实现**（LLaVA 的参数分组策略 / 两阶段划分 / 权重加载）等接入时再加

### 三档渐进

| 阶段 | 工作量 | 目标 | Framework 缺口 |
|---|---|---|---|
| **A. 推理** | 1 天 | 加载现有 LLaVA 权重，跑对话 | 无（只是推理脚本） |
| **B. LoRA 微调** | 2~3 天 | 冻结 + LoRA | **参数冻结** |
| **C. 从 0 训** | 1 周 | 完整 LLaVA 架构 | **多阶段训练** + 大数据加载 |

**路径**：A → B → C。**每层暴露不同缺口，撞到再改。**

### 触发条件

CS236 学完后，**先跑通 LLaVA 推理（阶段 A）**。

- A 阶段**不改 framework**（脚本级依赖 `transformers` + `accelerate` + `bitsandbytes`）
- 撞到 framework 缺口时再动架构
- 依赖边界：`transformers` 只出现在 `scripts/`，不进 `src/`

### 已排除

- ❌ **Serving `/caption` 端点** —— 未来被 vLLM 替代，死路
- ❌ **HF Tokenizer 适配** —— 不想被 HF 生态绑死
- ❌ **HF `Trainer` / `datasets` / `trl`** —— 只把 HF 当"权重加载器"，不当"训练框架"

---

## 方向 2：生成式（Diffusion / VAE）

### 战略

CS236 课程涵盖生成式。**目标是支持 diffusion / VAE 训练**——不追 SOTA，只求跑通。

### 架构应对（待观察）

**当前不确定的地方**：

| 问题 | 影响 |
|---|---|
| **数据形态**：`(x, y)`？还是 `(noise, condition)`？ | 影响 `DataBundle` 是否要强类型化 |
| **评估指标**：FID / IS / 采样图？ | 影响 `Task.eval_step` 接口 |
| **训练步骤**：DDPM 随机 t、VAE 重参数化？ | 影响 `Task.train_step` 是否需要 `**kwargs` |

**这 3 件事会告诉你**：

- `DataBundle` 是否够用（还是要加 `condition` / `timestep` 字段）
- `Task` 抽象是否够用（还是要支持"特殊训练步骤"）
- `Trainer` 重构是否必要

### 触发条件

**做 CS236 作业时留意**——撞到 framework 缺口时再动。

**不要预判性设计**——等第 3 类任务真实接入时才有依据。

---

## 方向 3：语言模型（LM）

### 战略

CS224N 课程涵盖 LM。**目标是支持语言模型训练**——为多模态的 LLM 部分铺路。

### 架构应对（待观察）

| 能力 | 现状 | 要做 | 层次 |
|---|---|---|---|
| **数据形态**：`(input_ids, labels)` / packed sequence | ❌ 无 | `data/lm_corpus.py` + packing 处理 | 实现 |
| **Tokenizer**：中文 tokenizer | ⚠️ 协议已备 | 加 SentencePiece adapter（≤3 文件） | 实现 |
| **推理**：KV cache | ❌ 无 | `generate` 加 cache | 实现 |
| **量化**：4bit / 8bit | ❌ 无 | `bitsandbytes` / `torch.quantization` | 实现 |
| **多阶段**：先预训练 → 再微调 | ❌ 无 | 复用方向 1 的多阶段能力 | **接口** |

### 触发条件

**做 CS224N 作业时留意**——撞到 framework 缺口时再动。

---

## 方向 4：大模型训练基础设施

### 战略

**远期目标**：framework 能训 7B+ 模型。

当前只能训 0.5M 模型。**放大 1000 倍时，训练基础设施要全面升级。**

### 架构应对

| 能力 | 现状 | 要做 | 层次 |
|---|---|---|---|
| **参数冻结** | ❌ | 见方向 1 | **接口** |
| **LoRA / QLoRA** | ❌ | `peft` 集成 | 实现 |
| **梯度检查点** | ❌ | `torch.utils.checkpoint` | 实现 |
| **FSDP / DeepSpeed** | ❌ 只有 DDP | `TrainingStrategy` 加子类 | 实现 |
| **混合精度** | ✅ AMP 有 | 大模型实测 | 实现 |
| **ZeRO 分片** | ❌ | DeepSpeed | 实现 |
| **分布式 checkpoint** | ❌ | 每 rank 保存分片 | 实现 |
| **多机多卡** | ❌ 只有单机 | 跨机 DDP / torchrun | 实现 |

### 触发条件

**模型放大到撞硬件时**——按需引入，不做预判。

**`TrainingStrategy` 接口已备**——加 FSDP 只需加一个子类，其他文件不动。

---

## 通用架构原则

这些不随时间变，是所有方向共享的指导思想。

### 1. 数据先行

> 数据是装配的起点。模型和任务各自从数据里取需要的参数。

**应用**：`build_data → DataBundle → from_data`

**反例**：模型负责构造 task / loader（memo-10 前的 `setup` 模式）。

### 2. 责任单一

> 模型不管 task 构造，task 不管模型构造，runner 不管具体任务。

**应用**：无 if 的 runner；每个角色只做一件事。

### 3. 接口早，实现晚

> **接口 / 契约 / 协议** → 早做。
> **具体实现 / 具体类型 / 具体形状** → 晚做。

**为什么**：

- 接口的晚做成本是**复利**——改 N 个调用者 + 已稳定代码
- 实现的早做成本是**猜错**——做出来不对，白做

**判据**："这个抽象的名字能写出来吗？"

- 能（`ParamGroupsProvider`）→ 接口，早做
- 不能（"支持未来可能的 X"）→ 实现，晚做

**应用状态**：

| 项 | 层 | 状态 |
|---|---|---|
| 训练层 4 能力 | 接口 | ✅ 早做 |
| Logger 级别 | 接口 | ✅ 早做 |
| `Task.stages` | 接口 | ✅ 可早做 |
| `Model.param_groups()` | 接口 | ✅ 可早做 |
| `Model.initialize()` | 接口 | ✅ 可早做 |
| `Trainer` 类 | 实现 | ⏳ 等需求明确 |
| `DataBundle` 强类型化 | 实现 | ⏳ 等字段稳定 |
| FSDP / DeepSpeed | 实现 | ⏳ 有多卡时 |

**注**：早期 memo（memo-2 / memo-9 / memo-11）中出现的 "Rule of Three" 原则**已废弃**。历史记录保留，但设计判断以本条为准。

### 4. 别让"未来可能"污染"当前设计"

> 转移学习、分布式、多模态融合——都不是单次实验层的事。

**应用**：不往 runner 里塞"未来可能"的东西。

### 5. 让"变化"集中在一个点

> 加模型改 2 文件；加 dataset 改 2 文件；加 task 改 2 文件。

**应用**：注册表 + `from_data` + `build_data`。

### 6. 契约测试用"注册表遍历"

> 加新实现 = 注册表加一行，测试自动覆盖。

**应用**：`tests/contracts/`。

### 7. 隐式类型分发是技术债

> `hasattr` / `isinstance` / ABC 继承——"根据对象身份决定行为"都是同一个坑。

**正确做法**：**声明 + 默认**——每个对象有默认实现，子类覆盖它。

**已排除**：

- `if hasattr(model_cls, "setup")`（memo-10 的债）
- `if isinstance(task, MultiStageTask)`
- `if task.higher_is_better`（已修：`Task.is_improvement()`）

**正确形态**：

```python
class Task(ABC):
    stages: list[Stage] | None = None   # 默认单阶段
```

**判据**："框架是在'猜'子类的类型，还是在'消费'子类的声明？"

- 猜 → 债
- 消费 → OCP

---

## 待定设计

每条记录：问题 / 设想 / 触发条件。**不是决策，是待办**。

### P1: `AppLogger` 合并三通道

**问题**：`AppLogger` / `CSVRecorder` / `SummaryWriter` 三个对象写同一份 epoch 数据。

**设想**：

```python
logger.record(epoch, train_loss, val_metrics, primary_metric, lr)
```

一个方法，内部写 CSV + TB。

**待定**：`record` 签名是否稳定？未来加 LLaVA 的 `stage_name` / `loss_align` 会怎样？

**触发**：现在可做（独立小改动）。

### P2: `Trainer` 类重构

**问题**：`train()` 现在 18 个参数 + 105 行。

**设想**：拆成 `Trainer` 类 + 方法。

**触发**：第 3 类任务撞出新的 hook / config 需求时。

### P3: `DataBundle` 强类型化

**问题**：`model_init` / `extras` 是 `dict[str, Any]`——mypy 弱检查。

**设想**：每类任务一种 dataclass（`LMBundle` / `ClassificationBundle`）。

**触发**：key 拼错经常发生时。

**注意**：这是"具体类型"（实现层），**字段形状现在还不知道**。等生成式 / LM 接入时才知道真实需求。

### P4: 框架和应用物理分离

**问题**：`framework` 和 `application` 混在一起。

**设想**：拆 repo 或 monorepo。

**触发**：第 2 个项目出现时。

**已做**：边界**已声明**（`framework/__init__.py`）——这是"接口"层。
**待做**：物理拆分——这是"实现"层。

---

## 演进日志

| 日期 | 事件 |
|---|---|
| 2026-09-27 | 确定 Tokenizer 协议化；契约测试；框架边界 |
| 2026-09-28 | 数据先行落地；训练层 4 能力；Checkpoint 自包含；确认 LLaVA 为主方向 |
| 2026-09-29 | 训练循环清理（strategy 归位）；Logger 级别系统（NONE / ERROR / INFO / FLOW / DEBUG） |
| 2026-09-30 | 废弃 Rule of Three，改用"接口早，实现晚"；定位文档为"战略规划" |

---

*Last updated: 2026-09-30*