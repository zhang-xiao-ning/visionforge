# Memo 8: 架构重构 + DDP

> 从"能跑"到"能扩展"
> 时间：2026-09-21 ~ 2026-09-22
> 版本：v0.1.0

---

## 一、为什么要重构

**触发点**：想加 DDP，发现要改 **4 个文件**：
- `main.py`（建 logger / recorder / writer / model / save）
- `data/datasets.py`（sampler）
- `training/train.py`（log 判断 + set_epoch）
- `main.py::run_experiment`（17 件事挤一坨）

**这是架构信号**：加一个功能动 N 个文件 = 分层不够。

**决策**：引入两个抽象

| 抽象 | 管什么 |
|---|---|
| `TrainingStrategy` | 训练拓扑的差异（Single / DDP / FSDP） |
| `RunArtifacts` | 运行产物（路径 / logger / recorder / writer） |
| `ExperimentRunner` | 编排（把 17 件事拆成方法） |

---

## 二、Commit 拆解（6 个）

### Commit 1：`TrainingStrategy`

- `src/training/strategy.py`
- 接口：`wrap_model` / `make_*_sampler` / `is_main_process` / `set_epoch` / `cleanup`
- 实现：`SingleDeviceStrategy` + `DDPStrategy`
- `build_strategy()`：自动检测 `RANK` / `WORLD_SIZE`
- **只加文件，不改现有代码**

### Commit 2：`RunArtifacts`

- `src/experiment/artifacts.py`
- `RunArtifacts.create()`：一次性建路径 + logger + recorder + writer
- DDP 时 `is_main=False` → 只写 rank 0
- **只加文件，不改现有代码**

### Commit 3：`ExperimentRunner` + `main.py` 切换

- `src/experiment/runner.py`
- 把 `run_experiment` 的 17 件事拆成方法
- `main.py`：从 200 行减到 90 行
- 数值**逐位一致**验证通过

### Commit 4：strategy 接入底层

- `data/datasets.py`：`build_loaders(strategy=...)`
- `training/train.py`：`train(strategy=...)`
- `runner.py`：把 strategy 传下去
- 行为**零变化**（SingleDevice 等价于原逻辑）

### Commit 5：DDP 脚本 + 4070 验证

- `scripts/train_ddp.sh`：torchrun 启动
- `Makefile` 加 `train-ddp`
- 4070 上 `NPROC=1 make train-ddp EXP=mlp EPOCHS=1` ✅
- 可复现性验证：跑两次，数值**逐位一致**

### Commit 6：文档

- 本 memo
- `COMPLETED.md` / `TODO.md`

---

## 三、核心收获

### 1. 判断"该不该重构"的信号

> **加一个功能要改 ≥3 个文件 = 该重构。**

DDP 要改 4 个文件 → 触发。

### 2. 重构的分层原则

```text
main.py            ← 只做：解析参数 → 建 Runner → run
    ↓
ExperimentRunner   ← 编排：建对象、串流程
    ↓
├── TrainingStrategy   ← 策略：单卡 / DDP / FSDP
├── RunArtifacts       ← 产物：日志 / CSV / TB
└── 底层（models / data / train）
```

**"变化点"集中在 strategy 和 runner，其他文件不动。**

### 3. 每次重构一个 commit

- Commit 1-2：**只加不改**，零风险
- Commit 3：**切换**，数值验证
- Commit 4：**接入**，行为不变
- Commit 5：**验证**，DDP 跑通

**每个 commit 独立验证，任何一步挂了都能回退。**

### 4. 验证重构正确性的方法

**数值逐位一致**：

```text
重构前: Iter 0 loss = 2.3835
重构后: Iter 0 loss = 2.3835   ← 完全一样
```

**这是最强的正确性证明**：行为完全等价。

### 5. DDP 可复现性

**DDP 自己的可复现 ≠ 和单卡一致**：

- DDP 用 `DistributedSampler`，数据顺序和单卡不同
- **但 DDP 内部两次运行完全一致**（相同 seed）

**结论**：DDP 可复现，但和单卡数值不同是正常的。

---

## 四、加 FSDP 有多简单

**只需要**：

1. 新建 `FSDPStrategy(TrainingStrategy)`
2. `build_strategy()` 里加判断

**其他文件（`main.py` / `runner.py` / `datasets.py` / `train.py`）完全不动。**

**这就是 OCP 的价值。**

---

## 五、下一步

### DevOps 剩余

- [ ] docker-compose 多服务
- [ ] 模型版本管理（MLflow / DVC）

### 多模态方向

- [ ] Image Captioning（A3 路线）
- [ ] CLIP（A4 路线）
- [ ] 抽象 `train.py` 的 loss / metric

### 工程质量

- [ ] 版本号单一来源（`importlib.metadata`）

---

## 六、一句话

> **本阶段把"改一个功能动 4 个文件"变成"改一个功能动 1 个文件"。**

---

*Last updated: 2026-09-22*