从 CS231N 教学代码，变成了一个正经的项目骨架。

ViT 跑通后，你可以继续往后做：

README（写清楚怎么跑）
CI（GitHub Actions 跑 pytest）
数据增强
混合精度
分布式
换更大的 ViT、换数据集

# 从 CS231N 学习项目到企业级项目：总结

---

## 一、今天做了什么转变

### 起点：CS231N 作业代码

```text
main.py        # 所有代码混在一起
layers.py      # 模型 + 初始化 + functional + 训练函数
```

问题：

- 改一个超参要翻遍文件
- 加一个模型要复制粘贴
- 换数据集要改训练代码
- 跑一次实验后结果不可复现
- 没有任何测试

### 终点：企业级项目骨架

```text
src/
├── config.py              # 配置集中
├── main.py                # 入口，只组装
├── data/
│   ├── transforms.py      # 数据变换
│   ├── cifar10.py         # 具体数据集
│   └── datasets.py        # 通用注册表 + loader 构建
├── models/
│   ├── mlp.py
│   ├── shallow_convnet.py
│   ├── deep_convnet.py
│   └── vit.py
├── training/
│   ├── train.py           # 通用训练循环
│   └── evaluator.py       # 通用评估
└── utils/
    ├── path.py            # 路径统一
    ├── logger.py          # 日志 + CSV
    ├── seed.py            # 可复现
    └── env.py             # 环境快照

tests/                     # pytest 冒烟测试
checkpoints/               # 模型权重
outputs/                   # 日志 + CSV + JSON
```

---

## 二、企业级必备的 10 个点

### 1. 关注点分离（Separation of Concerns）

**每个模块只做一件事：**

| 模块 | 只做 |
|---|---|
| `models/` | 定义网络结构 |
| `data/` | 加载和变换数据 |
| `training/` | 训练循环和评估 |
| `utils/` | 通用工具 |
| `main.py` | 组装和调度 |

**为什么重要：**

- 换模型不动训练代码
- 换数据集不动模型代码
- 换优化器不动数据代码

**你今天学到的：**

```python
# ❌ 学习代码
model = nn.Sequential(nn.Conv2d(...), ...)
train_part34(model, optimizer)

# ✅ 企业代码
model = DeepConvNet()
optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
train(model, optimizer, loader_train, loader_val, ...)
```

---

### 2. 配置管理（Configuration Management）

**所有超参集中在一个地方，不在代码里散落。**

你今天实现的：

```python
@dataclass
class TrainConfig:
    experiment: str = "mlp"
    epochs: int = 1
    learning_rate: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = True
    seed: int = 42
    lr_scheduler: str = "none"
    early_stop_patience: int = 0
```

**为什么重要：**

- 改超参不改代码
- 可以记录每次实验用了什么超参
- 支持命令行覆盖
- 以后可以换成 YAML / Hydra

**企业里更进一步的：**

- `argparse` → 命令行参数
- `YAML` / `Hydra` / `OmegaConf` → 层级配置
- 环境变量 → 敏感信息

---

### 3. 可复现性（Reproducibility）

**同一条命令，跑两次，结果要一致。**

你今天实现的：

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

再加上 JSON 快照：

```json
{
  "config": { "seed": 42, "learning_rate": 0.01, ... },
  "env": { "python": "3.12.3", "torch": "2.2.2", ... }
}
```

**为什么重要：**

- 论文复现
- Bug 排查
- A/B 对比实验

**企业里更进一步的：**

- Docker 固定环境
- `requirements.txt` / `pyproject.toml` 锁版本
- DVC / MLflow 管理数据和实验

---

### 4. 日志与监控（Logging & Monitoring）

**不要把 `print` 散落在代码里。**

你今天实现的：

```python
logger = get_logger("mlp", log_path)
logger.info("Epoch %d done. val_acc = %.4f", e, val_acc)
```

同时写：

- 终端
- `.log` 文件
- `.csv` 文件（方便画曲线）

**为什么重要：**

- 训练几小时后还能回溯
- 多个实验对比
- 出问题时能查

**企业里更进一步的：**

- TensorBoard / WandB / MLflow
- 实时监控 GPU 利用率
- 报警（loss 爆炸、NaN）

---

### 5. Checkpoint 管理

**保存模型不只是 `torch.save(model.state_dict())`。**

你今天实现的：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "epoch": result["last_epoch"],
    "best_acc": result["best_acc"],
    "config": dataclasses.asdict(cfg),
}, ckpt_path)
```

**为什么重要：**

- 断点续训
- 保留最优模型
- 回溯训练过程

**企业里更进一步的：**

- 定期保存 + 最优保存
- 云端存储（S3 / GCS）
- 模型版本管理

---

### 6. 测试（Testing）

**改动模型后，怎么知道没改坏？**

你今天实现的：

```python
def test_mlp_output_shape(dummy_batch):
    x, _ = dummy_batch
    model = MLP()
    out = model(x)
    assert out.shape == (4, 10)
```

**为什么重要：**

- 重构不怕
- 加新模型有模板
- CI 自动检查

**企业里更进一步的：**

- 单元测试（函数级）
- 集成测试（训练 1 步不报错）
- 回归测试（精度不低于基线）
- CI/CD（GitHub Actions）

---

### 7. 注册表模式（Registry Pattern）

**加新东西不改老代码。**

你今天实现的：

```python
EXPERIMENTS = {
    "mlp": (MLP, 1e-2),
    "shallow_convnet": (ShallowConvNet, 1e-2),
    "deep_convnet": (DeepConvNet, 0.1),
    "vit": (ViT, 3e-4),
}

DATASET_REGISTRY = {
    "cifar10": cifar10.build_datasets,
}
```

**为什么重要：**

- 加模型只改字典
- 加数据集只改字典
- 不用改 `main.py` 逻辑

**企业里更进一步的：**

- 装饰器自动注册
- 插件系统
- 配置文件驱动

---

### 8. 命令行接口（CLI）

**同一个脚本，不同参数，不同实验。**

你今天实现的：

```bash
python main.py --experiment vit --epochs 10 --batch-size 256
python main.py --experiment mlp --epochs 2 --resume checkpoints/mlp_xxx.pt
```

**为什么重要：**

- 不用改代码就能跑不同实验
- 方便脚本化 / 自动化
- 方便远程运行

**企业里更进一步的：**

- 子命令（`train` / `eval` / `export`）
- 参数校验
- 默认值管理

---

### 9. 环境快照（Environment Snapshot）

**每次运行，记录当时的软硬件环境。**

你今天实现的：

```python
{
    "python": "3.12.3",
    "platform": "macOS-26.6.2",
    "torch": "2.2.2",
    "torchvision": "0.17.2",
    "cuda_available": False
}
```

**为什么重要：**

- 换机器后对比
- 升级 torch 后回溯
- 排查环境相关 bug

**企业里更进一步的：**

- Docker 镜像
- `pip freeze` 锁版本
- 云平台元数据

---

### 10. 项目结构约定（Project Layout）

**目录名、文件名的约定。**

你今天学到的：

```text
src/            代码
├── models/     模型
├── data/       数据
├── training/   训练
└── utils/      工具
tests/          测试
checkpoints/    权重
outputs/        日志
datasets/       数据（不进 git）
```

**为什么重要：**

- 新人 5 分钟看懂
- 工具（pytest、mypy）能自动识别
- 打包、部署方便

**企业里更进一步的：**

- `src/` layout（避免 import 冲突）
- `pyproject.toml` 统一配置
- `Makefile` / `scripts/` 常用命令

---

## 三、你今天没做但企业里必备的

这些以后可以慢慢加：

| 项目 | 作用 | 什么时候加 |
|---|---|---|
| `pyproject.toml` 完整配置 | 依赖、工具配置 | 现在就有，可以完善 |
| `README.md` | 别人怎么跑 | 立刻 |
| `.gitignore` | 不提交 datasets/ 等 | 立刻 |
| GitHub Actions | 自动跑测试 | 有 GitHub 仓库时 |
| Dockerfile | 环境隔离 | 部署时 |
| 混合精度（AMP） | GPU 加速 2x | 有 GPU 时 |
| 分布式训练（DDP） | 多卡 | 多卡时 |
| 数据增强 | 提升精度 | 想提升时 |
| 学习率预热 | 稳定训练 | 大模型时 |
| 梯度裁剪 | 防止爆炸 | RNN / Transformer |
| 早停（Early Stopping） | 防止过拟合 | ✅ 已加 |
| 模型导出（ONNX/TorchScript） | 部署 | 上线时 |
| 推理服务（FastAPI/Triton） | API 化 | 上线时 |

---

## 四、你今天最该记住的 5 句话

1. **每个文件只做一件事。** 模型文件不写训练，训练文件不写数据。

2. **所有变化的东西，都应该是参数。** 学习率、batch size、epochs，不要写死在函数里。

3. **所有全局的东西，都应该从 `config` 拿。** `device`、`dtype`、`print_every`，不要在每个文件里重新定义。

4. **入口文件要薄。** `main.py` 只做组装，不做实现。

5. **可复现性 = 随机种子 + 完整配置快照 + 环境信息。** 没有这三样，实验等于白跑。

---

## 五、企业级 vs 学习代码：一张对照表

| 维度 | 学习代码 | 企业代码 |
|---|---|---|
| 文件组织 | 所有代码在一个文件 | 按职责分目录 |
| 模型定义 | 写在 `main.py` | 在 `models/` 一个文件一个模型 |
| 超参 | 硬编码 | `config.py` + CLI |
| 训练循环 | 和模型混在一起 | `training/train.py` 通用 |
| 日志 | `print` | `logging` + 文件 + CSV |
| 可复现 | 不管 | seed + JSON 快照 |
| Checkpoint | 只存 model | model + optimizer + scheduler + epoch |
| 断点续训 | 不支持 | `--resume` |
| 测试 | 没有 | pytest |
| 加新模型 | 复制粘贴 | 注册表加一行 |
| 加新数据集 | 改训练代码 | 注册表加一行 |
| 环境记录 | 没有 | JSON 快照 |

---

## 六、下一步学什么

按优先级：

1. **`pyproject.toml` 完整配置** — 依赖管理、工具配置
2. **`README.md`** — 项目说明书
3. **`.gitignore`** — 什么不该进 Git
4. **GitHub Actions** — 自动跑 pytest
5. **数据增强** — `torchvision.transforms`
6. **混合精度（AMP）** — GPU 上 2x 加速
7. **TensorBoard** — 可视化训练曲线
8. **Docker** — 环境隔离
9. **分布式训练** — 多卡
10. **模型部署** — ONNX / TorchScript / FastAPI

---

## 七、一句话总结

> **企业级项目的本质不是用了多少高级工具，而是：任何一个变化，都只需要改一个地方。**

- 改超参 → 只改 `config.py`
- 加模型 → 只加 `models/xxx.py` + 注册表一行
- 加数据集 → 只加 `data/xxx.py` + 注册表一行
- 换训练策略 → 只改 `training/train.py`

这就是你今天学到的最核心的东西。

---

回家用 4070 跑完 ViT，然后从上面"下一步学什么"里选一个开始做。有具体问题随时问。