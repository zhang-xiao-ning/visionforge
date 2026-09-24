# Memo 7: 训练镜像

> 训练也上 Docker，GPU 直通
> 时间：2026-09-20
> 版本：v0.1.0

---

## 一、本阶段做了什么（第 27 步）

### 第 27 步：训练 Docker 镜像

**目标**：让训练也容器化，环境隔离、可移植。

**产物**：

```text
docker/Dockerfile.train       # GPU 训练镜像
docker-compose.yml            # 加 train service
```

**Dockerfile 关键设计**：

```dockerfile
FROM python:3.12-slim

# 清华源加速非 torch 依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# GPU 版 torch（CUDA 12.1）
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchvision==0.17.2

# 其它依赖
RUN pip install --no-cache-dir \
    tensorboard \
    "numpy<2" \
    pillow

COPY ../src ./src/
COPY ../pyproject.toml ./

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "src/main.py"]
CMD ["--experiment", "mlp", "--epochs", "1"]
```

**`docker-compose.yml` 的 train service**：

```yaml
  train:
    build:
      context: .
      dockerfile: docker/Dockerfile.train
    image: visionforge-train:latest
    container_name: visionforge-train
    volumes:
      - ./datasets:/app/datasets:ro
      - ./checkpoints:/app/checkpoints
      - ./outputs:/app/outputs
    environment:
      - USE_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles:
      - train
```

**关键点**：

- `profiles: [train]`：默认 `docker compose up` 不启动
- `datasets:ro`：只读挂载，防止容器破坏数据
- `deploy.resources.reservations.devices`：请求 GPU
- `USE_GPU=true`：让 `config.py` 选 CUDA

**实测结果**：

```text
using device: cuda
torch: 2.2.2+cu121
cuda_available: True

1 epoch / ~6 秒
两次运行数值完全一致（可复现）
```

---

## 二、本阶段核心收获

### 1. 训练 / 推理镜像分离

| 镜像 | torch 版本 | 大小 | 用途 |
|---|---|---|---|
| `visionforge-serve` | `+cpu` | ~400 MB | 推理 |
| `visionforge-train` | `+cu121` | ~2.4 GB | 训练 |

**同一个项目，两个镜像，各取所需。**

**为什么不能共用一个**：

- 推理不需要 CUDA，用 CPU torch 省 2GB
- 训练需要 CUDA，必须装 GPU torch
- **分开是优化的必然结果**

### 2. profiles 的用法

```yaml
services:
  api:
    ...        # 默认启动
  train:
    ...
    profiles: [train]    # 只在 --profile train 时启动
```

- `docker compose up` → 只起 api
- `docker compose --profile train up` → 起 api + train
- `docker compose --profile train run --rm train ...` → 只跑一次 train

**用途**：把"默认服务"和"按需服务"分开。

### 3. GPU 直通的三层配置

```text
第 1 层：Windows / WSL2
  └── nvidia-container-toolkit 已装
第 2 层：docker-compose.yml
  └── deploy.resources.reservations.devices
第 3 层：容器内
  └── torch.cuda.is_available() → True
```

**任一层配置错 → 容器拿不到 GPU。**

### 4. 卷挂载的策略

```yaml
volumes:
  - ./datasets:/app/datasets:ro      # 只读
  - ./checkpoints:/app/checkpoints   # 读写
  - ./outputs:/app/outputs           # 读写
```

**为什么 `datasets:ro`**：

- 数据集不应该被容器改
- 只读挂载能防止意外
- 性能更好（Docker 可以优化）

### 5. `ENTRYPOINT` + `CMD` 的设计

```dockerfile
ENTRYPOINT ["python", "src/main.py"]
CMD ["--experiment", "mlp", "--epochs", "1"]
```

**效果**：

```bash
# 用默认参数
docker run visionforge-train

# 覆盖参数
docker run visionforge-train --experiment vit --epochs 10
```

**`ENTRYPOINT` 固定，`CMD` 可覆盖。**

---

## 三、项目现状

### 完整交付链路

```text
代码 → 训练 → 导出 → 部署 → 调用
 ↑       ↑      ↑      ↑      ↑
Mac     Docker  ONNX   Docker  SSH
```

**每一步都有对应的工具和验证。**

### 镜像总览

```bash
$ docker images | grep visionforge
visionforge-serve   latest   ...   ~400 MB
visionforge-train   latest   ...   ~2.4 GB
```

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
| 8 | CI 自动化 | ✅（26） |
| 9 | **训练镜像** | ✅（27） |

---

## 四、本阶段踩的坑

### 坑 1：`uv.lock` 锁死 CUDA torch

**症状**：CI 上 `UV_TORCH_BACKEND=cpu` 不生效。

**原因**：`uv sync` 读 lock 文件，不重新解析。

**解**：CI 里 `rm -f uv.lock`。

**延伸**：这个坑在训练镜像里**不存在**，因为 Dockerfile 直接 `pip install --index-url .../cu121`，不走 uv。

### 坑 2：PyCharm 一直在 "Updating python interpreter"

**症状**：Mac 上 PyCharm 卡了 10+ 分钟，下载 1.2GB。

**原因**：PyCharm 远程解释器同步，把 WSL 上的 CUDA torch 拉到本地。

**解**：待解决（下次开工先处理）。

### 坑 3：`make train-docker` 只能在 4070 上跑

**症状**：Mac 上 `make train-docker` 报错。

**原因**：Mac 没有 NVIDIA GPU，也没有 nvidia-container-toolkit。

**解**：`make train-docker` 是"GPU 机器专用"命令，Mac 上用本地 `make train`（走 MPS）。

---

## 五、下一步

### DevOps 剩余

- [ ] 分布式训练 DDP
- [ ] docker-compose 多服务
- [ ] 训练镜像的更多参数（`--data-dir` / `--output-dir`）

### 算法方向

- [ ] ResNet / EfficientNet
- [ ] 学习率 warmup
- [ ] MixUp / CutMix

### 多模态方向

- [ ] Image Captioning（A3 路线）
- [ ] CLIP（A4 路线）
- [ ] 抽象 `train.py` 支持不同 loss / metric

### 工程剩余

- [ ] 解决 PyCharm 远程解释器问题
- [ ] 版本号单一来源

---

## 六、一句话

> **本阶段让训练也变成"一次构建，到处运行"。**

关键区别：

- 之前：训练环境依赖本机 `uv sync`，换机器要重装
- 现在：`docker compose --profile train run train` 一句命令，在任何有 GPU 的机器上跑

**从"本机训练"到"容器化训练"。**

---

## 七、附：完整命令

```bash
# 1. 构建训练镜像（4070，一次性）
docker compose --profile train build train

# 2. 跑训练
make train-docker EXP=mlp EPOCHS=1
make train-docker EXP=vit EPOCHS=10 BS=256

# 3. 看结果（宿主机上）
ls checkpoints/
ls outputs/
```

---

*Last updated: 2026-09-20*