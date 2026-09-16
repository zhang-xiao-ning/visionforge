# Memo 5: DevOps 闭环——从模型到上线

> 训练 → 导出 → 部署 → 上线，全链路打通
> 时间：2026-09-16
> 版本：v0.1.0

---

## 一、本阶段做了什么（第 23-25 步）

### 第 23 步：ONNX 导出

**目标**：把 PyTorch checkpoint 转成 ONNX，让模型能被任何语言/框架调用。

**产物**：

```text
src/export/onnx_export.py
exports/<experiment>.onnx
```

**核心代码**：

```python
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

**验证**：PyTorch 和 ONNX Runtime 各跑一遍，对比输出。

```python
max_diff = float(np.abs(torch_out - onnx_out).max())
assert max_diff < 1e-4
```

**实测结果**：

- mlp: `max_diff = 9.54e-07`
- deep_convnet: `max_diff = 2.38e-06`
- vit: 跳过（Mac 上没有 checkpoint）

**学到的**：

- `dynamic_axes` 让 batch 维度可变，不写死 batch=1
- `opset_version` 影响算子支持，17 是当前稳定的主流
- **验证是必须的**：导出成功 ≠ 导出正确

---

### 第 24 步：FastAPI 推理服务

**目标**：起一个 HTTP 服务，接收图片，返回分类结果。

**产物**：

```text
src/serving/
├── api.py        # FastAPI 路由
├── inference.py  # ONNX Runtime 推理
└── schema.py     # Pydantic schema
```

**两个端点**：

- `GET /health` → `{"status": "ok", "model": "mlp.onnx"}`
- `POST /predict` → 上传图片，返回 top-k 分类

**响应格式**：

```json
{
  "predictions": [
    {"class_id": 3, "class_name": "cat", "confidence": 0.42},
    {"class_id": 5, "class_name": "dog", "confidence": 0.21}
  ]
}
```

**自动文档**：FastAPI 自带 Swagger UI

```text
http://localhost:8000/docs
```

代码即文档，直接页面上传图片测试。

**学到的**：

- `UploadFile = File(...)` 接收上传文件
- `response_model=PredictionResponse` 自动序列化 + 验证
- `@app.post("/predict")` 同步函数即可，FastAPI 自动处理
- **Pydantic 是 FastAPI 的灵魂**：请求/响应都有类型

---

### 第 25 步：Docker 部署

**目标**：把服务打包成镜像，能部署到任何有 Docker 的机器。

**产物**：

```text
docker/Dockerfile.serve
docker-compose.yml
.dockerignore
```

**Dockerfile 关键设计**：

```dockerfile
FROM python:3.12-slim

# 清华源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# CPU 版 torch（体积小）
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2

# 其它依赖
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    "onnxruntime<1.20" \
    "numpy<2"

COPY src/ ./src/
COPY exports/ ./exports/

ENV PYTHONPATH=/app/src
ENV ONNX_PATH=/app/exports/mlp.onnx
EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**镜像大小**：~400 MB（vs GPU 版 2.4 GB）

**闭环验证**：

```text
Mac curl
   ↓ SSH 隧道 (-p 6000, -L 8000:localhost:8000)
阿里云中转服务器
   ↓
WSL2 (4070)
   ↓ Docker 端口映射
FastAPI 容器
   ↓ onnxruntime
返回 JSON
```

---

## 二、本阶段核心收获

### 1. 部署的三层抽象

| 层 | 工具 | 作用 |
|---|---|---|
| 模型 | ONNX | 跨框架、跨语言 |
| 服务 | FastAPI | HTTP 接口，结构化 I/O |
| 部署 | Docker | 环境隔离，可移植 |

**每一层解决一个不同的问题**，缺一不可。

### 2. 为什么用 CPU 版 torch

| 版本 | 大小 | 用途 |
|---|---|---|
| `torch+cpu` | ~190 MB | 只推理 |
| `torch+cu121` | ~2.4 GB | 训练 / GPU 推理 |

**ONNX Runtime 推理不依赖 PyTorch CUDA**。用 CPU 版：
- 镜像小
- 启动快
- 部署简单

### 3. Docker 的层缓存

```dockerfile
FROM python:3.12-slim          # 缓存
RUN pip install torch ...      # 缓存
RUN pip install fastapi ...    # 缓存
COPY src/ ./src/               # 改代码 → 只重跑这层
COPY exports/ ./exports/       # 改模型 → 只重跑这层
```

**改 Python 代码重建只需 3 秒**，不重装依赖。

### 4. 网络拓扑的现实

你的部署环境：

```text
Mac → 阿里云中转 → WSL2 → Docker
```

**WSL2 是 NAT 网络**，Mac 直接访问不到。三种解法：

| 方案 | 复杂度 | 适用 |
|---|---|---|
| SSH 隧道 | ⭐ | 开发 / 调试 |
| 端口转发 | ⭐⭐ | 局域网 |
| mirrored 网络 | ⭐⭐ | Windows 11 |

**开发时用 SSH 隧道最省事**。

### 5. 企业里的部署方式

| 场景 | 方式 |
|---|---|
| 开发 / 调试 | SSH 隧道 |
| 测试环境 | 内网 IP |
| 生产环境 | Nginx / Traefik + HTTPS + 域名 |

**核心概念**：容器化 → 不可变基础设施 → 一次构建，到处运行。

---

## 三、项目现状

### 完整闭环

```text
✅ 训练（4070 / CUDA）
✅ 导出（ONNX）
✅ 部署（Docker）
✅ 上线（docker compose up）
✅ 调用（Mac 通过 SSH 隧道）
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
| 7 | **DevOps 闭环** | ✅（23-25） |

### 质量维度

| 维度 | 工具 | 状态 |
|---|---|---|
| 代码风格 | ruff | ✅ |
| 类型安全 | mypy（严格） | ✅ |
| 提交规范 | pre-commit | ✅ |
| 测试 | pytest（22 passed） | ✅ |
| 可观测 | TensorBoard | ✅ |
| 可复现 | seed + snapshot | ✅ |
| 文档 | README | ✅ |
| **部署** | **Docker + FastAPI** | ✅ |

---

## 四、本阶段踩的坑

### 坑 1：`numpy<2` 必须加引号

**症状**：

```text
/bin/sh: 1: cannot open 2: No such file
```

**原因**：shell 把 `<` 解释为重定向。

**解**：

```dockerfile
"numpy<2"       # ✅
numpy<2         # ❌
```

### 坑 2：中文引号

**症状**：

```text
/bin/sh: 1: cannot open 2”: No such file
```

**原因**：用了中文引号 `“ ”`，shell 不认。

**教训**：Dockerfile / Makefile / shell 脚本里，**所有引号必须是英文 `"`**。

### 坑 3：numpy 2.x 与 torch 2.2.2 ABI 冲突

**症状**：

```text
RuntimeError: Numpy is not available
```

**原因**：torch 2.2.2 用 numpy 1.x 编译，Docker 里默认装了 2.5.2。

**解**：Dockerfile 里 pin `"numpy<2"`。

### 坑 4：Docker 层缓存

**症状**：改了 Dockerfile 但 numpy 还是 2.x。

**原因**：Docker 复用了缓存。

**解**：`docker compose build --no-cache`。

### 坑 5：ONNX 文件不同步

**症状**：4070 上 `ls exports/` 空。

**原因**：`exports/` 在 `.gitignore` 里，`git pull` 不带过来。

**解**：
- 4070 上本地导出 `make export EXP=mlp`
- 或 `scp` 从 Mac 传过去

### 坑 6：WSL2 NAT 网络

**症状**：Mac 访问不到 4070 上的服务。

**解**：SSH 隧道

```bash
ssh -p 6000 -L 8000:localhost:8000 guinness@47.116.44.23
```

---

## 五、下一步

### DevOps 剩余

- [ ] CI（Gitee Go / Jenkins / GitHub Actions）
- [ ] 分布式训练 DDP
- [ ] 训练镜像 `Dockerfile.train`
- [ ] docker-compose 多服务

### 算法方向

- [ ] ResNet / EfficientNet
- [ ] 学习率 warmup
- [ ] MixUp / CutMix
- [ ] 知识蒸馏

### 架构方向（多模态）

- [ ] 文本模型（Transformer / RNN）
- [ ] 抽象 `train.py` 支持不同 loss / metric
- [ ] 抽象 `data/` 支持不同 batch 结构

---

## 六、一句话

> **本阶段把"训练脚本"变成了"可上线的服务"。**

关键区别：

- 之前：模型是 `.pt` 文件，只有训练脚本能用
- 现在：模型是 HTTP API，任何语言、任何机器都能调用

**从"能训练"到"能交付"。**

---

## 七、未来的方向

你提到要向**多模态**拓展。当前架构的可扩展点：

| 组件 | 可复用 | 需改造 |
|---|---|---|
| `utils/` | ✅ | 无 |
| `config.py` | ⚠️ | 加 NLP 相关字段 |
| `data/` | ⚠️ | 加文本处理 |
| `models/` | ✅ | 加 Transformer / CLIP |
| `training/train.py` | ⚠️ | 抽象 loss / metric |
| `export/` | ✅ | 支持多输入 ONNX |
| `serving/` | ⚠️ | 支持多模态输入 |

**核心改造**：把 `train.py` 的 loss 和 evaluator 抽象成参数化。

---

## 八、附：完整闭环命令

```bash
# 1. 训练（4070）
make train EXP=deep_convnet EPOCHS=10 LR=0.1

# 2. 导出（Mac 或 4070）
make export EXP=deep_convnet

# 3. 部署（WSL2）
docker compose up -d

# 4. 调用（Mac 通过 SSH 隧道）
ssh -p 6000 -L 8000:localhost:8000 guinness@47.116.44.23
# 另开终端
curl -X POST http://localhost:8000/predict -F "file=@cat.jpg"
```

---

*Last updated: 2026-09-16*