# 项目 TODO List

## 🎯 每步的原则

1. 每一步结束时，项目都能跑
2. 不引入不需要的东西
3. 一次只改一个主题
4. 跑通再进下一步
5. 跑通后回一句 **"第 X 步跑通了"** 再继续

## ✅ 已完成（第 1–10 步）

- [x] **第 1 步**：能跑起来（修 import、统一风格）
- [x] **第 2 步**：删教学代码，模型抽成 Module 类
- [x] **第 3 步**：训练循环工程化（每 epoch 评估、保存最优、test 评估）
- [x] **第 4 步**：配置管理（`TrainConfig` + argparse）
- [x] **第 5 步**：日志与输出（`logging` + CSV + JSON）
- [x] **第 6 步**：数据模块化（`transforms.py` + `datasets.py` 注册表）
- [x] **第 7 步**：可复现性（seed + config 快照 + env 快照）
- [x] **第 8 步**：测试（pytest 冒烟测试）
- [x] **第 9 步**：学习率调度 + early stopping + 恢复训练
- [x] **第 10 步**：换 ViT（模型注册表加一行即可）
- [x] **第 11 步**：`pyproject.toml` + `README.md` + `.gitignore`
---

## 📋 待办（第 12–21 步）

### 代码质量

- [ ] **第 12 步**：ruff + mypy + pre-commit
  - [ ] 配置 `ruff`（lint + format）
  - [ ] 配置 `mypy`（类型检查）
  - [ ] 配置 `.pre-commit-config.yaml`
  - [ ] 跑一遍 `ruff check src/` 和 `mypy src/`，修掉问题
- [ ] **第 13 步**：类型注解补齐
  - [ ] 所有函数加参数和返回值类型
  - [ ] `mypy src/` 无报错

### 自动化

- [ ] **第 14 步**：Makefile + GitHub Actions
  - [ ] `Makefile`：`make install / test / lint / format / train`
  - [ ] `.github/workflows/ci.yml`：push 时自动跑 pytest + ruff + mypy

### 可打包

- [ ] **第 15 步**：`src/__init__.py` 暴露公共 API + 版本号
  - [ ] `from my_test_project import MLP, DeepConvNet, ViT`
  - [ ] 版本号与 `pyproject.toml` 对齐

### 训练增强

- [ ] **第 16 步**：数据增强
  - [ ] `train_transform`：RandomCrop + RandomHorizontalFlip
  - [ ] `test_transform`：只归一化
  - [ ] 训练/验证用不同 transform
- [ ] **第 17 步**：混合精度 AMP
  - [ ] `torch.cuda.amp.autocast` + `GradScaler`
  - [ ] `--amp` 命令行开关

### 可观测

- [ ] **第 18 步**：TensorBoard 集成
  - [ ] `SummaryWriter` 记录 loss / acc / lr
  - [ ] `outputs/<exp>_<ts>/` 里加 `tensorboard/`

### 可部署

- [ ] **第 19 步**：Docker
  - [ ] `Dockerfile`（GPU 版 + CPU 版）
  - [ ] `.dockerignore`
  - [ ] 验证：`docker build` + `docker run` 能跑 1 epoch
- [ ] **第 20 步**：分布式训练
  - [ ] `DistributedDataParallel` 支持
  - [ ] `torchrun --nproc_per_node=2` 验证
- [ ] **第 21 步**：模型部署
  - [ ] 导出 ONNX / TorchScript
  - [ ] FastAPI 推理服务（可选）

---


- [ ] **第 12 步**：Makefile（常用命令集中）
- [ ] **第 13 步**：GitHub Actions（CI，先只跑 pytest）
- [ ] **第 14 步**：Docker（环境隔离）
- [ ] **第 15 步**：`src/__init__.py` 暴露 API + 版本号

### 🚧 训练增强阶段

- [ ] **第 16 步**：数据增强（`train_transform` / `test_transform`）
- [ ] **第 17 步**：混合精度 AMP（GPU 2x 加速）
- [ ] **第 18 步**：TensorBoard 集成

### 📦 部署阶段

- [ ] **第 19 步**：模型导出 ONNX / TorchScript
- [ ] **第 20 步**：FastAPI 推理服务（可选）
- [ ] **第 21 步**：分布式训练（DDP，可选，有多卡时做）

### 🎨 打磨阶段（最后做）

- [ ] **第 22 步**：ruff + mypy + pre-commit（一次性清理）
- [ ] **第 23 步**：类型注解补齐