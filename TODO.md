# TODO

## DevOps 阶段

- [x] CI（GitHub Actions）
  - [x] 自动跑 pytest
  - [x] 自动跑 ruff + mypy
  - [x] README 加 CI 徽章（待加）
- [x] Docker（推理镜像）
  - [x] `docker/Dockerfile.serve`
  - [x] `.dockerignore`
- [x] Docker（训练镜像）
  - [x] `docker/Dockerfile.train`（GPU 版）
  - [x] 支持挂载 `datasets/` 和 `checkpoints/`
  - [x] 验证：`docker run` 训练 1 epoch
- [x] 分布式训练 DDP
  - [x] `DistributedDataParallel` 支持
  - [x] `torchrun --nproc_per_node=1` 验证（4070 上跑通）
- [ ] docker-compose 多服务
  - [ ] 加数据库 / 前端等其他服务
- [ ] docker-compose 多服务
  - [ ] 加数据库 / 前端等其他服务

## 打磨阶段

- [x] 版本号单一来源（`importlib.metadata`）
- [x] 集成测试（训练 1 步不报错）
- [x] 回归测试（精度不低于基线）
- [ ] CHANGELOG.md（Keep a Changelog 格式）
- [ ] 覆盖率报告（pytest-cov）
- [ ] CI 矩阵（Python 3.12 + 3.13）
- [ ] API 文档（mkdocs / pdoc）
- [ ] `setuptools-scm` 从 git tag 自动生成版本（可选）

## 算法方向

- [ ] 学习率 warmup
- [ ] MixUp / CutMix
- [ ] RandAugment
- [ ] 知识蒸馏
- [ ] ResNet / EfficientNet
- [ ] 更多数据集（MNIST / FashionMNIST）

## 可选

- [ ] TensorBoard 远端服务器
- [ ] WandB 集成
- [ ] MLflow 实验管理