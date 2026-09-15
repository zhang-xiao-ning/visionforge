# TODO

## DevOps 阶段（后做）

- [ ] Docker
  - [ ] `Dockerfile`（CPU / GPU 版）
  - [ ] `.dockerignore`
  - [ ] 验证：`docker build` + `docker run` 能跑 1 epoch
- [ ] CI（Jenkins / GitHub Actions 二选一）
  - [ ] 自动跑 pytest
  - [ ] 自动跑 ruff + mypy
  - [ ] README 加 CI 徽章
- [ ] 模型导出 ONNX / TorchScript
  - [ ] 导出脚本 `scripts/export_onnx.py`
  - [ ] 验证导出模型推理结果与 PyTorch 一致
- [ ] 分布式训练 DDP
  - [ ] `DistributedDataParallel` 支持
  - [ ] `torchrun --nproc_per_node=2` 验证

## 打磨阶段（最后做）

- [ ] 类型注解补齐（`mypy src/` 无报错）
- [ ] README 完善（架构图、实验对比表）
- [ ] 版本号单一来源（`importlib.metadata`）
- [ ] `setuptools-scm` 从 git tag 自动生成版本（可选，更进阶）

## 可选

- [ ] FastAPI 推理服务
- [ ] TensorBoard 远端服务器
- [ ] 学习率 warmup
- [ ] MixUp / CutMix
- [ ] RandAugment
- [ ] 知识蒸馏