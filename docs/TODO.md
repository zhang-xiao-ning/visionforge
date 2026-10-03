# TODO

## 短期（按优先级）

- [ ] **LLaVA-A 推理脚本**（1 天）
  - `scripts/llava_infer.py`：加载 HF 权重（4bit 量化）
  - 不改 framework，先跑通推理
  - `transformers` 只在 `scripts/`
- [ ] **评估层扩展**：CIDEr / METEOR / ROUGE
  - 依赖 `pycocoevalcap` 或 `nltk`
  - 与 BLEU4 同一模式（Metric + registry）

## 中期（撞到需求再做）

- [ ] 加载预训练权重（CLIP / HF）
- [ ] KV cache（生成式推理）
- [ ] 量化（int8 / 4bit / bnb）
- [ ] 参数冻结 + LoRA 骨架
- [ ] 多阶段训练（LLaVA 对齐 + 指令微调）
- [ ] AMP 实测（大模型）

## 长期（做完整 LLaVA / LM 训练时）

- [ ] 数据缓存（tokenized 结果存盘）
- [ ] packed sequence（多句拼接）
- [ ] vLLM / TGI 集成
- [ ] FSDP / DeepSpeed
- [ ] 多机多卡（跨机 DDP）

## 工程细节

- [ ] `DataBundle` / `EvalBundle` 强类型化（等 LLaVA 接入后字段稳定）
- [ ] `Trainer` 类重构（等第 3 类任务撞出需求）
- [ ] `from_data` 优化（`DatasetInfo` 合并到 `DataBundle`）
- [ ] `CHANGELOG.md`（Keep a Changelog 格式）
- [ ] 覆盖率报告（pytest-cov）
- [ ] CI 矩阵（Python 3.12 + 3.13）

## 可选

- [ ] TensorBoard 远端服务器
- [ ] WandB 集成
- [ ] MLflow 实验管理
- [ ] mkdocs / pdoc API 文档

## 已排除

- [x] HF Tokenizer 适配（不想绑 HF 生态）
- [x] Serving `/caption` 端点（会被 vLLM 替代，死路）
- [x] 评估层集成到 Task（已剥离为 Metric，见 memo-12）
- [x] `Task.stages`（已移到 `Experiment` 配置，见 memo-12）