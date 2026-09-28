# TODO

## 短期（按优先级）
## 短期（按优先级）

- [ ] **评估层**：BLEU / CIDEr / METEOR / ROUGE（半天）
- [ ] **LLaVA 推理脚本**（1 天，撞 framework 缺口）
- [ ] **评估层**：BLEU / CIDEr / METEOR / ROUGE
  - 从 `CaptioningTask.eval_step` 输出
  - 依赖 `pycocoevalcap` 或 `nltk`
- [ ] **LLaVA 推理脚本**（撞 framework 缺口）
  - `scripts/llava_infer.py`：加载 HF 权重（4bit 量化）
  - 不改 framework，先跑通推理
- [ ] **Logger 级别系统**（NONE / ERROR / INFO / FLOW / DEBUG）
  - 环境变量 `VISIONFORGE_LOG_LEVEL` 控制
  - 用途：NaN 排查、静音跑 benchmark
  - 详见 `architecture-notes.md`

## 中期（撞到需求再做）

- [ ] 加载预训练权重（CLIP / HF）
- [ ] KV cache（生成式推理）
- [ ] 量化（int8 / 4bit / bnb）
- [ ] 参数冻结 + LoRA 骨架
- [ ] 多阶段训练（对齐 + 指令微调）
- [ ] AMP 实测（大模型）

## 长期（做完整 LLaVA / LM 训练时）

- [ ] 数据缓存（tokenized 结果存盘）
- [ ] packed sequence（多句拼接）
- [ ] vLLM / TGI 集成
- [ ] FSDP / DeepSpeed
- [ ] 多机多卡（跨机 DDP）

## 工程细节

- [ ] `from_data` 优化（`DatasetInfo` 合并到 `DataBundle`）
- [ ] Trainer 类重构（等第 3 类任务撞出需求）
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