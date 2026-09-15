# Memo 3: 本地开发环境成型

> 从"能跑"到"能安心开发"
> 时间：2026-09-15
> 版本：v0.1.0

---

## 一、本阶段做了什么（第 16-19 步）

### 第 16 步：TensorBoard 集成

- `SummaryWriter` 写 loss / val_acc / lr
- 每次运行在 `outputs/<exp>_<timestamp>/` 下生成 event 文件
- `make board` 一键启动
- 浏览器打开 `http://localhost:6006`

**知识点**：

- `torch.utils.tensorboard.SummaryWriter`
- `add_scalar(tag, value, step)`
- `writer.close()` 显式关闭
- TensorBoard 默认 5 秒刷新，长时间训练才能看到"实时"

**学到的**：

- TensorBoard 是 PyTorch 生态默认的可视化工具
- WandB 是更现代、更漂亮的替代，个人免费
- 两者概念 1:1 对应，以后切换只要 10 分钟

---

### 第 17 步：ruff 代码风格清理

- 删除坏文件 `src/utils/init.py`
- `ruff check --fix`：import 排序、未使用 import
- `ruff check --fix --unsafe-fixes`：`%` → f-string
- `ruff format`：长列表、函数调用自动换行
- 结果：`All checks passed!`，20 个文件改动

**知识点**：

- `I001`：import 排序
- `F401`：未使用 import
- `UP031`：旧式 `%` 格式化
- `ruff` 一个工具替代 `flake8` + `isort` + `black`
- `--unsafe-fixes` 的含义：可能改变行为的修复

**学到的**：

- **教学代码的 `%` 格式化不是错，但现代 Python 用 f-string**
- ruff 自动修 90%，剩下 10% 手动

---

### 第 18 步：pre-commit

- `.pre-commit-config.yaml` 使用 **local 模式**
- `uv run ruff check` / `ruff format` / `mypy` 作为 hook
- 每次 `git commit` 自动跑
- 不合规范的代码**进不了 git**

**知识点**：

- pre-commit 默认从 GitHub 下载 hook 仓库 + 隔离环境
- **国内网络差**：GitHub clone 慢，PyPI 下载 torch 2GB 慢
- 清华源只对 `uv` 生效，不影响 pre-commit
- `language: system` + `repo: local` 完全绕过网络
- `pass_filenames: false` 让命令跑全目录而不是只跑改动文件

**学到的**：

- **pre-commit 隔离环境的代价：每个 hook 重新下依赖**
- 国内最好用 local 模式
- 用 `uv run` 复用项目环境，版本由 `uv.lock` 统一控制

**验证**：

```bash
echo "import os" >> src/config.py
git add src/config.py
git commit -m "test hook"
# → ruff 自动删 import os，拒绝 commit
```

输出：

```text
ruff.....................................................................Failed
- hook id: ruff
- files were modified by this hook

Found 1 error (1 fixed, 0 remaining).

ruff-format..............................................................Passed
mypy.....................................................................Passed
```

**验证通过**：hook 正确拦截了不合规的代码。

---

### 第 19 步：`make install-hooks`

- Makefile 加 `install-hooks` 目标
- `make help` 里加说明
- README 里加安装说明
- 新人 clone 后一句命令装 hook

**Makefile 改动**：

```makefile
.PHONY: help install install-hooks test lint format train board clean

install-hooks:
	uv run pre-commit install
	@echo "Pre-commit hook installed."
```

**验证**：

```bash
$ make help
Available targets:
  make install              uv sync
  make install-hooks        Install git pre-commit hooks
  make test                 uv run pytest
  ...

$ make install-hooks
uv run pre-commit install
pre-commit installed at .git/hooks/pre-commit
Pre-commit hook installed.
```

**学到的**：

- 新人友好的项目要把"环境准备"也脚本化
- `make` 是**项目级快捷键**，比记长命令更好

---

## 二、本阶段的核心收获

### 1. 工具选择的国内现实

| 工具 | 国外体验 | 国内体验 |
|---|---|---|
| pre-commit 默认模式 | 秒装 | 几分钟到失败 |
| pre-commit local 模式 | 秒装 | 秒装 |
| pip install | 正常 | 慢，需要清华源 |
| uv sync + 清华源 | 正常 | 快 |
| GitHub clone | 正常 | 慢 / 失败 |

**结论**：国内开发 = **能用本地就用本地，能走清华源就走清华源**。

### 2. pre-commit 的哲学

> **规范代码进不了 git。**

- 老一代：靠 code review
- 现代：靠 pre-commit hook

**代价**：新人第一次 commit 会被"打回"几次。  
**收益**：代码风格问题永远不是 review 的负担。

### 3. 里程碑分界

| 里程碑 | 范围 | 状态 |
|---|---|---|
| 1 | 核心骨架 | ✅ 完成（1-10） |
| 2 | 项目外观 | ✅ 完成（11-13） |
| 3 | 训练增强 | ✅ 完成（14-15） |
| 4 | 可观测 | ✅ 完成（16） |
| 5 | 本地开发环境 | ✅ 完成（17-19） |
| 6 | 部署与运维 | ⏳ 待做 |

**本地开发环境已经完整**：写代码 → 自动 lint → 自动格式化 → 类型检查 → 测试 → 可视化。

### 4. 类型注解的取舍

- `mypy src/` 现在全过——因为**没开严格模式**
- 开启 `disallow_untyped_defs` 后会报几十个"缺少类型注解"
- **暂不开启**，作为"打磨阶段"的一次性工作

**这是工程判断**：不是技术做不到，是**现在不该做**。

### 5. 三份 memo 的分工

| 文件 | 内容 | 时间线 |
|---|---|---|
| `memo-1.md` | 早期总结草稿（已废弃） | 第 1-10 步 |
| `memo-2.md` | 项目从零到骨架 | 第 1-15 步 |
| `memo-3.md` | 本地开发环境成型 | 第 16-19 步 |

**未来继续写 memo-4、memo-5 ……**

---

## 三、项目现状

### 目录结构

```text
my_test_project/
├── src/
│   ├── __init__.py              # 公开 API
│   ├── py.typed
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   ├── transforms.py
│   │   ├── cifar10.py
│   │   └── datasets.py
│   ├── models/
│   │   ├── mlp.py
│   │   ├── shallow_convnet.py
│   │   ├── deep_convnet.py
│   │   └── vit.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluator.py
│   └── utils/
│       ├── path.py
│       ├── logger.py
│       ├── seed.py
│       └── env.py
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_transforms.py
│   └── test_data.py
├── .pre-commit-config.yaml      # pre-commit 配置
├── Makefile                     # 项目快捷键
├── pyproject.toml
├── README.md
├── TODO.md
├── COMPLETED.md
├── memo-1.md
├── memo-2.md
├── memo-3.md                    # 本次
├── .gitignore
├── pack.sh
└── dump_code.sh
```

### 可用命令

```bash
make install          # uv sync
make install-hooks    # 装 pre-commit hook
make test             # pytest
make lint             # ruff + mypy
make format           # ruff format
make train EXP=mlp EPOCHS=5
make board            # TensorBoard
make clean            # 清缓存
```

### 开发流程

```text
写代码
   ↓
git add
   ↓
git commit
   ↓
pre-commit 自动跑 ruff + mypy
   ↓
通过？ ──是──→ commit 成功
   │
   否
   ↓
自动修复 → 重新 git add → 再 commit
```

### 测试覆盖

```text
10 passed, 1 skipped
```

跳过的是 `test_build_loaders_real`（需要真实数据）。

### 实验列表

| 实验名 | 模型 | 默认 LR | 备注 |
|---|---|---|---|
| `mlp` | Two-layer MLP | 1e-2 | 基线 |
| `shallow_convnet` | 2 层 ConvNet | 1e-2 | |
| `deep_convnet` | 5 层 ConvNet + BN | 0.1 | CIFAR-10 上 ~74% |
| `vit` | 小号 ViT | 3e-4 | patch=4, dim=192, depth=6 |

---

## 四、下一步

### 候选方向

**A. DevOps（重）**

- CI（Jenkins / Gitee Go / GitHub Actions）
- Docker
- 分布式训练 DDP
- 模型导出 ONNX / TorchScript
- FastAPI 推理服务

**B. 打磨（轻）**

- 类型注解补齐（`disallow_untyped_defs`）
- README 完善（架构图、实验对比表）
- 补单元测试（trainer / evaluator）
- 数据增强进阶（MixUp / CutMix）

**C. 算法方向**

- 换数据集（MNIST / FashionMNIST）
- 加 ResNet / EfficientNet
- 学习率 warmup
- 知识蒸馏
- 量化

### 判断标准

| 目标 | 推荐 |
|---|---|
| 学部署、运维 | A |
| 学工程质量 | B |
| 学模型、算法 | C |
| 找工作（CV 方向） | A + C |
| 找工作（MLOps 方向） | A + B |

---

## 五、一句话

> **本阶段把"能跑"变成了"能安心开发"。**

关键区别：

- **之前**：改代码怕破坏风格、怕类型错、怕忘跑测试
- **现在**：pre-commit 自动拦截，改代码零心理负担

**这就是本地开发环境成熟的标准。**

---

## 六、附录：本阶段遇到的主要坑

### 坑 1：pre-commit 从 GitHub 下载巨慢

**症状**：`Initializing environment for https://github.com/...` 卡住几分钟。

**原因**：默认模式下 pre-commit 从 GitHub clone hook 仓库，在国内网络下慢。

**解决**：用 `repo: local` + `language: system` 完全绕过。

### 坑 2：mypy hook 需要重新下载 torch

**症状**：pre-commit 初始化时准备下载 2GB 的 torch。

**原因**：mypy hook 有独立的隔离环境，需要 torch 的类型信息。

**解决**：同上，本地模式。

### 坑 3：`.idea/` 一直在 `git status` 里

**症状**：即使 `.gitignore` 写了 `.idea/`，`git status` 还显示 `.idea/` 的改动。

**原因**：`.gitignore` 只对"未追踪"文件生效，`.idea/` 已被追踪。

**解决**：

```bash
git rm -r --cached .idea
git commit -m "chore: stop tracking .idea"
```

### 坑 4：`git diff --stat` 停在 `:` 不动

**症状**：命令输出后停在 `:` 提示符，终端不动。

**原因**：git 用了分页器 `less`。

**解决**：按 `q` 退出。

### 坑 5：ruff-format 第一次跑"失败"

**症状**：`pre-commit run --all-files` 里 `ruff-format Failed`。

**原因**：它自动改了文件，然后报"失败"让你看到改动。

**解决**：`git add -A` 后重跑，会 `Passed`。

---

*Last updated: 2026-09-15*