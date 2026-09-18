# Memo 6: CI 与项目更名

> 从本地验证到远端自动化
> 时间：2026-09-18
> 版本：v0.1.0
> 项目名：visionforge

---

## 一、本阶段做了什么（第 26 步）

### 第 26 步：CI 自动化

**目标**：每次 push 到 main，自动跑 lint + type check + test。

**产物**：

```text
scripts/ci.sh                # 平台无关的 CI 脚本
.github/workflows/ci.yml     # GitHub Actions 配置
```

**`scripts/ci.sh` 的关键设计**：

```bash
#!/usr/bin/env bash
set -euo pipefail

uv sync
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest -v
```

**核心理念**：CI 逻辑写在脚本里，**任何平台都只是"调用这个脚本"**。

| 平台 | 怎么用 |
|---|---|
| 本地 | `make ci` |
| GitHub Actions | `bash scripts/ci.sh` |
| GitLab CI | `script: bash scripts/ci.sh` |
| Jenkins | `sh 'bash scripts/ci.sh'` |

**以后换平台，脚本不变。**

**GitHub Actions 配置**：

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.12

      - name: Run CI script
        run: bash scripts/ci.sh
```

**实测结果**：

```text
Status: Success
Total duration: 5m 35s
```

第一次跑（冷缓存）5 分钟，以后会降到 1~2 分钟。

---

## 二、项目更名：`my_test_project` → `visionforge`

### 为什么改名

- `my_test_project` 是占位名，不适合长期
- 项目是 CV 方向，未来可能做**自回归 ViT（看图生文）**
- `visionforge` = vision + forge，"视觉模型锻造厂"
- 好念、好记、不绑死模型

### 改名范围

| 位置 | 改前 | 改后 |
|---|---|---|
| GitHub 仓库名 | `my_test_project` | `visionforge` |
| Gitee 仓库名 | `my_test_project` | `visionforge` |
| `pyproject.toml` 的 `name` | `my-test-project` | `visionforge` |
| `docker-compose.yml` 的 `image` | `my-test-project-serve` | `visionforge-serve` |
| `docker-compose.yml` 的 `container_name` | `my-test-project-api` | `visionforge-api` |
| `src/serving/api.py` 的 `FastAPI(title=...)` | `my_test_project inference` | `visionforge inference` |
| `README.md` 标题 | `My Test Project` | `VisionForge` |
| `src/__init__.py` docstring | — | `VisionForge: ...` |
| 本地目录名 | `my_test_project` | `visionforge` |
| WSL 目录名 | `my_test_project` | `visionforge` |

### 不用改的

- 所有 `import`：用的是 `from models.xxx` / `from data.xxx`，**基于 `src` 目录**，和顶层包名无关
- 代码逻辑
- 测试
- `Makefile`（用相对路径）
- `utils/` / `training/` / `export/` / `serving/`

### 教训

**项目命名时的 import 风格，影响改名成本。**

你用的是**脚本式 import**（`from models.xxx`），所以改名几乎零成本。

如果用的是**包式 import**（`from visionforge.models.xxx`），改名就要全项目替换。

**这是一个隐含的架构决策**。

---

## 三、从 Gitee Go 切到 GitHub Actions

### 为什么切

| 维度 | Gitee Go | GitHub Actions |
|---|---|---|
| 免费额度 | 200 分钟（试用） | **无限（公开仓库）** |
| 生态 | 小 | **大** |
| Action 市场 | 无 | **丰富** |
| 学习价值 | 低 | **业界标准** |
| 国内网络 | 快 | 慢（但 runner 在境外，跑得快） |

### 迁移方式

1. 代码主仓库：**Gitee（origin）**
2. CI 仓库：**GitHub（github remote）**
3. Gitee 作为日常 push 目标
4. GitHub 用于跑 CI + 展示

**这样两边的好处都要到了。**

### 教训

**CI 脚本与平台无关**，所以迁移成本几乎为零。

如果一开始就把 CI 逻辑写在 `.github/workflows/*.yml` 里，迁移到 Jenkins 就要重写。

---

## 四、本阶段核心收获

### 1. CI 的三层结构

```text
第 1 层：scripts/ci.sh        ← 逻辑
第 2 层：make ci              ← 本地入口
第 3 层：.github/workflows/    ← 平台触发
```

**职责分明**：

- `ci.sh` 定义"跑什么"
- `make ci` 让本地能手动跑
- Actions 定义"什么时候跑"

### 2. 本地 CI vs 远端 CI

| | 本地 | 远端 |
|---|---|---|
| 入口 | `make ci` | `git push` |
| 触发 | 手动 | 自动 |
| 目的 | 提交前自检 | 强制把关 |
| 覆盖 | 自己的代码 | 所有人 |

**pre-commit + 本地 CI + 远端 CI = 三层防线**。

### 3. 为什么要两个 remote

```bash
origin   → Gitee（主仓库）
github   → GitHub（CI + 展示）
```

**原因**：

- Gitee 国内访问快，日常开发舒服
- GitHub Actions 免费、生态好，CI 用它
- 一个本地仓库，push 到两个远端

**指令**：

```bash
git push origin main     # 推 Gitee
git push github main     # 推 GitHub（触发 CI）
```

### 4. 徽章的意义

README 顶部的 CI 徽章：

```markdown
![CI](https://github.com/zhang-xiao-ning/visionforge/actions/workflows/ci.yml/badge.svg)
```

**作用**：

- 一眼看出项目是否健康
- 别人 clone 前能先看 CI 状态
- **是项目"专业度"的信号**

---

## 五、项目现状

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
| 8 | **CI 自动化** | ✅（26） |

### 完整链路

```text
代码提交 ──→ pre-commit（本地）
              ↓
          git push
              ↓
          GitHub Actions（远端）
              ↓
          装 uv + Python
              ↓
          bash scripts/ci.sh
              ↓
   ┌──────────┼──────────┬──────────┐
   ↓          ↓          ↓          ↓
  ruff      format      mypy      pytest
```

### 质量维度

| 维度 | 工具 | 状态 |
|---|---|---|
| 代码风格 | ruff | ✅ |
| 类型安全 | mypy（严格） | ✅ |
| 提交规范 | pre-commit | ✅ |
| 测试 | pytest（22 passed） | ✅ |
| 远端 CI | GitHub Actions | ✅ |
| 可观测 | TensorBoard | ✅ |
| 可复现 | seed + snapshot | ✅ |
| 文档 | README | ✅ |
| 部署 | Docker + FastAPI | ✅ |

---

## 六、本阶段踩的坑

### 坑 1：Gitee Go 是付费服务

**症状**：Gitee Go 页面显示"免费尝鲜 200 分钟"，之后要付费。

**原因**：Gitee 把 CI 当增值服务卖。

**解**：切到 GitHub Actions（公开仓库完全免费）。

### 坑 2：`.workflow/` 和 `.github/workflows/` 混淆

**症状**：

- Gitee Go 用 `.workflow/ci.yml`
- GitHub Actions 用 `.github/workflows/ci.yml`

**教训**：**不同平台配置文件放在不同目录**，不能混。

### 坑 3：改名时目录名和仓库名不一致

**症状**：本地目录叫 `my_test_project`，GitHub 仓库叫 `visionforge`。

**解**：

```bash
mv ~/Documents/pyprojects/my_test_project ~/Documents/pyprojects/visionforge
```

PyCharm 里也要重新打开项目。

### 坑 4：CI 首次跑 5m35s

**原因**：冷缓存，要装 torch 等依赖。

**解**：第二次会快（依赖缓存生效）。

**可选优化**：

```yaml
- name: Cache uv
  uses: actions/cache@v4
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('uv.lock') }}
```

**先不加**，等真觉得慢了再说。

---

## 七、下一步

### DevOps 剩余

- [ ] 分布式训练 DDP
- [ ] 训练镜像 `Dockerfile.train`
- [ ] docker-compose 多服务
- [ ] 模型版本管理（MLflow / DVC）

### 算法方向

- [ ] ResNet / EfficientNet
- [ ] 学习率 warmup
- [ ] MixUp / CutMix
- [ ] 知识蒸馏

### 多模态方向

- [ ] 自回归 ViT（看图生文）
- [ ] 文本模型（Transformer / RNN）
- [ ] 抽象 `train.py` 支持不同 loss / metric

### 工程质量

- [ ] 版本号单一来源（`importlib.metadata`）
- [ ] 集成测试（训练 1 步不报错）
- [ ] 回归测试（精度不低于基线）

---

## 八、一句话

> **本阶段让项目从"能开发"变成"能交付"。**

关键区别：

- 之前：改完代码靠自觉跑测试
- 现在：**push 到 GitHub 自动验证**，红了会通知

**从"能开发"到"能保证质量"。**

---

## 九、附：本阶段完整命令

```bash
# 1. 本地跑 CI
make ci

# 2. 推到 Gitee
git push origin main

# 3. 推到 GitHub（触发 CI）
git push github main

# 4. 看 CI 状态
open https://github.com/zhang-xiao-ning/visionforge/actions
```

---

*Last updated: 2026-09-18*