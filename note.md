visionforge Architecture
========================

原则
----
- 单向依赖：application -> framework（禁止反向）
- 机制在框架，内容在应用
- 边界靠 framework/__init__.py 声明 + 边界测试保护

目录结构
--------
src/
  framework/                引擎
    __init__.py             公开 API（边界声明）
    runtime.py              常量 + device
    interfaces/             Task / Model / Metric / DataBundle
    training/               train / hooks / strategy / tracker
    metrics/                通用：Accuracy / CrossEntropy / Perplexity / corpus_bleu
    experiment/             spec / loader / artifacts / runner
    registry.py             注册机制（空表）
    cli.py                  argparse + YAML
    utils/                  logger / path / seed / env

  application/              内容
    models/                 mlp / vit / mamba / captioning
    datasets/               cifar10 / flickr8k / transforms / tokenizers
    tasks/                  classification / captioning
    metrics/                bleu4
    export/                 ONNX 导出的应用逻辑
    serving/                FastAPI
    registry.py             往框架注册表填内容
    main.py                 入口

关键接口
--------
Task：
  train_step(model, batch, device, dtype) -> Tensor

Model：
  from_data(bundle) -> Self
  param_groups() -> dict[str, list[Parameter]]
  setup()                       # LoRA 注入等
  on_step_end(optimizer, step)  # EMA

Metric：
  name: str
  higher_is_better: bool
  run_every_n_epochs: int | None
  evaluate(model, loader, device, dtype) -> float

DataBundle：
  loader_train / loader_val / loader_test
  model_init / task_init / extras

Experiment：
  model / task / data / metrics / primary_metric / config / category
  build_optimizer: Callable | None
  build_scheduler: Callable | None
  export_spec: Callable | None

数据流
------
main.py [A]
  registry 填内容
  cli.build_run -> RunParams
  ExperimentRunner(experiment, config, ...) [F]
    DATASETS.build(...) -> DataBundle [F 接口]
    experiment.model.from_data(bundle) [A]
    experiment.task.from_data(bundle) [A]
    metrics[i].from_data(bundle) [F/A]
    train() [F]

加新东西
--------
模型            models/xxx.py + registry 1 行            2 处
数据集          datasets/xxx.py + registry 1 行          2 处
任务            tasks/xxx.py + registry 1 行             2 处
自定义 optimizer  Experiment(build_optimizer=...)         1 处
自定义导出        Experiment(export_spec=...)              1 处
框架永不动。

演化
----
阶段 1（现在）：门面 framework/__init__.py + 边界测试
阶段 2：物理分离 framework/ 和 application/
阶段 3：framework 独立成 pip 包

判断标准
--------
只依赖 torch/stdlib/其他框架代码   -> framework
依赖具体模型/数据集/任务          -> application
