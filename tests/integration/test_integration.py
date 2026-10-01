from experiment.runner import ExperimentRunner
from experiment.spec import TrainConfig
from training.strategy import SingleDeviceStrategy

# test_train_one_epoch_on_real_data
runner = ExperimentRunner(
    experiment_name="vit",
    config=TrainConfig(epochs=1),
    strategy=SingleDeviceStrategy(),
    num_train=32,
)

# test_runner_saves_and_resumes 同理，两处都改
