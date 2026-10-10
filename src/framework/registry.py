from collections.abc import Iterator, MutableMapping

from framework.experiment.spec import Experiment


class ExperimentRegistry(MutableMapping[str, Experiment]):
    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}

    def register(self, name: str, experiment: Experiment) -> None:
        self._experiments[name] = experiment

    def __getitem__(self, name: str) -> Experiment:
        return self._experiments[name]

    def __setitem__(self, name: str, experiment: Experiment) -> None:
        self._experiments[name] = experiment

    def __delitem__(self, name: str) -> None:
        del self._experiments[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._experiments)

    def __len__(self) -> int:
        return len(self._experiments)


EXPERIMENTS = ExperimentRegistry()
