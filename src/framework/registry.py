from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping
from typing import TYPE_CHECKING

from framework.experiment.spec import Experiment

if TYPE_CHECKING:
    from framework.interfaces.bundle import DataBundle, DataContext, EvalBundle


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


class DatasetRegistry(MutableMapping[str, Callable]):
    """Registry of dataset builders: name -> build_bundle(ctx).

    Mirrors ExperimentRegistry: framework provides the mechanism, the
    application layer fills the content. The framework never imports
    concrete datasets.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Callable] = {}

    def register(self, name: str, builder: Callable) -> None:
        self._builders[name] = builder

    def build(self, name: str, ctx: DataContext) -> tuple[DataBundle, EvalBundle | None]:
        if name not in self._builders:
            raise ValueError(f"Unknown dataset: {name}")
        return self._builders[name](ctx)

    def __getitem__(self, name: str) -> Callable:
        return self._builders[name]

    def __setitem__(self, name: str, builder: Callable) -> None:
        self._builders[name] = builder

    def __delitem__(self, name: str) -> None:
        del self._builders[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def __len__(self) -> int:
        return len(self._builders)


EXPERIMENTS = ExperimentRegistry()
DATASETS = DatasetRegistry()
