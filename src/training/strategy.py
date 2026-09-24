"""Training strategies for single-device and distributed training.

A strategy encapsulates everything that differs between training topologies:

- how the model is wrapped (plain vs DDP)
- how the training/validation data is sampled
- who is allowed to log / save checkpoints (rank 0 only in DDP)

Adding a new topology (FSDP, DeepSpeed, ...) means adding a new subclass,
without touching the rest of the codebase.
"""

import os
from collections.abc import Sized
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler
from torch.utils.data import sampler as torch_sampler
from torch.utils.data.distributed import DistributedSampler


class TrainingStrategy:
    """Interface for training topologies.

    Concrete implementations:
    - SingleDeviceStrategy: one process on CPU / MPS / a single GPU
    - DDPStrategy: N processes via torchrun, one per GPU
    """

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Optionally wrap the model (e.g. DDP) and move it to the right device."""
        raise NotImplementedError

    def make_train_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        """Return the sampler for the training split."""
        raise NotImplementedError

    def make_val_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        """Return the sampler for the validation split."""
        raise NotImplementedError

    def is_main_process(self) -> bool:
        """Whether this process is allowed to log / save.

        In DDP only rank 0 writes logs and checkpoints.
        """
        return True

    def set_epoch(self, epoch: int) -> None:
        """Hook called at the start of every epoch.

        DDP uses it to reshuffle the DistributedSampler.
        """
        return

    def cleanup(self) -> None:
        """Release any resources (e.g. destroy the process group)."""
        return


class SingleDeviceStrategy(TrainingStrategy):
    """Single process, single device (CPU, MPS, or one GPU)."""

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        return model.to(device)

    def make_train_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        n = len(cast(Sized, dataset))
        return torch_sampler.SubsetRandomSampler(range(n))

    def make_val_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        n = len(cast(Sized, dataset))
        return torch_sampler.SubsetRandomSampler(range(n))


class DDPStrategy(TrainingStrategy):
    """Distributed Data Parallel via torchrun.

    Expects RANK, LOCAL_RANK, WORLD_SIZE to be set by torchrun.
    Uses NCCL on CUDA, Gloo otherwise (CPU fallback for testing).
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

        if not dist.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._train_sampler: DistributedSampler | None = None

        if device.type == "cuda":
            torch.cuda.set_device(self.local_rank)

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        if device.type == "cuda":
            model = model.to(self.local_rank)
            return DDP(model, device_ids=[self.local_rank])
        return DDP(model)

    def make_train_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        self._train_sampler = DistributedSampler(dataset, shuffle=True)
        return self._train_sampler

    def make_val_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        return DistributedSampler(dataset, shuffle=False)

    def is_main_process(self) -> bool:
        return self.rank == 0

    def set_epoch(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_strategy(device: torch.device) -> TrainingStrategy:
    """Auto-detect: torchrun sets RANK + WORLD_SIZE, otherwise single device."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return DDPStrategy(device)
    return SingleDeviceStrategy()
