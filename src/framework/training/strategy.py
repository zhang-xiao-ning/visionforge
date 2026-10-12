"""Training strategies for single-device and distributed training.

A strategy encapsulates everything that differs between training topologies:

- how the model is wrapped (plain vs DDP)
- how the training/validation data is sampled
- which device this process uses
- who is allowed to save checkpoints (rank 0 only in DDP)

Adding a new topology (FSDP, DeepSpeed, ...) means adding a new subclass,
without touching the rest of the codebase.
"""

import os
from collections.abc import Sized
from pathlib import Path
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

    #: Process rank. 0 for single-device; set by DDPStrategy.
    rank: int = 0

    @property
    def device(self) -> torch.device:
        """The device this process trains on."""
        raise NotImplementedError

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Optionally wrap the model (e.g. DDP) and move it to this device."""
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

    def broadcast_model(self, model: nn.Module) -> None:
        """Bring all ranks' parameters and buffers into agreement.

        Single device: no-op.
        DDP: rank 0's full state_dict is broadcast to all ranks. Called
        after `MetricTracker.restore()`, which only has data on rank 0,
        and covers both parameters (weight/bias) and buffers
        (running_mean/running_var/num_batches_tracked).
        """
        return

    def save(self, path: Path, payload: Any) -> None:
        """Save a checkpoint. Only the main process actually writes."""
        if self.is_main_process():
            torch.save(payload, path)

    def sync_stop(self, stop: bool) -> bool:
        """Return the stop flag that all ranks agree on.

        Single device: return `stop` unchanged.
        DDP: rank 0's value is broadcast to all ranks; the other ranks'
        input values are ignored. This keeps early stopping in sync —
        otherwise non-main ranks keep training after rank 0 breaks,
        and the next allreduce hangs.
        """
        return stop

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of every epoch."""
        return

    def cleanup(self) -> None:
        """Release any resources (e.g. destroy the process group)."""
        return


class SingleDeviceStrategy(TrainingStrategy):
    """Single process, single device (CPU, MPS, or one GPU)."""

    def __init__(self, device: torch.device | None = None) -> None:
        if device is None:
            from framework.runtime import get_device

            device = get_device()
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self._device)

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

    def __init__(self) -> None:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._train_sampler: DistributedSampler | None = None

        if torch.cuda.is_available():
            self._device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        else:
            self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self._device)
        if self._device.type == "cuda":
            return DDP(model, device_ids=[self.local_rank])
        return DDP(model)

    def make_val_sampler(self, dataset: Dataset[Any]) -> Sampler[Any]:
        # Validation runs only on rank 0 and must cover the FULL set,
        # so we do NOT use DistributedSampler here.
        return torch_sampler.SequentialSampler(cast(Sized, dataset))

    def is_main_process(self) -> bool:
        return self.rank == 0

    def broadcast_model(self, model: nn.Module) -> None:
        # state_dict() yields references (not copies), so broadcasting
        # in place writes back into the module's parameters and buffers.
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)

    def on_epoch_start(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

    def sync_stop(self, stop: bool) -> bool:
        tensor = torch.tensor([int(stop)], device=self._device)
        dist.broadcast(tensor, src=0)
        return bool(tensor.item())

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_strategy() -> TrainingStrategy:
    """Auto-detect: torchrun sets RANK + WORLD_SIZE, otherwise single device."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return DDPStrategy()
    return SingleDeviceStrategy()
