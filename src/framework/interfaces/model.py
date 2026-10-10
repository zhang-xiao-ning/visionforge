"""Model base class.

Every model in `src/models/` inherits from `Model`. This base provides
three things the framework needs in order to train a model generically:

- `from_data`: build the model from a DataBundle
- `param_groups()`: named parameter groups (for freezing / LoRA / multi-lr)
- `initialize()`: load pretrained weights (default: no-op)

All three have defaults that are correct for the common case of
"train everything from scratch". Models override only what they need.

The base lives in `models/` (application), not in `framework/`. The
framework operates on DataBundle and Task; it never needs to know what
a Model *is*.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch.nn as nn

if TYPE_CHECKING:
    from framework.interfaces.bundle import DataBundle


class Model(nn.Module):
    """Base class for all models.

    Subclassing is a convention, not a hard contract: nothing enforces
    that every model inherits from `Model`. But doing so gives the model
    default `from_data` / `param_groups` / `initialize`, and lets contract
    tests iterate over registered models uniformly.
    """

    @classmethod
    def from_data(cls, bundle: "DataBundle") -> Self:
        """Build the model from a DataBundle's `model_init`.

        Default: `cls(**bundle.model_init)`. Override when the model needs
        information from `bundle.extras` or does custom construction.
        """
        return cls(**bundle.model_init)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named parameter groups.

        Default: one group "all" containing every parameter.

        Override for LoRA / freezing / multi-lr scenarios, e.g.:
            {"vision": [...], "projector": [...], "llm_lora": [...]}
        """
        return {"all": list(self.parameters())}

    def initialize(self, checkpoint: Path) -> None:
        """Load pretrained weights from a checkpoint. Default: no-op."""
        return
