"""Task definitions.

A Task bundles together the parts that differ between problem types
(classification, captioning, language modelling, ...):

- how to compute the training loss from a batch
- the per-step lifecycle hooks (attach / step_end) and checkpoint state

Everything else (loop, optimizer, AMP, DDP) is task-agnostic.
"""

from application.tasks.captioning import CaptioningTask
from application.tasks.classification import ClassificationTask
from application.tasks.dino import DINOTask
from application.tasks.lm import LMTask

__all__ = ["CaptioningTask", "ClassificationTask", "DINOTask", "LMTask"]
