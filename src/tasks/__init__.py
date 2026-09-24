"""Task definitions.

A Task bundles together the parts that differ between problem types
(classification, captioning, ...):

- how to compute the training loss from a batch
- how to compute evaluation metrics from a batch
- which metric is used for "best model" selection

Everything else (loop, optimizer, AMP, DDP) is task-agnostic.
"""

from tasks.base import Task
from tasks.classification import ClassificationTask

__all__ = ["Task", "ClassificationTask"]
