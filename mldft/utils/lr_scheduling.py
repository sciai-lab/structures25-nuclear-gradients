"""Helper functions for configuring learning rate scheduling using Hydra."""

from typing import Callable, List

from torch.optim.lr_scheduler import ChainedScheduler, SequentialLR
from torch.optim.optimizer import Optimizer


def chain_schedulers(optimizer: Optimizer, schedulers: List[Callable]) -> ChainedScheduler:
    """
    Chain multiple schedulers together, using :class:`torch.optim.lr_scheduler.ChainScheduler`.
    The point of this wrapper function is to make it easier to use in hydra config files:
    The optimizer has to be passed only once, in the same way as for a single scheduler.

    See ``configs/ml/model/schedulers/warmup_linear.yaml`` for an example.

    Args:
        optimizer: The optimizer to be scheduled.
        schedulers: A list of partially initialized schedulers, mapping optimizers to
            :class:`~torch.optim.lr_scheduler.LRScheduler`.

    Returns:
        A chained scheduler.

    """
    return ChainedScheduler([scheduler(optimizer) for scheduler in schedulers])


def sequential_scheduler(
    optimizer: Optimizer,
    schedulers: List[Callable],
    milestones: List[int],
    steps_per_epoch: int | None = None,
    step_attr: List | None = None,
) -> SequentialLR:
    """Chain multiple schedulers together, using :class:`torch.optim.lr_scheduler.SequentialLR`.
    This wrapper also handles converting epochs to steps for all attributes (if steps_per_epoch is
    provided) given in the step_attr list. Timings in the configs should thus always be provided in
    terms of epochs, even when "step" is used as the scheduler interval. If you want to use this as
    a step scheduler, make sure to set model.scheduler_interval = "step".

    Args:
        optimizer: The optimizer to be scheduled.
        schedulers: A list of partially initialized schedulers, mapping optimizers to
            :class:`~torch.optim.lr_scheduler.LRScheduler`.
        milestones: Iterations at which schedulers are swapped
        steps_per_epoch: Estimated number of training steps during each epoch.
        step_attr: Attributes which should be converted from epochs to steps.

    Returns:
        A sequential scheduler
    """

    if steps_per_epoch is None:
        return SequentialLR(
            optimizer=optimizer,
            schedulers=[scheduler(optimizer) for scheduler in schedulers],
            milestones=milestones,
        )
    else:
        if step_attr is None:
            step_attr = []
        else:
            # Convert milestones to steps
            milestones = [m * steps_per_epoch for m in milestones]

        # Convert attributes from epochs to steps
        step_schedulers = []
        for scheduler in schedulers:
            # Convert attributes
            scheduler_kwargs = scheduler.keywords.copy()
            for attr in step_attr:
                if attr in scheduler_kwargs:
                    scheduler_kwargs[attr] *= steps_per_epoch  # Convert epochs → steps

            # Initialize scheduler with updated parameters
            step_schedulers.append(scheduler.func(optimizer, **scheduler_kwargs))

        # Create SequentialLR scheduler
        return SequentialLR(optimizer=optimizer, schedulers=step_schedulers, milestones=milestones)
