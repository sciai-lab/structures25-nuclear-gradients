import torch
from numpy import exp, log10, mean


class OffsetOneCycleScheduler(torch.optim.lr_scheduler._LRScheduler):
    """A variant of the OneCycle learning rate scheduler with an initial warm-up offset phase.

    This scheduler behaves like PyTorch's OneCycleLR, but includes an initial offset period
    (`offset_steps`) during which the learning rate is held constant at the minimum value (`min_lr`).
    After the offset phase, it transitions into the regular OneCycleLR schedule for `total_steps`.
    Once the OneCycle schedule completes, it reverts to the fixed `min_lr`.

    Parameters:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        max_lr (float): The peak learning rate to use during the OneCycle schedule.
        total_steps (int): The number of steps over which the OneCycle schedule will run.
        offset_steps (int): Number of initial steps to maintain `min_lr` before starting OneCycleLR.
        min_lr (float): Minimum learning rate to use during offset and after the cycle completes.
        div_factor (float, optional): Initial learning rate = max_lr / div_factor.
            If None, calculated automatically to match `min_lr`.
        final_div_factor (float): Determines the final learning rate via
            `max_lr / (div_factor * final_div_factor)`.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 0.009,
        min_lr: float = 0.003,
        total_steps: int = 2000,
        offset_steps: int = 80,
        div_factor: float = None,
        final_div_factor: float = 1,
        last_epoch: int = -1,
    ):
        self.offset_steps = offset_steps
        self.total_steps = total_steps
        self.lower_bound = min_lr

        if div_factor is None:
            div_factor = max_lr / min_lr  # Calculate div_factor to ensure min_lr is respected

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        super().__init__(optimizer, last_epoch)

    def step(self):
        """Advance the learning rate schedule by one step.

        - During the initial `offset_steps`, the learning rate is held constant at `min_lr`.
        - From `offset_steps` to `offset_steps + total_steps`, the OneCycleLR schedule is applied.
        - After `offset_steps + total_steps`, the learning rate returns to `min_lr` and remains fixed.

        This method should be called once per training step (e.g., per batch).
        """

        self.last_epoch += 1

        if self.last_epoch < self.offset_steps:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lower_bound

        elif self.offset_steps <= self.last_epoch < self.offset_steps + self.total_steps:
            self.scheduler.step()

        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lower_bound

    def __str__(self):
        """Returns a string representation of the scheduler's current state."""
        return (
            f"{self.__class__.__name__}("
            f"offset_steps={self.offset_steps}, "
            f"total_steps={self.total_steps}, "
            f"lower_bound={self.lower_bound}, "
            f"last_epoch={self.last_epoch}, "
            f"current_lr={[pg['lr'] for pg in self.optimizer.param_groups]}"
            f")"
        )


class AdaptiveSigmoidScheduler:
    """A learning rate scheduler that adjusts the learning rate based on the gradient norm using a
    sigmoid function.

    This scheduler is designed to ramp up to a high learning rate and reduces it to the
    mean of the upper and lower bound. The learning rate is computed using a sigmoid function over
    the log10 of the gradient norm. It changes the learning rate dynamically until 300 steps were made.

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
        max_lr (float): Initial upper bound of the sigmoid function (maximum learning rate).
        min_lr (float): Lower bound of the sigmoid function (minimum learning rate).
        offset (float): Horizontal offset for the sigmoid input. Shifts the curve left or right.
        slope (float): Controls the steepness of the sigmoid curve.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 0.09,
        min_lr: float = 0.003,
        offset: int = 1,
        slope: float = 3.0,
    ):
        self.optimizer = optimizer

        if max_lr <= min_lr:
            raise ValueError("max_lr must be greater than min_lr")

        self.diff = max_lr - min_lr  # ensure max_lr is plateau of the sigmoid
        self.lower_bound = min_lr
        self.offset = offset
        self.slope = slope
        self.last_epoch = -1
        self.mean_lr = (max_lr + min_lr) / 2

    def sigmoid(self, x):
        """Sigmoid function to adjust learning rates dynamically."""
        return self.diff / (1 + exp(self.slope * (x + self.offset))) + self.lower_bound

    def step(self, grad_norm: float):
        """Update the learning rate based on the current gradient norm.

        - For the first 300 steps, compute the learning rate dynamically using a sigmoid
          function of the log-scaled gradient norm.
        - After 300 steps, fix the learning rate to a constant value (`mean_lr`).

        Args:
            grad_norm (float): The norm of the gradients at the current step.
        """

        self.last_epoch += 1
        if self.last_epoch < 300:
            log_grad = log10(grad_norm + 1e-12)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.sigmoid(log_grad)
        else:
            for (
                param_group
            ) in self.optimizer.param_groups:  # After 300 epochs, keep the learning rate constant
                param_group["lr"] = self.mean_lr

    def __str__(self):
        """Returns a string representation of the scheduler's current state."""
        return (
            f"{self.__class__.__name__}("
            f"offset={self.offset}, "
            f"slope={self.slope}, "
            f"lower_bound={self.lower_bound}, "
            f"last_epoch={self.last_epoch}, "
            f"current_lr={[pg['lr'] for pg in self.optimizer.param_groups]}"
            f")"
        )


class AdaptiveSigmoidSchedulerWithDecay:
    """A learning rate scheduler that adjusts the learning rate based on the gradient norm using a
    sigmoid function, with optional decay upon detecting oscillatory behavior.

    This scheduler is designed to ramp up to a high learning rate and adaptively reduce it
    as training progresses. The learning rate is computed using a sigmoid function over
    the log10 of the gradient norm. If oscillations in the gradient norm are detected over
    time, the scheduler begins decaying the upper bound of the sigmoid curve to stabilize
    learning.

    These are the recommended parameters for the state-of-the-art STRUCTURES models:

        +------------------------+--------+--------+----------------+--------+--------+----------+
        | model                  | min_lr | max_lr | min_lr_decay   | offset | slope  | speed up |
        +========================+========+========+================+========+========+==========+
        | STRUCTURES25 QM9       | 0.003  | 0.009  | 0.003          | 3.5    | 1      | 1.5x     |
        +------------------------+--------+--------+----------------+--------+--------+----------+
        | STRUCTURES25 QMugs     | 0.0015 | 0.0045 | 0.003          | 1      | 2      | 2.7x     |
        +------------------------+--------+--------+----------------+--------+--------+----------+


    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
        max_lr (float): Initial upper bound of the sigmoid function (maximum learning rate).
        min_lr (float): Lower bound of the sigmoid function (minimum learning rate).
        min_lr_decay (float, optional): Minimum learning rate to enforce after decay. Defaults to `min_lr`.
        offset (float): Horizontal offset for the sigmoid input. Shifts the curve left or right.
        slope (float): Controls the steepness of the sigmoid curve.
        decay_rate (float): Multiplicative factor (in (0, 1)) to reduce `max_lr` upon oscillation detection.
        window_size (int): Number of steps in each window used for oscillation detection.
        detect_offset (int): Step size between windows when computing rolling means for oscillation detection.
        threshold (float): Minimum increase in rolling mean required to trigger oscillation detection.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 0.09,
        min_lr: float = 0.003,
        min_lr_decay: float = 0.006,
        offset: float = 1,
        slope: float = 3.5,
        decay_rate: float = 0.995,
        window_size: int = 10,
        detect_offset: int = 5,
        threshold: float = 1e-5,
    ):
        self.optimizer = optimizer

        # Sigmoid parameters
        if max_lr <= min_lr:
            raise ValueError("max_lr must be greater than min_lr")

        self.upper_bound = max_lr
        self.lower_bound = min_lr

        if min_lr_decay is None:
            min_lr_decay = min_lr
        self.decay_lower_bound = min_lr_decay

        self.offset = offset
        self.slope = slope

        # Oscillation detection
        self.window_size = window_size
        self.detect_offset = detect_offset
        self.threshold = threshold
        self.history = []
        self.oscillation_detected = False
        self.fixed_log_input = None

        self.decay_rate = decay_rate
        if self.decay_rate <= 0 or self.decay_rate >= 1:
            raise ValueError("decay_rate must be in the range (0, 1)")

        self.last_epoch = -1
        self.decay_steps = 0

    def sigmoid(self, x):
        """Sigmoid function to adjust learning rates dynamically."""
        diff = self.upper_bound - self.lower_bound
        return diff / (1 + exp(self.slope * (x + self.offset))) + self.lower_bound

    def detect_oscillation(self):
        """Detects oscillatory behavior in the log-gradient history.

        The method slides overlapping windows over the gradient history and compares
        the mean values of successive windows. If the mean increases by more than
        a threshold, it is considered an oscillation.

        Returns:
            bool: True if oscillation is detected, False otherwise.
        """

        N = len(self.history)
        if N < self.window_size + self.detect_offset:
            return False

        window_means = []
        i = 0
        while i + self.window_size <= N:
            window = self.history[i : i + self.window_size]
            window_means.append(mean(window))
            i += self.detect_offset

        for j in range(1, len(window_means)):
            if window_means[j] > window_means[j - 1] + self.threshold:
                return True
        return False

    def step(self, grad_norm: float):
        """Update the learning rate based on the gradient norm using an adaptive sigmoid schedule.

        - Logs the current gradient norm and appends it to history.
        - If oscillation is detected in the log-gradient signal, begins exponential decay
          of the upper bound of the sigmoid schedule.
        - Computes the new learning rate using the sigmoid function and applies it to the optimizer.

        Args:
            grad_norm (float): The norm of the gradients at the current step.
        """

        self.last_epoch += 1

        log_grad = log10(grad_norm + 1e-12)
        self.history.append(log_grad)

        # Detect oscillation once
        if not self.oscillation_detected and self.detect_oscillation():
            self.oscillation_detected = True
            self.fixed_log_input = log_grad  # Store last log_grad
            self.decay_steps = 0

        # Decay upper bound if oscillation was detected
        if self.oscillation_detected:
            self.upper_bound *= self.decay_rate
            self.upper_bound = max(self.upper_bound, self.lower_bound)
            self.decay_steps += 1
            x = self.fixed_log_input
        else:
            x = log_grad

        new_lr = self.sigmoid(x)

        if self.oscillation_detected:
            # If oscillation was detected, use lower bound for decay
            new_lr = max(new_lr, self.decay_lower_bound)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def __str__(self):
        """Returns a string representation of the scheduler's current state."""
        return (
            f"{self.__class__.__name__}("
            f"offset={self.offset}, "
            f"slope={self.slope}, "
            f"upper_bound={self.upper_bound:.6f}, "
            f"lower_bound={self.lower_bound}, "
            f"decay_lower_bound={self.decay_lower_bound}, "
            f"decay_rate={self.decay_rate}, "
            f"oscillation_detected={self.oscillation_detected}, "
            f"last_epoch={self.last_epoch}, "
            f"current_lr={[pg['lr'] for pg in self.optimizer.param_groups]}"
            f")"
        )
