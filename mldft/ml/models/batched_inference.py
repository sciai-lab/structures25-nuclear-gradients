import threading
import time
from queue import Empty, Queue

import torch
from loguru import logger

from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.model_output import ModelOutput
from mldft.ml.models.mldft_module import MLDFTLitModule


class BatchedInferenceWrapper(MLDFTLitModule):
    def __init__(self, *args, inference_batch_size: int = 6, timeout: float = 0.05, **kwargs):
        """Wraps a PyTorch model for batched inference with a background worker thread.

        Args:
            model: The model to be wrapped.
            inference_batch_size: Maximum number of samples to batch.
            timeout: Maximum waiting time (in seconds) before processing a batch.
        """
        super().__init__(*args, **kwargs)

        self.inference_batch_size = inference_batch_size
        self.timeout = timeout
        self.queue = Queue()

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def forward(self, sample: OFData, *model_args) -> ModelOutput:
        """Enqueues a single sample for inference and waits for its result.

        Args:
            sample: Input sample to be inferred.
            model_args: Additional arguments to be passed to the forward method of the model.

        Returns:
            ModelOutput: Typed model outputs for the input sample.
        """
        if not self._worker_thread.is_alive():
            raise RuntimeError("Worker thread has died")

        result_queue = Queue()
        # cloning seems to be slower than batching the matrices in lists
        # sample = sample.clone()
        # sample.delete_item("overlap_matrix")
        # sample.delete_item("coulomb_matrix")
        # sample.delete_item("transformation_matrix")
        # sample.delete_item("inv_transformation_matrix")

        # we use sample_forward in the functional, but we can't have samples in the batch with
        # and without the predicted quantities in the collation, so we remove them here
        if hasattr(sample, "pred_energy"):
            sample.delete_item("pred_energy")
        if hasattr(sample, "pred_gradient"):
            sample.delete_item("pred_gradient")
        if hasattr(sample, "pred_diff"):
            sample.delete_item("pred_diff")

        self.queue.put((sample, result_queue, model_args))

        try:
            model_output = result_queue.get(timeout=100)
        except Empty:
            raise RuntimeError("Inference timed out")

        return model_output

    def _worker(self):
        """Background thread that collects incoming requests and performs batched inference."""
        buffer = []
        last_time = time.time()
        current_args = None

        while True:
            try:
                sample, result_queue, model_args = self.queue.get(timeout=self.timeout)
                if current_args is None:
                    current_args = model_args
                else:
                    message = f"Model args must be the same in batched inference, got {current_args} and {model_args}"
                    assert current_args == model_args, message
                buffer.append((sample, result_queue))
            except Empty:
                pass

            batch_full = len(buffer) >= self.inference_batch_size
            timeout_reached = (time.time() - last_time) >= self.timeout
            if batch_full or (timeout_reached and buffer):
                batch = buffer[: self.inference_batch_size]
                buffer = buffer[self.inference_batch_size :]

                self.process_batch(batch, *current_args)
                last_time = time.time()

    def process_batch(self, batch: list[tuple[OFData, Queue]], *model_args):
        """Process a batch of samples.

        Args:
            batch: List of tuples containing samples and their corresponding result queues.
            model_args: Additional arguments to be passed to the forward method of the model.
        """
        sample_list = [item[0] for item in batch]
        result_queue = [item[1] for item in batch]

        of_batch = OFBatch.from_data_list(
            sample_list,
            follow_batch=["coeffs", "atomic_numbers"],
            list_keys=[
                "overlap_matrix",
                "coulomb_matrix",
                "transformation_matrix",
                "inv_transformation_matrix",
            ],
        )

        model_output = super().forward(of_batch, *model_args)
        slices = of_batch._slice_dict["coeffs"]
        slices = slices[1:] - slices[:-1]
        slices = list(slices)

        pred_energy = model_output.pred_energy
        pred_gradient = torch.split(model_output.pred_gradient, slices)
        pred_diff = torch.split(model_output.pred_diff, slices)

        for i, res in enumerate(result_queue):
            res.put(
                ModelOutput(
                    pred_energy=pred_energy[i],
                    pred_gradient=pred_gradient[i],
                    pred_diff=pred_diff[i],
                )
            )

        logger.trace(f"Batch of {len(sample_list)} samples processed")
