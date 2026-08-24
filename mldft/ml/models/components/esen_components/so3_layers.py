"""Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the LICENSE file in the root directory
of this source tree.
"""

from __future__ import annotations

import math

import torch


class SO3_Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)
        scalar_mask = torch.zeros((1, (lmax + 1) ** 2, 1))
        scalar_mask[:, 0, :] = 1
        # Keep this runtime-only buffer out of checkpoints for backwards compatibility.
        self.register_buffer("scalar_mask", scalar_mask, persistent=False)

    # Rebuild runtime-only buffers when unpickling legacy checkpoints.
    def _ensure_scalar_mask(self) -> None:
        if self._buffers.get("scalar_mask") is not None:
            return
        scalar_mask = self.bias.new_zeros((1, (self.lmax + 1) ** 2, 1))
        scalar_mask[:, 0, :] = 1
        self._buffers["scalar_mask"] = scalar_mask
        self._non_persistent_buffers_set.add("scalar_mask")

    def __setstate__(self, state):
        super().__setstate__(state)
        self._ensure_scalar_mask()

    def forward(self, input_embedding):
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        return out + self.scalar_mask * self.bias.view(1, 1, self.out_features)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"
