# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sigma-Uno continuous noisy stream.

Uno (Sahoo et al., 2026) trains a gated LoRA so the noisy half of an ``[x_t | x_0]`` forward drafts a block
that the frozen AR model verifies losslessly. Sigma-Uno keeps that setup and replaces Uno's token-noised
``x_t`` with the continuous diffusion stream of Sigma (Wang et al., 2026; arXiv:2610.02665):

- a learnable ``[vocab, d_e]`` diffusion embedding table whose rows lie on the unit sphere, ``d_e = 16``
  (Sigma Sec. 2 "Plaid", Sec. 3.2.1);
- variance-preserving Gaussian noising ``z_t = alpha_t e + sigma_t eps`` (Sigma Eq. 1) with
  ``alpha_t^2 = sigmoid(-gamma(t))`` and ``sigma_t^2 = sigmoid(gamma(t))`` (Sigma Sec. 2 "Learned noise
  schedule"); here ``gamma`` is linear between fixed endpoints, as Sigma freezes the schedule during SFT
  (App. H.1);
- the variance-rescaled ``z_t`` through a linear up-projection to the hidden width, plus Fourier features of
  ``gamma`` (frequencies ``exp(linspace(-5, 5, 32))``, ``[sin, cos]``) through a bias-free, zero-initialized
  linear layer (Sigma App. E "Time conditioning").

The stream lives on the model as a child of the input embedding, so FSDP2, the optimizer and the PEFT
checkpoint pick it up like any trainable parameter. A forward hook on the input embedding swaps the noisy
half's AR embeddings for the stream's output while :func:`noisy_stream_inputs` is active. Token ids stay
clean; noise lives only in embedding space (Sigma App. H.1).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

NUM_TIME_FREQUENCIES = 32


class NoisyStream(nn.Module):
    """Sigma diffusion-stream input: unit-norm embeddings, Gaussian noising, up-projection, time features.

    Args:
        vocab_size: Number of rows of the diffusion embedding table.
        hidden_size: Width of the transformer the stream feeds.
        diffusion_dim: Diffusion embedding width ``d_e``.
        gamma_min: Minus log-SNR ``gamma(0)`` (least noise).
        gamma_max: Minus log-SNR ``gamma(1)`` (most noise).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        diffusion_dim: int = 16,
        gamma_min: float = -3.0,
        gamma_max: float = 6.0,
    ):
        super().__init__()
        if not gamma_min < gamma_max:
            raise ValueError(f"gamma_min={gamma_min} must be smaller than gamma_max={gamma_max}.")
        self.diffusion_dim = int(diffusion_dim)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.embedding = nn.Parameter(torch.randn(vocab_size, self.diffusion_dim))
        self.input_proj = nn.Linear(self.diffusion_dim, hidden_size, bias=False)
        self.time_proj = nn.Linear(2 * NUM_TIME_FREQUENCIES, hidden_size, bias=False)
        nn.init.zeros_(self.time_proj.weight)
        self.register_buffer(
            "time_frequencies", torch.exp(torch.linspace(-5.0, 5.0, NUM_TIME_FREQUENCIES)), persistent=False
        )
        # Filled by :func:`noisy_stream_inputs` for the duration of one forward and backward.
        self._t: torch.Tensor | None = None
        self._noisy_mask: torch.Tensor | None = None
        self._eps: torch.Tensor | None = None
        self._latents: torch.Tensor | None = None

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Minus log-SNR ``gamma(t)``, computed in fp64 as in Sigma App. E "Numerical precision"."""
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t.double()

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(alpha_t, sigma_t)`` in fp32 with ``alpha_t^2 + sigma_t^2 = 1``."""
        gamma = self.gamma(t)
        return torch.sigmoid(-gamma).sqrt().float(), torch.sigmoid(gamma).sqrt().float()

    def unit_embedding(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Unit-norm diffusion embeddings, the whole ``[vocab, d_e]`` table or the rows of ``token_ids``."""
        table = F.normalize(self.embedding.float(), dim=-1)
        return table if token_ids is None else table[token_ids]

    def noise(self, token_ids: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Noise the clean diffusion embeddings of ``token_ids``: ``z_t = alpha_t e + sigma_t eps`` (Sigma Eq. 1).

        Args:
            token_ids: Clean token ids, Tensor of shape [batch, sequence].
            t: Diffusion time in ``[0, 1]``, Tensor of shape [batch, sequence].
            eps: Standard Gaussian noise, Tensor of shape [batch, sequence, d_e].

        Returns:
            ``z_t``, Tensor of shape [batch, sequence, d_e] in fp32.
        """
        alpha, sigma = self.alpha_sigma(t)
        return alpha.unsqueeze(-1) * self.unit_embedding(token_ids) + sigma.unsqueeze(-1) * eps.float()

    def forward(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Project ``z_t`` and its time to the hidden width.

        Args:
            latents: ``z_t``, Tensor of shape [batch, sequence, d_e].
            t: Diffusion time, Tensor of shape [batch, sequence].

        Returns:
            Tensor of shape [batch, sequence, hidden] in fp32.
        """
        alpha, sigma = self.alpha_sigma(t)
        # Per-coordinate variance of z_t is alpha^2 / d_e + sigma^2 for a unit-norm e; rescale it to one.
        scale = torch.rsqrt(alpha.square() / self.diffusion_dim + sigma.square()).unsqueeze(-1)
        angles = self.gamma(t).float().unsqueeze(-1) * self.time_frequencies
        time_features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.input_proj(latents.float() * scale) + self.time_proj(time_features)


def _noisy_stream_hook(embedding: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
    stream: NoisyStream = embedding.sigma_noisy_stream
    if stream._t is None:
        return output
    token_ids = args[0]
    seq_len = stream._t.size(1)
    latents = stream._latents
    if latents is None:
        latents = stream.noise(token_ids[:, :seq_len], stream._t, stream._eps)
    noisy = stream(latents, stream._t).to(output.dtype)
    noisy_half = torch.where(stream._noisy_mask.unsqueeze(-1), noisy, output[:, :seq_len])
    return torch.cat([noisy_half, output[:, seq_len:]], dim=1)


def attach_noisy_stream(
    model: nn.Module, diffusion_dim: int = 16, gamma_min: float = -3.0, gamma_max: float = 6.0
) -> NoisyStream:
    """Register a :class:`NoisyStream` as ``sigma_noisy_stream`` on the model's input embedding.

    Args:
        model: HF causal LM exposing ``get_input_embeddings()`` and ``config.hidden_size``.
        diffusion_dim: Diffusion embedding width ``d_e``.
        gamma_min: Minus log-SNR at ``t = 0``.
        gamma_max: Minus log-SNR at ``t = 1``.

    Returns:
        The attached stream.
    """
    embedding = model.get_input_embeddings()
    stream = NoisyStream(
        embedding.num_embeddings, model.config.hidden_size, diffusion_dim, gamma_min=gamma_min, gamma_max=gamma_max
    ).to(device=embedding.weight.device)
    embedding.sigma_noisy_stream = stream
    embedding.register_forward_hook(_noisy_stream_hook)
    return stream


def from_pretrained_with_noisy_stream(
    pretrained_model_name_or_path: str,
    *,
    diffusion_dim: int = 16,
    gamma_min: float = -3.0,
    gamma_max: float = 6.0,
    **kwargs: Any,
) -> nn.Module:
    """Load an HF causal LM and attach the Sigma-Uno noisy stream; a ``model._target_`` entry point.

    Args:
        pretrained_model_name_or_path: HF hub id or local path.
        diffusion_dim: Diffusion embedding width ``d_e``.
        gamma_min: Minus log-SNR at ``t = 0``.
        gamma_max: Minus log-SNR at ``t = 1``.
        **kwargs: Forwarded to ``AutoModelForCausalLM.from_pretrained``.

    Returns:
        The model with ``get_input_embeddings().sigma_noisy_stream`` attached.
    """
    from transformers import AutoModelForCausalLM

    # The dLLM recipe sets this NeMoAutoModel-only flag on every nemo_automodel target.
    kwargs.pop("_restore_loaded_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
    attach_noisy_stream(model, diffusion_dim, gamma_min=gamma_min, gamma_max=gamma_max)
    return model


def _find_noisy_stream(model: nn.Module) -> NoisyStream:
    streams = [module for module in model.modules() if isinstance(module, NoisyStream)]
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one NoisyStream in the model, found {len(streams)}.")
    return streams[0]


@contextmanager
def noisy_stream_inputs(
    model: nn.Module,
    t: torch.Tensor,
    noisy_mask: torch.Tensor,
    *,
    eps: torch.Tensor | None = None,
    latents: torch.Tensor | None = None,
) -> Iterator[None]:
    """Feed the noisy stream instead of the AR embeddings at ``noisy_mask`` positions of the first half.

    The model input is the ``[x_t | x_0]`` concatenation whose first ``sequence`` ids are the clean tokens.
    Training passes ``eps`` and the stream noises the clean diffusion embeddings itself, so gradients reach the
    embedding table; sampling passes the current ``latents`` instead. Run the backward pass inside the context
    too.

    Args:
        model: Model with an attached :class:`NoisyStream`.
        t: Diffusion time, Tensor of shape [batch, sequence].
        noisy_mask: Bool Tensor of shape [batch, sequence]; other positions keep their AR embedding.
        eps: Standard Gaussian noise, Tensor of shape [batch, sequence, d_e].
        latents: ``z_t``, Tensor of shape [batch, sequence, d_e].

    Yields:
        None. The inputs are cleared on exit, including on error.
    """
    if (eps is None) == (latents is None):
        raise ValueError("Pass exactly one of eps or latents.")
    stream = _find_noisy_stream(model)
    stream._t, stream._noisy_mask, stream._eps, stream._latents = t, noisy_mask.bool(), eps, latents
    try:
        yield
    finally:
        stream._t = stream._noisy_mask = stream._eps = stream._latents = None


def sample_block_times(
    noise_mask: torch.Tensor, block_size: int, p_rec: float = 0.1, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw one diffusion time per block (Sigma Alg. 1, lines 3-5): ``t = 0`` w.p. ``p_rec``, else ``U[0, 1]``.

    Args:
        noise_mask: Bool Tensor of shape [batch, sequence] marking supervised positions; others get ``t = 0``.
        block_size: Block length; blocks follow the absolute position grid of the block-diffusion mask.
        p_rec: Probability of a reconstruction block at ``t = 0``.
        generator: Optional seeded generator.

    Returns:
        Tensor of shape [batch, sequence] holding each position's block time.
    """
    batch, seq_len = noise_mask.shape
    num_blocks = math.ceil(seq_len / block_size)
    device = noise_mask.device
    t = torch.rand(batch, num_blocks, device=device, generator=generator)
    rec = torch.rand(batch, num_blocks, device=device, generator=generator) < p_rec
    t = torch.where(rec, torch.zeros_like(t), t)
    t = t.repeat_interleave(block_size, dim=1)[:, :seq_len]
    return torch.where(noise_mask.bool(), t, torch.zeros_like(t))
