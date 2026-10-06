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

import types

import pytest
import torch

from nemo_automodel.components.speculative.sigma_uno import (
    NoisyStream,
    attach_noisy_stream,
    noisy_stream_inputs,
    sample_block_times,
)


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab=32, hidden=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)
        self.config = types.SimpleNamespace(hidden_size=hidden)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))


def test_schedule_is_variance_preserving_and_monotone():
    stream = NoisyStream(32, 8, gamma_min=-3.0, gamma_max=6.0)
    t = torch.linspace(0, 1, 11)
    alpha, sigma = stream.alpha_sigma(t)
    torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(t))
    assert (alpha[1:] < alpha[:-1]).all()  # SNR(t) decreases over [0, 1]


def test_unit_embedding_rows_lie_on_the_sphere():
    stream = NoisyStream(32, 8)
    torch.testing.assert_close(stream.unit_embedding().norm(dim=-1), torch.ones(32))


def test_time_inection_starts_at_zero():
    """Sigma App. E: the time features enter through a zero-initialized layer."""
    stream = NoisyStream(32, 8)
    latents = torch.randn(1, 3, 16)
    t = torch.rand(1, 3)
    alpha, sigma = stream.alpha_sigma(t)
    scale = torch.rsqrt(alpha.square() / 16 + sigma.square()).unsqueeze(-1)
    torch.testing.assert_close(stream(latents, t), stream.latent_in(latents * scale))


def test_sample_block_times_shares_t_within_a_block_and_zeroes_unsupervised():
    noise_mask = torch.zeros(4, 10, dtype=torch.bool)
    noise_mask[:, 3:] = True
    t = sample_block_times(noise_mask, block_size=4, p_rec=0.0, generator=torch.Generator().manual_seed(0))
    assert (t[:, :3] == 0).all()
    assert (t[:, 4:8] == t[:, 4:5]).all() and (t[:, 8:] == t[:, 8:9]).all()
    assert ((t[:, 3:] > 0) & (t[:, 3:] <= 1)).all()
    t_rec = sample_block_times(noise_mask, block_size=4, p_rec=1.0)
    assert (t_rec == 0).all()


def test_hook_replaces_only_masked_noisy_half_positions():
    torch.manual_seed(0)
    model = _TinyLM()
    stream = attach_noisy_stream(model)
    seq_len = 4
    clean = torch.randint(0, 32, (1, seq_len))
    concat = torch.cat([clean, clean], dim=1)
    mask = torch.tensor([[False, True, True, False]])
    t = torch.full((1, seq_len), 0.5)
    eps = torch.randn(1, seq_len, 16)

    base = model.embed.weight[concat]
    with noisy_stream_inputs(model, t, mask, eps=eps):
        out = model.embed(concat)
    expected_noisy = stream(stream.noise(clean, t, eps), t)
    torch.testing.assert_close(out[:, seq_len:], base[:, seq_len:])
    torch.testing.assert_close(out[:, :seq_len][~mask], base[:, :seq_len][~mask])
    torch.testing.assert_close(out[:, :seq_len][mask], expected_noisy[mask])
    torch.testing.assert_close(model.embed(concat), base)  # cleared on exit
    assert stream._t is None


def test_latents_override_the_internal_noising():
    model = _TinyLM()
    stream = attach_noisy_stream(model)
    clean = torch.randint(0, 32, (1, 3))
    mask = torch.ones(1, 3, dtype=torch.bool)
    t = torch.full((1, 3), 0.25)
    latents = torch.randn(1, 3, 16)
    with noisy_stream_inputs(model, t, mask, latents=latents):
        out = model.embed(torch.cat([clean, clean], dim=1))
    torch.testing.assert_close(out[:, :3], stream(latents, t))


def test_noisy_stream_inputs_needs_exactly_one_of_eps_or_latents():
    model = _TinyLM()
    attach_noisy_stream(model)
    t = torch.zeros(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="exactly one"):
        with noisy_stream_inputs(model, t, mask):
            pass
    with pytest.raises(ValueError, match="exactly one"):
        with noisy_stream_inputs(model, t, mask, eps=torch.zeros(1, 2, 16), latents=torch.zeros(1, 2, 16)):
            pass
