# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the host/device overlap helpers and the repr-only reference cache."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from gfmbench_api.utils.inference_utils import BackgroundPrefetcher, tensor_to_numpy


def test_tensor_to_numpy_cpu_matches_plain_path():
    t = torch.randn(3, 5, 7)[:, 1:-1, :]  # non-contiguous
    out = tensor_to_numpy(t)
    assert np.array_equal(out, t.detach().cpu().numpy())
    assert out.dtype == np.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, torch.int64, torch.bool])
def test_tensor_to_numpy_cuda_is_bitwise_identical_and_independent(dtype):
    x = (torch.randn(4, 33, 17, device="cuda") * 100).to(dtype)
    view = x[:, 1:-1, :]
    first = tensor_to_numpy(view)
    reference = view.cpu().numpy()
    assert first.dtype == reference.dtype and first.shape == reference.shape
    assert first.tobytes() == reference.tobytes()
    # a later (larger, then smaller) call must not overwrite earlier results
    snapshot = first.copy()
    tensor_to_numpy(torch.zeros(1000, 1000, device="cuda").to(dtype))
    tensor_to_numpy(torch.ones(2, device="cuda").to(dtype))
    assert np.array_equal(first, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tensor_to_numpy_handles_empty_and_requires_grad():
    assert tensor_to_numpy(torch.zeros(0, 3, device="cuda")).shape == (0, 3)
    w = torch.randn(3, 3, device="cuda", requires_grad=True)
    assert np.array_equal(tensor_to_numpy(w * 2), (w * 2).detach().cpu().numpy())


def test_prefetcher_hit_miss_and_single_use():
    p = BackgroundPrefetcher()
    assert not p.has_pending
    p.new_generation()
    p.submit(("k", (1, 2)), lambda x: x * 2, 21)
    assert p.has_pending
    assert p.take(("other",)) is None
    assert p.take(("k", (1, 2))) == 42
    assert p.take(("k", (1, 2))) is None  # consumed
    assert not p.has_pending


def test_prefetcher_failed_job_is_a_miss():
    p = BackgroundPrefetcher()
    p.new_generation()

    def bad():
        raise ValueError("nope")

    p.submit("bad", bad)
    assert p.take("bad") is None


def test_prefetcher_runs_on_one_background_thread_in_order():
    p = BackgroundPrefetcher()
    p.new_generation()
    seen = []
    main = threading.get_ident()
    for i in range(5):
        p.submit(i, lambda i=i: (seen.append(i), threading.get_ident())[1])
    idents = {p.take(i) for i in range(5)}
    assert seen == list(range(5))
    assert len(idents) == 1 and main not in idents


def test_prefetcher_drops_entries_older_than_two_generations():
    p = BackgroundPrefetcher()
    p.new_generation()
    p.submit("old", lambda: 1)
    p.new_generation()
    p.submit("current", lambda: 2)
    assert p.take("old") == 1  # still within two generations
    p.submit("older", lambda: 3)
    p.new_generation()
    p.new_generation()
    assert p.take("older") is None and p.take("current") is None


def test_prefetcher_overlaps_with_caller_work():
    p = BackgroundPrefetcher()
    p.new_generation()
    p.submit("slow", lambda: time.sleep(0.2) or "done")
    start = time.perf_counter()
    time.sleep(0.2)  # caller's own work
    assert p.take("slow") == "done"
    assert time.perf_counter() - start < 0.35  # ~0.2s, not 0.4s


def test_projection_wrapper_caches_only_the_representative():
    from usage_examples.trainers.model_wrapper import GFMWithProjection

    class Base:
        device = "cpu"
        calls = []

        def infer_sequence_to_sequence(self, sequences, conditional_input=None):
            self.calls.append(list(sequences))
            n = len(sequences)
            emb = np.ones((n, 50, 16), dtype=np.float32)
            rep = np.stack([np.full(16, float(len(s)), dtype=np.float32) for s in sequences])
            return None, emb, rep

    proj = torch.nn.Linear(32, 2)

    def run(cache_size):
        base = Base()
        base.calls = []
        wrapper = GFMWithProjection(base, proj, num_labels=1, num_classes=2, cache_size=cache_size)
        out = [
            wrapper.infer_variant_ref_sequences_to_labels_probs(["AA", "AAA"], ["CC", "CCCC"]),
            wrapper.infer_variant_ref_sequences_to_labels_probs(["A", "AAAA"], ["CC", "CCC"]),  # one ref hit
        ]
        return wrapper, base, out

    cached, base_cached, out_cached = run(None)
    _, base_plain, out_plain = run(0)
    for a, b in zip(out_cached, out_plain):
        assert np.array_equal(a, b)
    assert ["CC", "CCC"] not in base_cached.calls and ["CCC"] in base_cached.calls  # miss-only batch
    entry_bytes = cached._ref_cache.current_size_bytes
    assert 0 < entry_bytes <= 3 * 16 * 4  # three unique refs x one 16-float representative each


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_staging_buffer_growth_is_capped(monkeypatch):
    from gfmbench_api.utils import inference_utils as iu

    cap = 1 << 20
    monkeypatch.setattr(iu, "_MAX_STAGING_BYTES", cap)
    iu._staging.buf = None
    tensor_to_numpy(torch.zeros(300_000, dtype=torch.uint8, device="cuda"))
    tensor_to_numpy(torch.zeros(900_000, dtype=torch.uint8, device="cuda"))
    assert iu._staging.buf.numel() <= cap
    iu._staging.buf = None
