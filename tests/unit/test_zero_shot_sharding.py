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

"""Sharded / prefetching zero-shot evaluation must reproduce the plain run exactly.

Uses a stateless mock model (output depends only on each sequence), so any difference
between runs would come from the batching/merging code under test.
"""

from __future__ import annotations

import zlib
from typing import List, Optional

import numpy as np
import pickle

import pytest
from torch.utils.data import DataLoader, Subset

from gfmbench_api.metrics import SequenceEmbeddingsL2AUROC
from gfmbench_api.utils.sharding import (
    ShardSpec,
    num_batches,
    shard_batch_ids,
    shard_sample_indices,
)
from tests.e2e.fixture_tasks import FixtureZeroShotSNVTask

BASES = "ACGT"


def _make_dataset(n: int) -> list:
    rng = np.random.RandomState(7)
    rows = []
    for i in range(n):
        ref = "".join(rng.choice(list(BASES), size=40))
        pos = FixtureZeroShotSNVTask.SNV_POSITION
        alt = BASES[(BASES.index(ref[pos]) + 1) % 4]
        rows.append((ref[:pos] + alt + ref[pos + 1:], ref, int(rng.rand() < 0.5), np.array([])))
    return rows


class SyntheticSNVTask(FixtureZeroShotSNVTask):
    N = 103  # 26 batches of 4 with a short final batch

    def get_task_name(self) -> str:
        return "synthetic_snv"

    def _create_test_dataset(self) -> list:
        return _make_dataset(self.N)


class StatelessModel:
    """Deterministic per-sequence outputs; records prefetch calls."""

    hidden = 8

    def __init__(self, with_prefetch: bool = False) -> None:
        self.prefetched: List[tuple] = []
        self.calls = 0
        if with_prefetch:
            self.prefetch_batch = self._prefetch_batch

    def _prefetch_batch(self, variant_sequences, reference_sequences, variant_pos=None):
        self.prefetched.append((tuple(variant_sequences), variant_pos))

    @staticmethod
    def _rng(seq: str) -> np.random.RandomState:
        return np.random.RandomState(zlib.crc32(seq.encode()))

    def infer_sequence_to_sequence(self, sequences, conditional_input=None):
        self.calls += 1
        L = max(len(s) for s in sequences)
        probs = np.zeros((len(sequences), L), dtype=np.float32)
        emb = np.zeros((len(sequences), L, self.hidden), dtype=np.float32)
        rep = np.zeros((len(sequences), self.hidden), dtype=np.float32)
        for i, s in enumerate(sequences):
            r = self._rng(s)
            probs[i, : len(s)] = 0.05 + 0.9 * r.rand(len(s))
            emb[i, : len(s)] = r.randn(len(s), self.hidden)
            rep[i] = r.randn(self.hidden)
        return probs, emb, rep

    def sequence_pos_to_prob_pos(self, sequences, pos):
        return np.full(len(sequences), pos, dtype=np.int32)

    def infer_masked_sequence_to_token_probs(self, sequences, pos, var_letters, ref_letters, conditional_input=None):
        v = np.array([self._rng(s + a).rand() for s, a in zip(sequences, var_letters)], dtype=np.float32)
        r = np.array([self._rng(s + a).rand() for s, a in zip(sequences, ref_letters)], dtype=np.float32)
        return v, r


def _config(**extra) -> dict:
    return {"batch_size": 4, "max_sequence_length": 64, "max_num_samples": None, **extra}


def _run_unsharded(**extra):
    task = SyntheticSNVTask("unused", _config(**extra))
    return task.eval_test_set(StatelessModel())


def _run_sharded(world: int, tmp_path, order=None, run_id="run-1"):
    results = []
    for rank in order if order is not None else range(world):
        shard = {"rank": rank, "world": world, "dir": str(tmp_path), "run_id": run_id}
        task = SyntheticSNVTask("unused", _config(shard=shard))
        results.append(task.eval_test_set(StatelessModel()))
    return results


@pytest.mark.parametrize("n,bs,world", [(103, 4, 3), (8, 4, 5), (1, 4, 2), (100, 4, 4), (7, 7, 3)])
def test_shard_loaders_reproduce_exact_batches(n, bs, world):
    data = list(range(n))
    expected = [b.tolist() for b in DataLoader(data, batch_size=bs, shuffle=False)]
    seen = {}
    for rank in range(world):
        spec = ShardSpec(rank, world, "unused", "run-1")
        ids = shard_batch_ids(n, bs, spec)
        loader = DataLoader(Subset(data, shard_sample_indices(n, bs, spec)), batch_size=bs, shuffle=False)
        batches = [b.tolist() for b in loader]
        assert len(batches) == len(ids)
        for batch_id, batch in zip(ids, batches):
            seen[batch_id] = batch
    assert sorted(seen) == list(range(num_batches(n, bs)))
    assert [seen[i] for i in sorted(seen)] == expected


@pytest.mark.parametrize("world", [2, 3, 5, 40])
def test_sharded_scores_are_bitwise_identical(world, tmp_path):
    expected = _run_unsharded()
    assert all(v is not None for v in expected.values())
    results = _run_sharded(world, tmp_path)
    assert all(r is None for r in results[:-1])
    assert results[-1] == expected  # exact equality, no tolerance


def test_shard_completion_order_does_not_matter(tmp_path):
    expected = _run_unsharded()
    results = _run_sharded(3, tmp_path, order=[2, 0, 1])
    assert results[-1] == expected


def test_two_shards_finishing_together_both_merge_identically(tmp_path):
    expected = _run_unsharded()
    _run_sharded(2, tmp_path)
    again = _run_sharded(2, tmp_path, order=[0])  # rank 0 reruns after all files exist
    assert again[-1] == expected


def test_mismatched_shard_dataset_is_rejected(tmp_path):
    _run_sharded(2, tmp_path, order=[0])

    class OtherTask(SyntheticSNVTask):
        N = 99

    shard = {"rank": 1, "world": 2, "dir": str(tmp_path), "run_id": "run-1"}
    with pytest.raises((RuntimeError, ValueError)):
        OtherTask("unused", _config(shard=shard)).eval_test_set(StatelessModel())


def test_prefetch_hook_sees_every_batch_once_ahead_of_processing():
    model = StatelessModel(with_prefetch=True)
    task = SyntheticSNVTask("unused", _config())
    scores = task.eval_test_set(model)
    data = task.test_dataset
    expected = [tuple(d[0] for d in data[i:i + 4]) for i in range(0, len(data), 4)]
    assert [seqs for seqs, _ in model.prefetched] == expected
    assert {pos for _, pos in model.prefetched} == {FixtureZeroShotSNVTask.SNV_POSITION}
    assert scores == _run_unsharded()


def test_prefetch_can_be_disabled_and_failures_are_ignored():
    model = StatelessModel(with_prefetch=True)
    SyntheticSNVTask("unused", _config(prefetch=False)).eval_test_set(model)
    assert model.prefetched == []

    def boom(*args, **kwargs):
        raise RuntimeError("prefetch exploded")

    broken = StatelessModel()
    broken.prefetch_batch = boom
    assert SyntheticSNVTask("unused", _config()).eval_test_set(broken) == _run_unsharded()

    # Prefetching is best-effort even when model inference errors are surfaced.
    strict = StatelessModel()
    strict.prefetch_batch = boom
    cfg = _config(disable_safe_model_call=True)
    assert SyntheticSNVTask("unused", cfg).eval_test_set(strict) == _run_unsharded()


def test_metric_state_export_and_extend_roundtrip():
    rng = np.random.RandomState(0)
    batches = [(rng.randn(5, 4), rng.randn(5, 4), rng.randint(0, 2, 5)) for _ in range(4)]
    whole, parts = SequenceEmbeddingsL2AUROC(), SequenceEmbeddingsL2AUROC()
    deltas = []
    for a, b, y in batches:
        mark = whole.state_lengths()
        whole.calc(a, b, y)
        deltas.append(whole.state_since(mark))
    for delta in deltas:
        parts.extend_state(delta)
    assert parts.get_final_results() == whole.get_final_results()
    with pytest.raises(KeyError):
        parts.extend_state({"_bogus": [1]})


def test_shard_spec_validation():
    assert ShardSpec.from_config(None) is None
    assert ShardSpec.from_config({"rank": 0, "world": 1, "dir": "x"}) is None
    with pytest.raises(ValueError):
        ShardSpec(rank=2, world=2, directory="x")


def test_shard_files_from_a_previous_run_are_never_merged(tmp_path):
    expected = _run_unsharded()
    _run_sharded(2, tmp_path, run_id="old-run")
    # New run: rank 0 finishes first and must NOT pick up the old run's rank-1 file.
    assert _run_sharded(2, tmp_path, order=[0], run_id="new-run") == [None]
    # Once rank 1 of the new run finishes, the merged result is the correct one.
    assert _run_sharded(2, tmp_path, order=[1], run_id="new-run")[-1] == expected


def test_sharding_requires_a_run_id():
    with pytest.raises(ValueError):
        ShardSpec(rank=0, world=2, directory="x")


def test_metric_with_non_list_state_cannot_be_sharded():
    metric = SequenceEmbeddingsL2AUROC()
    metric.counter = 3
    with pytest.raises(TypeError):
        metric.state_lengths()


def test_corrupt_shard_file_is_retried_then_raises(tmp_path, monkeypatch):
    from gfmbench_api.utils import sharding

    monkeypatch.setattr(sharding.time, "sleep", lambda _: None)
    spec = sharding.ShardSpec.from_config({"rank": 0, "world": 2, "dir": str(tmp_path), "run_id": "r"})
    open(sharding.shard_path(spec, "t", 0), "wb").write(b"\x80\x04trunc")
    with pytest.raises((EOFError, pickle.UnpicklingError)):
        sharding.load_all_shards(spec, "t")
