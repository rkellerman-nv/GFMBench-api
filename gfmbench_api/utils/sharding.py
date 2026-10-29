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

"""Batch-level sharding of a zero-shot evaluation across processes (one per GPU).

A dataset is cut into the same fixed-size batches a single process would use; shard
``rank`` of ``world`` evaluates batches ``rank, rank + world, ...`` unchanged. Each shard
writes the per-batch metric state it produced, and whichever shard finishes last merges
all of them back in the original batch order. Metrics therefore see exactly the same
per-batch values in exactly the same order as in an unsharded run.

This relies on the model's output for a batch not depending on which batches ran before
it (true for stateless inference). A task with ``use_reference_cache() == True`` feeds the
model only the cache *misses*, so the batches it sees depend on history; sharding such a
task is bit-exact only if the model is bitwise independent of batch size and composition.
Shard tasks with the cache disabled, or verify the model is batch-invariant.
"""
import hashlib
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ShardSpec:
    rank: int
    world: int
    directory: str
    run_id: str = ""  # shared by all shards of one launch; files of other runs are ignored

    def __post_init__(self) -> None:
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError(f"Invalid shard rank={self.rank} world={self.world}")
        if self.world > 1 and not self.run_id:
            raise ValueError("Sharding needs a run_id shared by all shards of this launch.")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> Optional["ShardSpec"]:
        """Build from ``task_config["shard"]``; ``None`` when absent or single-shard."""
        if not config or int(config["world"]) <= 1:
            return None
        return cls(
            rank=int(config["rank"]),
            world=int(config["world"]),
            directory=config["dir"],
            run_id=str(config.get("run_id", "")),
        )


def num_batches(num_samples: int, batch_size: int) -> int:
    return -(-num_samples // batch_size)


def shard_batch_ids(num_samples: int, batch_size: int, spec: ShardSpec) -> List[int]:
    """Ids of the batches (in ascending order) owned by this shard."""
    return list(range(spec.rank, num_batches(num_samples, batch_size), spec.world))


def shard_sample_indices(num_samples: int, batch_size: int, spec: ShardSpec) -> List[int]:
    """Dataset indices of this shard's batches, concatenated in batch order.

    Only the globally last batch can be short, and it is the last batch of whichever
    shard owns it, so a sequential loader over these indices reproduces the exact batches.
    """
    indices: List[int] = []
    for batch_id in shard_batch_ids(num_samples, batch_size, spec):
        start = batch_id * batch_size
        indices.extend(range(start, min(start + batch_size, num_samples)))
    return indices


def dataset_fingerprint(dataset: Any) -> str:
    """Cheap identity check that every shard built the same dataset in the same order."""
    digest = hashlib.sha256()
    n = len(dataset)
    digest.update(str(n).encode())
    probes = list(range(0, n, max(1, n // 64)))
    if n and probes[-1] != n - 1:
        probes.append(n - 1)
    for i in probes:
        item = dataset[i]
        for field in (item[0], item[1], item[2]):
            digest.update(str(field).encode())
    return digest.hexdigest()


def shard_path(spec: ShardSpec, task_name: str, rank: int) -> str:
    return os.path.join(spec.directory, f"{task_name}.shard{rank}of{spec.world}.pkl")


def write_shard(spec: ShardSpec, task_name: str, payload: Dict[str, Any]) -> None:
    """Atomically publish this shard's payload (readers never see a partial file)."""
    os.makedirs(spec.directory, exist_ok=True)
    path = shard_path(spec, task_name, spec.rank)
    payload = {**payload, "run_id": spec.run_id}
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: str, attempts: int = 5, delay: float = 1.0) -> Dict[str, Any]:
    """Load a shard file, retrying briefly: on network filesystems a freshly renamed file
    can be visible before its contents are."""
    for attempt in range(attempts):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, FileNotFoundError):
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


def load_all_shards(spec: ShardSpec, task_name: str) -> Optional[List[Dict[str, Any]]]:
    """Payloads of all shards ordered by rank.

    ``None`` while any shard is missing. A file left by a different run (other ``run_id``)
    counts as missing: the shard that owns it will overwrite it when it finishes.
    """
    shards = []
    for rank in range(spec.world):
        path = shard_path(spec, task_name, rank)
        if not os.path.exists(path):
            return None
        payload = _load_pickle(path)
        if payload.get("run_id") != spec.run_id:
            return None
        shards.append(payload)
    return shards
