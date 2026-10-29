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

"""Helpers that keep model inference from stalling the GPU on host-side work.

Both helpers are result-preserving: they only change *when* and *how* data moves
between host and device, never the values that are computed.
"""
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import numpy as np
import torch

_staging = threading.local()
_MAX_STAGING_BYTES = 1 << 30


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Equivalent to ``tensor.detach().cpu().numpy()`` but faster for large CUDA tensors.

    A pageable device-to-host copy into freshly allocated memory runs at a fraction of
    the PCIe bandwidth. Here the copy goes through a reused pinned staging buffer and is
    then duplicated into a new array, so the returned array is independent of the buffer
    (callers such as ``SequenceInferenceCache`` keep views of it). The bytes are identical
    to the plain path.
    """
    tensor = tensor.detach()
    if tensor.device.type != "cuda" or tensor.numel() == 0:
        return tensor.cpu().numpy()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes > _MAX_STAGING_BYTES:  # never pin multi-GB buffers for the lifetime of the process
        return tensor.cpu().numpy()
    try:
        buf = getattr(_staging, "buf", None)
        if buf is None or buf.numel() < nbytes:
            grown = min(max(nbytes, 2 * buf.numel()), _MAX_STAGING_BYTES) if buf is not None else nbytes
            buf = torch.empty(grown, dtype=torch.uint8, pin_memory=True)
            _staging.buf = buf
        staged = buf[:nbytes].view(tensor.dtype).view(tensor.shape)
        staged.copy_(tensor)  # blocking copy: data is complete on return
        return staged.numpy().copy()
    except (RuntimeError, TypeError):
        return tensor.cpu().numpy()


class BackgroundPrefetcher:
    """Run keyed jobs on one background thread and hand each result to the caller once.

    The consumer submits the jobs for the *next* batch while the GPU works on the current
    one, then ``take``s a result when it needs it. ``take`` returns ``None`` on a miss or a
    failed job, so callers always keep their direct code path as the fallback.

    All bookkeeping happens on the calling thread; only the submitted function runs on the
    worker. A single worker serializes jobs, so a function may own non-thread-safe state
    (for example a private tokenizer instance) as long as only prefetch jobs touch it.

    Entries are kept for two generations (the batch being processed and the next one);
    older unconsumed entries are dropped so memory stays bounded.
    """

    def __init__(self) -> None:
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: Dict[Hashable, Tuple[int, Future]] = {}
        self._generation = 0

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def new_generation(self) -> None:
        """Start a new batch generation and drop entries from two generations ago."""
        self._generation += 1
        for key in [k for k, (gen, _) in self._pending.items() if gen < self._generation - 1]:
            self._pending.pop(key)[1].cancel()

    def submit(self, key: Hashable, fn: Callable[..., Any], *args: Any) -> None:
        if key in self._pending:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gfm-prefetch")
        self._pending[key] = (self._generation, self._executor.submit(fn, *args))

    def take(self, key: Hashable) -> Optional[Any]:
        entry = self._pending.pop(key, None)
        if entry is None:
            return None
        try:
            return entry[1].result()
        except Exception:
            return None

    def clear(self) -> None:
        for _, future in self._pending.values():
            future.cancel()
        self._pending.clear()
