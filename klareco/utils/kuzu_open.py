"""Centralized Kuzu database opener.

By default this helper does **not** cap ``buffer_pool_size`` — that's
Kuzu's job, and the production v2.1 graph (~10 GB on disk) genuinely
needs Kuzu's 80%-of-RAM default for multi-hop traversals (a 4 GiB cap
was observed to OOM with "buffer pool is full" exceptions on
biographical queries).

The cap matters in **parallel** workloads where N workers would each
try to claim 80% of RAM; for those, set the environment variable so a
single knob bounds every Kuzu instance the process opens:

    KLARECO_KUZU_BUFFER_MB    cap on buffer pool size (megabytes)
    KLARECO_KUZU_MAX_THREADS  cap on Kuzu's worker thread count

See ``scripts/local_parallel_bench.sh`` for the canonical
``RAM/2/N_workers`` formula.

Read-only is the default — query workloads share the DB file across
processes safely. Writers (loaders, schema migrations) pass
``read_only=False``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import kuzu


def open_kuzu(
    db_path: Union[str, Path],
    *,
    read_only: bool = True,
) -> kuzu.Database:
    """Open a Kuzu database, honoring memory/thread env vars when set.

    Falls back to a no-kwarg open on older Kuzu versions that don't
    accept the modern keyword arguments.
    """
    kwargs: dict = {'read_only': read_only}

    if mb := os.environ.get('KLARECO_KUZU_BUFFER_MB'):
        kwargs['buffer_pool_size'] = int(mb) * 1024 * 1024
    if n := os.environ.get('KLARECO_KUZU_MAX_THREADS'):
        kwargs['max_num_threads'] = int(n)

    try:
        return kuzu.Database(str(db_path), **kwargs)
    except TypeError:
        return kuzu.Database(str(db_path))
