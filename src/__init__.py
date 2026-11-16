"""OSIRIS package defaults."""

from __future__ import annotations

import os


_cpu_count = os.cpu_count() or 2
os.environ.setdefault("POLARS_MAX_THREADS", str(max(1, min(4, _cpu_count // 2))))
