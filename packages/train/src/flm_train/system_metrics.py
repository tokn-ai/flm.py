"""Background collection of wall-clock system metrics."""

from __future__ import annotations

import csv
import os
import resource
import subprocess
import threading
from collections.abc import Callable
from io import StringIO

from flm_train.sinks.base import JsonValue

SystemMetrics = dict[str, JsonValue]


class SystemMetricsSampler:
  def __init__(
    self,
    *,
    every_seconds: float,
    emit: Callable[[SystemMetrics], None],
    collect: Callable[[], SystemMetrics] | None = None,
  ) -> None:
    if every_seconds <= 0:
      raise ValueError("every_seconds must be positive")
    self.every_seconds = every_seconds
    self.emit = emit
    self.collect = collect or collect_system_metrics
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    if self._thread is not None:
      raise RuntimeError("system metrics sampler is already running")
    self._thread = threading.Thread(
      target=self._run,
      name="flm-system-metrics",
      daemon=True,
    )
    self._thread.start()

  def stop(self) -> None:
    if self._thread is None:
      return
    self._stop.set()
    self._thread.join(timeout=max(1.0, self.every_seconds + 1.0))
    self._thread = None

  def _run(self) -> None:
    while not self._stop.is_set():
      try:
        self.emit(self.collect())
      except Exception:
        pass
      self._stop.wait(self.every_seconds)


def collect_system_metrics() -> SystemMetrics:
  payload: SystemMetrics = {
    "process": _collect_process_metrics(),
    "gpus": _collect_gpu_metrics(),
  }
  return payload


def _collect_process_metrics() -> SystemMetrics:
  usage = resource.getrusage(resource.RUSAGE_SELF)
  return {
    "pid": os.getpid(),
    "max_rss_bytes": int(usage.ru_maxrss) * 1024,
  }


def _collect_gpu_metrics() -> list[SystemMetrics]:
  nvidia_smi_metrics = _collect_nvidia_smi_metrics()
  if nvidia_smi_metrics:
    return nvidia_smi_metrics
  return _collect_torch_cuda_metrics()


def _collect_nvidia_smi_metrics() -> list[SystemMetrics]:
  query = ",".join(
    [
      "index",
      "name",
      "utilization.gpu",
      "utilization.memory",
      "memory.used",
      "memory.total",
      "temperature.gpu",
      "power.draw",
    ]
  )
  try:
    completed = subprocess.run(
      [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
      ],
      capture_output=True,
      check=False,
      text=True,
      timeout=2.0,
    )
  except (FileNotFoundError, subprocess.SubprocessError, OSError):
    return []
  if completed.returncode != 0:
    return []

  rows = csv.reader(StringIO(completed.stdout))
  metrics: list[SystemMetrics] = []
  for row in rows:
    if len(row) != 8:
      continue
    metrics.append(
      {
        "source": "nvidia-smi",
        "index": _to_int(row[0]),
        "name": row[1].strip(),
        "utilization_pct": _to_float(row[2]),
        "memory_utilization_pct": _to_float(row[3]),
        "memory_used_mb": _to_float(row[4]),
        "memory_total_mb": _to_float(row[5]),
        "temperature_c": _to_float(row[6]),
        "power_draw_w": _to_float(row[7]),
      }
    )
  return metrics


def _collect_torch_cuda_metrics() -> list[SystemMetrics]:
  try:
    import torch
  except ImportError:
    return []
  if not torch.cuda.is_available():
    return []

  metrics: list[SystemMetrics] = []
  for index in range(torch.cuda.device_count()):
    metrics.append(
      {
        "source": "torch.cuda",
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
      }
    )
  return metrics


def _to_float(value: str) -> float | None:
  value = value.strip()
  if not value or value == "N/A":
    return None
  return float(value)


def _to_int(value: str) -> int | None:
  number = _to_float(value)
  if number is None:
    return None
  return int(number)
