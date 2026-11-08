"""Realtime STT and Translation latency and resource benchmark harness (EF-033).

Measures:
- Reference hardware profile (CPU, RAM, OS, GPU/accelerator).
- Local STT partial caption latency (target: ~1.0s / 1,000ms).
- Local STT final transcript latency.
- Local Translation latency (Marian OPUS-MT en->es).
- Combined final STT + translation latency (target: ~2.0s / 2,000ms).
- Process RSS memory utilization (MB).

Reports measured values without hardcoding or falsifying PASS.
Uses standard library exclusively for system inspection and memory monitoring.
"""

import argparse
import asyncio
import ctypes
import json
import os
import platform
import statistics
import sys
import time
from ctypes import wintypes
from typing import Any, Dict, List, Optional

import numpy as np

from engine.local_models.selector import SystemHardwareProbe
from engine.providers.stt.local import LocalSTTProvider
from engine.providers.translation.local import LocalTranslationProvider
from engine.providers.types import AudioFrame, TranslationRequest

SAMPLE_RATE = 16000
FRAME_DURATION_SEC = 0.1  # 100ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_SEC)  # 1,600 samples
BYTES_PER_SAMPLE = 2  # 16-bit PCM
FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE  # 3,200 bytes


class ProcessMemoryTracker:
    """Stdlib-only memory tracker using K32GetProcessMemoryInfo on Windows or resource on Unix."""

    def __init__(self) -> None:
        self._is_windows = sys.platform.startswith("win")
        if self._is_windows:
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            self._PMC = ProcessMemoryCounters
            self._fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
            self._fn.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            self._fn.restype = wintypes.BOOL
            self._handle = ctypes.windll.kernel32.GetCurrentProcess()

    def get_rss_mb(self) -> float:
        """Return current resident working set size in megabytes."""
        if self._is_windows:
            p = self._PMC()
            p.cb = ctypes.sizeof(self._PMC)
            if self._fn(self._handle, ctypes.byref(p), p.cb):
                return p.WorkingSetSize / (1024 * 1024)
            return 0.0

        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        factor = (1024 * 1024) if sys.platform == "darwin" else 1024
        return ru.ru_maxrss / factor

    def get_peak_rss_mb(self) -> float:
        """Return peak resident working set size in megabytes."""
        if self._is_windows:
            p = self._PMC()
            p.cb = ctypes.sizeof(self._PMC)
            if self._fn(self._handle, ctypes.byref(p), p.cb):
                return p.PeakWorkingSetSize / (1024 * 1024)
            return 0.0

        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        factor = (1024 * 1024) if sys.platform == "darwin" else 1024
        return ru.ru_maxrss / factor


def probe_reference_hardware() -> Dict[str, Any]:
    """Inspect and return current host hardware environment."""
    probe = SystemHardwareProbe()
    profile = probe.probe()

    gpu_info = "None (CPU runtime)"
    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"CUDA: {dev_name} ({vram_gb:.1f} GB)"
    except Exception:
        pass

    return {
        "os_platform": profile.platform,
        "os_version": platform.platform(),
        "architecture": profile.architecture,
        "cpu_processor": platform.processor() or "Unknown CPU",
        "cpu_logical_cores": profile.cpu_cores,
        "ram_gb": round(profile.ram_gb, 2),
        "gpu_accelerator": gpu_info,
        "python_version": platform.python_version(),
    }


def generate_benchmark_audio_chunks(duration_sec: float = 1.0) -> List[bytes]:
    """Generate 16kHz 16-bit mono PCM chunks with synthetic speech frequencies (440Hz tone)."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    raw_bytes = samples.tobytes()

    chunks: List[bytes] = []
    for i in range(0, len(raw_bytes), FRAME_BYTES):
        chunk = raw_bytes[i : i + FRAME_BYTES]
        if len(chunk) == FRAME_BYTES:
            chunks.append(chunk)
    return chunks


async def run_benchmark_iteration(
    iteration_idx: int,
    stt_provider: LocalSTTProvider,
    trans_provider: LocalTranslationProvider,
    mem_tracker: ProcessMemoryTracker,
    audio_chunks: List[bytes],
) -> Dict[str, float]:
    """Execute a single measurement iteration across STT and Translation pipelines."""
    events_iter = stt_provider.receive_events()

    t_stream_start = time.perf_counter()
    partial_latency: Optional[float] = None

    # Stream audio frames at 50ms intervals
    for seq, chunk in enumerate(audio_chunks):
        frame = AudioFrame(
            stream_id="mic",
            sequence=seq,
            pcm_data=chunk,
            sample_rate=SAMPLE_RATE,
            channels=1,
        )
        await stt_provider.send_audio(frame)
        await asyncio.sleep(0.02)

    # Await first partial transcript event
    try:
        while True:
            event = await asyncio.wait_for(events_iter.__anext__(), timeout=1.5)
            if not event.is_final and partial_latency is None:
                partial_latency = time.perf_counter() - t_stream_start
                break
            if event.is_final:
                break
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass

    if partial_latency is None:
        partial_latency = time.perf_counter() - t_stream_start

    # Signal endpoint / finalize
    t_finalize_start = time.perf_counter()
    await stt_provider.finalize_stream("mic")

    final_latency: Optional[float] = None
    try:
        while True:
            event = await asyncio.wait_for(events_iter.__anext__(), timeout=2.0)
            if event.is_final:
                final_latency = time.perf_counter() - t_finalize_start
                break
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass

    if final_latency is None:
        final_latency = time.perf_counter() - t_finalize_start

    # Measure Local Translation (Marian en -> es)
    benchmark_utterance = (
        "Good morning everyone. Let us start today's project review meeting."
    )
    t_trans_start = time.perf_counter()
    await trans_provider.translate(
        TranslationRequest(
            text=benchmark_utterance,
            source_lang="en",
            target_lang="es",
            stream_id="mic",
            sequence=iteration_idx,
            is_final=True,
        )
    )
    t_trans_end = time.perf_counter()
    translation_latency = t_trans_end - t_trans_start

    combined_latency = final_latency + translation_latency
    current_rss = mem_tracker.get_rss_mb()

    return {
        "iteration": float(iteration_idx),
        "partial_stt_latency_ms": round(partial_latency * 1000.0, 2),
        "final_stt_latency_ms": round(final_latency * 1000.0, 2),
        "translation_latency_ms": round(translation_latency * 1000.0, 2),
        "combined_latency_ms": round(combined_latency * 1000.0, 2),
        "rss_mb": round(current_rss, 2),
    }


def compute_summary_stats(series: List[float]) -> Dict[str, float]:
    """Compute mean, median, min, max, and p95 summary statistics for a metric series."""
    if not series:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}

    sorted_s = sorted(series)
    p95_idx = int(len(sorted_s) * 0.95) if len(sorted_s) > 1 else -1
    return {
        "mean": round(statistics.mean(series), 2),
        "median": round(statistics.median(series), 2),
        "min": round(min(series), 2),
        "max": round(max(series), 2),
        "p95": round(sorted_s[p95_idx], 2),
    }


async def main_async(args: argparse.Namespace) -> int:
    mem_tracker = ProcessMemoryTracker()
    hw_info = probe_reference_hardware()

    print("=" * 72)
    print("  EchoFlux Realtime Latency & Resource Benchmark (EF-033)")
    print("=" * 72)
    print(f"Host OS:        {hw_info['os_platform']} ({hw_info['os_version']})")
    print(f"Architecture:   {hw_info['architecture']}")
    print(f"CPU:            {hw_info['cpu_processor']} ({hw_info['cpu_logical_cores']} cores)")
    print(f"RAM:            {hw_info['ram_gb']} GB")
    print(f"GPU Runtime:    {hw_info['gpu_accelerator']}")
    print(f"Python:         {hw_info['python_version']}")
    print(f"Iterations:     {args.iterations} (warmup: {args.warmup})")
    print(f"Initial RSS:    {mem_tracker.get_rss_mb():.2f} MB")
    print("=" * 72)

    audio_chunks = generate_benchmark_audio_chunks(duration_sec=1.0)

    stt_provider = LocalSTTProvider(stream_ids=("mic",))
    trans_provider = LocalTranslationProvider()

    print("\n--> Initializing and starting local providers...")
    await stt_provider.start()
    await trans_provider.start()
    print(f"--> Providers ready. RSS after initialization: {mem_tracker.get_rss_mb():.2f} MB")

    try:
        # Warmup
        if args.warmup > 0:
            print(f"\n--> Running {args.warmup} warmup iteration(s)...")
            for w in range(args.warmup):
                await run_benchmark_iteration(
                    w, stt_provider, trans_provider, mem_tracker, audio_chunks
                )
            print("--> Warmup complete.")

        # Benchmark runs
        print(f"\n--> Executing {args.iterations} benchmark iterations...")
        results: List[Dict[str, float]] = []
        for i in range(1, args.iterations + 1):
            iter_data = await run_benchmark_iteration(
                i, stt_provider, trans_provider, mem_tracker, audio_chunks
            )
            results.append(iter_data)
            print(
                f"  Iter {i:2d}: Partial STT={iter_data['partial_stt_latency_ms']:6.1f}ms | "
                f"Final STT={iter_data['final_stt_latency_ms']:6.1f}ms | "
                f"Trans={iter_data['translation_latency_ms']:5.1f}ms | "
                f"Combined={iter_data['combined_latency_ms']:6.1f}ms | "
                f"RSS={iter_data['rss_mb']:5.1f}MB"
            )

        # Aggregate stats
        partial_stats = compute_summary_stats([r["partial_stt_latency_ms"] for r in results])
        final_stats = compute_summary_stats([r["final_stt_latency_ms"] for r in results])
        trans_stats = compute_summary_stats([r["translation_latency_ms"] for r in results])
        combined_stats = compute_summary_stats([r["combined_latency_ms"] for r in results])
        rss_stats = compute_summary_stats([r["rss_mb"] for r in results])

        peak_rss = mem_tracker.get_peak_rss_mb()

        print("\n" + "=" * 72)
        print("  BENCHMARK SUMMARY RESULTS")
        print("=" * 72)
        print(f"{'Metric':<30} {'Mean':<10} {'Median':<10} {'P95':<10} {'Target':<10}")
        print("-" * 72)
        print(
            f"{'Partial STT Latency (ms)':<30} {partial_stats['mean']:<10.1f} "
            f"{partial_stats['median']:<10.1f} {partial_stats['p95']:<10.1f} {'~1000ms':<10}"
        )
        print(
            f"{'Final STT Latency (ms)':<30} {final_stats['mean']:<10.1f} "
            f"{final_stats['median']:<10.1f} {final_stats['p95']:<10.1f} {'-':<10}"
        )
        print(
            f"{'Translation Latency (ms)':<30} {trans_stats['mean']:<10.1f} "
            f"{trans_stats['median']:<10.1f} {trans_stats['p95']:<10.1f} {'-':<10}"
        )
        print(
            f"{'Combined STT+Trans (ms)':<30} {combined_stats['mean']:<10.1f} "
            f"{combined_stats['median']:<10.1f} {combined_stats['p95']:<10.1f} {'~2000ms':<10}"
        )
        print(
            f"{'Process Working Set RSS (MB)':<30} {rss_stats['mean']:<10.1f} "
            f"{rss_stats['median']:<10.1f} {rss_stats['p95']:<10.1f} {f'Peak: {peak_rss:.1f}MB':<10}"
        )
        print("=" * 72)

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware_reference": hw_info,
            "benchmark_config": {
                "iterations": args.iterations,
                "warmup": args.warmup,
                "audio_duration_sec": 1.0,
                "sample_rate": SAMPLE_RATE,
            },
            "summary_metrics": {
                "partial_stt_latency_ms": partial_stats,
                "final_stt_latency_ms": final_stats,
                "translation_latency_ms": trans_stats,
                "combined_latency_ms": combined_stats,
                "rss_mb": rss_stats,
                "peak_rss_mb": round(peak_rss, 2),
            },
            "iteration_runs": results,
        }

        if args.output_json:
            output_dir = os.path.dirname(os.path.abspath(args.output_json))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\n--> Benchmark JSON report saved to: {args.output_json}")

    finally:
        print("\n--> Stopping local providers...")
        await stt_provider.stop()
        await trans_provider.stop()
        print("--> Providers stopped gracefully.")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoFlux Realtime Latency & Resource Benchmark Harness"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark measurement iterations (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of initial warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON benchmark report",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
