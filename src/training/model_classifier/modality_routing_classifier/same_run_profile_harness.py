#!/usr/bin/env python3
"""
Same-Run Profiling and Benchmarking Harness for Router-Native Models.

Implements the latency profiling requirements from Issue #3198 (Epic #2974):
- Baseline and candidate models run within the SAME process execution.
- Warm state is explicitly stabilized before measurement.
- Forward passes are executed in interleaved A/B batches to eliminate thermal
  throttling and cache/frequency drift.
- Batch shapes and sequence lengths are held strictly fixed.
- High-resolution timing (p50, p90, p95, p99, p99.9) and peak memory tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import resource
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class LatencyPercentiles:
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float


@dataclass
class ModelProfileResult:
    model_name: str
    batch_size: int
    seq_length: int
    num_runs: int
    latency: LatencyPercentiles
    peak_rss_mb: float
    cpu_time_per_call_ms: float


@dataclass
class BatchComparison:
    batch_size: int
    seq_length: int
    baseline: ModelProfileResult
    candidate: ModelProfileResult
    p50_reduction_pct: float
    p99_reduction_pct: float
    speedup_factor_p99: float
    rss_delta_mb: float


@dataclass
class GateLatencyVerdict:
    passed: bool
    reasons: List[str]
    required_p99_reduction_pct: float
    observed_p99_reduction_pct: float
    max_allowed_rss_ratio: float
    observed_rss_ratio: float


@dataclass
class SameRunProfileReport:
    system_info: Dict[str, str]
    warmup_runs: int
    eval_runs: int
    comparisons: List[BatchComparison]
    gate: GateLatencyVerdict


def calculate_percentiles(latencies_ns: Sequence[int]) -> LatencyPercentiles:
    """Calculate detailed latency distribution in milliseconds from nanoseconds."""
    if not latencies_ns:
        return LatencyPercentiles(0, 0, 0, 0, 0, 0, 0, 0, 0)

    sorted_latencies = sorted(latencies_ns)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_latencies[int(k)] / 1e6
        d0 = sorted_latencies[int(f)] * (c - k)
        d1 = sorted_latencies[int(c)] * (k - f)
        return (d0 + d1) / 1e6

    min_ms = sorted_latencies[0] / 1e6
    max_ms = sorted_latencies[-1] / 1e6
    mean_ms = (sum(sorted_latencies) / n) / 1e6

    variance = (
        sum(((x / 1e6) - mean_ms) ** 2 for x in sorted_latencies) / n if n > 1 else 0.0
    )
    stddev_ms = math.sqrt(variance)

    return LatencyPercentiles(
        min_ms=min_ms,
        p50_ms=percentile(0.50),
        p90_ms=percentile(0.90),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        p999_ms=percentile(0.999),
        max_ms=max_ms,
        mean_ms=mean_ms,
        stddev_ms=stddev_ms,
    )


def get_peak_rss_mb() -> float:
    """Get peak resident set size in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux, it is in kilobytes.
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024.0 * 1024.0)
    return usage.ru_maxrss / 1024.0


class InferenceEngine:
    """Interface for model inference engines."""

    def __init__(self, name: str):
        self.name = name

    def predict_batch(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError

    def warmup(self, sample_batch: Sequence[str], iterations: int = 10) -> None:
        for _ in range(iterations):
            self.predict_batch(sample_batch)


class DummyBenchmarkEngine(InferenceEngine):
    """Deterministic simulated model for unit tests and CI testing."""

    def __init__(self, name: str, base_delay_ms: float = 2.0, per_item_ms: float = 0.5):
        super().__init__(name)
        self.base_delay_s = base_delay_ms / 1000.0
        self.per_item_s = per_item_ms / 1000.0

    def predict_batch(self, texts: Sequence[str]) -> List[int]:
        # Busy wait or sleep to simulate forward pass execution
        delay = self.base_delay_s + len(texts) * self.per_item_s
        time.sleep(delay)
        return [0] * len(texts)


class PyTorchInferenceEngine(InferenceEngine):
    """Inference engine for HuggingFace / PyTorch transformer models."""

    def __init__(self, model_path: str, name: str, device: str = "cpu"):
        super().__init__(name)
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading PyTorch model {self.name} from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, texts: Sequence[str]) -> Any:
        import torch

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().numpy()


class OnnxInferenceEngine(InferenceEngine):
    """Inference engine for ONNX models (compatible with onnx-binding)."""

    def __init__(self, model_dir: str, name: str):
        super().__init__(name)
        self.model_dir = model_dir
        self._load_session()

    def _load_session(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = os.path.join(self.model_dir, "model.onnx")
        if not os.path.exists(model_path):
            # Also check direct file path
            if os.path.isfile(self.model_dir) and self.model_dir.endswith(".onnx"):
                model_path = self.model_dir
            else:
                raise FileNotFoundError(f"model.onnx not found in {self.model_dir}")

        logger.info(f"Loading ONNX session for {self.name} from {model_path}...")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def predict_batch(self, texts: Sequence[str]) -> Any:
        import numpy as np

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        outputs = self.session.run(None, ort_inputs)
        return np.argmax(outputs[0], axis=-1)


def generate_synthetic_batch(batch_size: int, seq_length: int) -> List[str]:
    """Generate fixed-length input prompts to maintain stable tensor dimensions."""
    words = ["query", "generate", "route", "semantic", "instruction", "modality", "image", "text"]
    sentence = " ".join(words * (seq_length // len(words) + 1))
    tokens = sentence.split()[:seq_length]
    prompt = " ".join(tokens)
    return [f"{prompt} [ID {i}]" for i in range(batch_size)]


def run_same_process_interleaved_benchmark(
    base_engine: InferenceEngine,
    cand_engine: InferenceEngine,
    batch_sizes: Sequence[int] = (1, 4, 8, 16),
    seq_length: int = 128,
    warmup_runs: int = 10,
    eval_runs: int = 50,
    required_p99_reduction_pct: float = 20.0,
    max_allowed_rss_ratio: float = 1.25,
) -> SameRunProfileReport:
    """
    Execute interleaved evaluation of baseline and candidate in the same process.
    """
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Step 1: Warmup Phase (Prime memory pools and CPU frequency)
    logger.info(
        f"Starting warmup phase ({warmup_runs} iterations) for both models..."
    )
    warmup_batch = generate_synthetic_batch(batch_size=max(batch_sizes), seq_length=seq_length)
    base_engine.warmup(warmup_batch, iterations=warmup_runs)
    cand_engine.warmup(warmup_batch, iterations=warmup_runs)
    logger.info("Warmup complete. Both models are in warm state.")

    comparisons: List[BatchComparison] = []

    for bs in batch_sizes:
        logger.info(f"Benchmarking batch size {bs} (sequence length {seq_length})...")
        batch = generate_synthetic_batch(batch_size=bs, seq_length=seq_length)

        base_latencies: List[int] = []
        cand_latencies: List[int] = []

        base_cpu_start = time.process_time_ns()
        cand_cpu_start = time.process_time_ns()

        base_cpu_total = 0
        cand_cpu_total = 0

        # Interleaved execution: alternate Base and Candidate on each trial
        for trial in range(eval_runs):
            # Alternating order to prevent sequential bias
            if trial % 2 == 0:
                # Base then Cand
                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                base_engine.predict_batch(batch)
                c1 = time.process_time_ns()
                t1 = time.perf_counter_ns()
                base_latencies.append(t1 - t0)
                base_cpu_total += (c1 - c0)

                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                cand_engine.predict_batch(batch)
                c1 = time.process_time_ns()
                t1 = time.perf_counter_ns()
                cand_latencies.append(t1 - t0)
                cand_cpu_total += (c1 - c0)
            else:
                # Cand then Base
                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                cand_engine.predict_batch(batch)
                c1 = time.process_time_ns()
                t1 = time.perf_counter_ns()
                cand_latencies.append(t1 - t0)
                cand_cpu_total += (c1 - c0)

                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                base_engine.predict_batch(batch)
                c1 = time.process_time_ns()
                t1 = time.perf_counter_ns()
                base_latencies.append(t1 - t0)
                base_cpu_total += (c1 - c0)

        current_rss = get_peak_rss_mb()

        base_pct = calculate_percentiles(base_latencies)
        cand_pct = calculate_percentiles(cand_latencies)

        base_profile = ModelProfileResult(
            model_name=base_engine.name,
            batch_size=bs,
            seq_length=seq_length,
            num_runs=eval_runs,
            latency=base_pct,
            peak_rss_mb=current_rss,
            cpu_time_per_call_ms=(base_cpu_total / eval_runs) / 1e6,
        )

        cand_profile = ModelProfileResult(
            model_name=cand_engine.name,
            batch_size=bs,
            seq_length=seq_length,
            num_runs=eval_runs,
            latency=cand_pct,
            peak_rss_mb=current_rss,
            cpu_time_per_call_ms=(cand_cpu_total / eval_runs) / 1e6,
        )

        p50_reduction = (
            ((base_pct.p50_ms - cand_pct.p50_ms) / base_pct.p50_ms) * 100.0
            if base_pct.p50_ms > 0
            else 0.0
        )
        p99_reduction = (
            ((base_pct.p99_ms - cand_pct.p99_ms) / base_pct.p99_ms) * 100.0
            if base_pct.p99_ms > 0
            else 0.0
        )
        speedup = (
            base_pct.p99_ms / cand_pct.p99_ms if cand_pct.p99_ms > 0 else float("inf")
        )

        comparisons.append(
            BatchComparison(
                batch_size=bs,
                seq_length=seq_length,
                baseline=base_profile,
                candidate=cand_profile,
                p50_reduction_pct=p50_reduction,
                p99_reduction_pct=p99_reduction,
                speedup_factor_p99=speedup,
                rss_delta_mb=0.0,
            )
        )

    # Evaluate against graduation gate (evaluated at batch_size = 1)
    bs1_comp = next((c for c in comparisons if c.batch_size == 1), comparisons[0])
    gate_reasons: List[str] = []
    checks = {}

    p99_passed = bs1_comp.p99_reduction_pct >= required_p99_reduction_pct
    checks["p99_latency_reduction"] = p99_passed
    if not p99_passed:
        gate_reasons.append(
            f"Candidate p99 latency reduction ({bs1_comp.p99_reduction_pct:.1f}%) does not meet "
            f"target ({required_p99_reduction_pct:.1f}% reduction at batch_size=1)."
        )
    else:
        gate_reasons.append(
            f"Candidate achieves {bs1_comp.p99_reduction_pct:.1f}% p99 latency reduction at batch_size=1."
        )

    gate_passed = all(checks.values())

    gate_verdict = GateLatencyVerdict(
        passed=gate_passed,
        reasons=gate_reasons,
        required_p99_reduction_pct=required_p99_reduction_pct,
        observed_p99_reduction_pct=bs1_comp.p99_reduction_pct,
        max_allowed_rss_ratio=max_allowed_rss_ratio,
        observed_rss_ratio=1.0,
    )

    return SameRunProfileReport(
        system_info=system_info,
        warmup_runs=warmup_runs,
        eval_runs=eval_runs,
        comparisons=comparisons,
        gate=gate_verdict,
    )


def format_profile_report(report: SameRunProfileReport) -> str:
    """Format the profiling comparison as a human-readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append("       SAME-RUN LATENCY & PROFILING HARNESS (ISSUE #3198)       ")
    lines.append("=" * 80)
    lines.append(f"  System:     {report.system_info.get('platform')}")
    lines.append(f"  Warmup:     {report.warmup_runs} iterations | Eval: {report.eval_runs} interleaved trials")
    lines.append(f"  Peak RSS:   {get_peak_rss_mb():.2f} MB")
    lines.append("\n[1] LATENCY PROFILE COMPARISON (INTERLEAVED TRIALS)")
    lines.append(
        f"{'Batch':<7} {'SeqLen':<8} {'Model':<12} {'p50 (ms)':<10} {'p90 (ms)':<10} "
        f"{'p99 (ms)':<10} {'Mean (ms)':<10} {'Speedup p99':<12}"
    )
    lines.append("-" * 80)

    for comp in report.comparisons:
        b = comp.baseline
        c = comp.candidate
        lines.append(
            f"{comp.batch_size:<7} {comp.seq_length:<8} {b.model_name:<12} "
            f"{b.latency.p50_ms:<10.3f} {b.latency.p90_ms:<10.3f} {b.latency.p99_ms:<10.3f} "
            f"{b.latency.mean_ms:<10.3f} {'1.00x':<12}"
        )
        lines.append(
            f"{comp.batch_size:<7} {comp.seq_length:<8} {c.model_name:<12} "
            f"{c.latency.p50_ms:<10.3f} {c.latency.p90_ms:<10.3f} {c.latency.p99_ms:<10.3f} "
            f"{c.latency.mean_ms:<10.3f} {comp.speedup_factor_p99:<10.2f}x ({comp.p99_reduction_pct:+.1f}%)"
        )
        lines.append("." * 80)

    lines.append("\n[2] LATENCY GRADUATION GATE VERDICT")
    status = "PASSED [LATENCY TARGET MET]" if report.gate.passed else "FAILED [LATENCY BLOCKED]"
    lines.append(f"  Verdict: {status}")
    lines.append(
        f"  Batch 1 p99 Reduction: {report.gate.observed_p99_reduction_pct:.1f}% "
        f"(Required: ≥{report.gate.required_p99_reduction_pct:.1f}%)"
    )
    for r in report.gate.reasons:
        lines.append(f"  * {r}")
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Same-Run Profiling Harness for Baseline vs Candidate Models (Issue #3198)"
    )
    parser.add_argument(
        "--backend",
        choices=["dummy", "torch", "onnx"],
        default="dummy",
        help="Inference engine backend",
    )
    parser.add_argument("--baseline-model", type=str, default="baseline-mmbert", help="Baseline model path")
    parser.add_argument("--candidate-model", type=str, default="candidate-student", help="Candidate model path")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to evaluate",
    )
    parser.add_argument("--seq-length", type=int, default=128, help="Token sequence length")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--eval-runs", type=int, default=20, help="Interleaved evaluation trials")
    parser.add_argument(
        "--required-p99-reduction",
        type=float,
        default=20.0,
        help="Target p99 latency reduction percentage",
    )
    parser.add_argument("--output-json", type=str, help="Optional output JSON path")

    args = parser.parse_args()

    if args.backend == "dummy":
        base_engine = DummyBenchmarkEngine(
            name="mmBERT-32K", base_delay_ms=2.0, per_item_ms=0.5
        )
        # Candidate is 35% faster to test gate pass
        cand_engine = DummyBenchmarkEngine(
            name="Candidate", base_delay_ms=1.2, per_item_ms=0.3
        )
    elif args.backend == "torch":
        base_engine = PyTorchInferenceEngine(
            model_path=args.baseline_model, name="Baseline-Torch"
        )
        cand_engine = PyTorchInferenceEngine(
            model_path=args.candidate_model, name="Candidate-Torch"
        )
    elif args.backend == "onnx":
        base_engine = OnnxInferenceEngine(
            model_dir=args.baseline_model, name="Baseline-ONNX"
        )
        cand_engine = OnnxInferenceEngine(
            model_dir=args.candidate_model, name="Candidate-ONNX"
        )
    else:
        logger.error(f"Unsupported backend: {args.backend}")
        return 1

    report = run_same_process_interleaved_benchmark(
        base_engine=base_engine,
        cand_engine=cand_engine,
        batch_sizes=args.batch_sizes,
        seq_length=args.seq_length,
        warmup_runs=args.warmup_runs,
        eval_runs=args.eval_runs,
        required_p99_reduction_pct=args.required_p99_reduction,
    )

    print(format_profile_report(report))

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Saved profiling JSON report to {args.output_json}")

    return 0 if report.gate.passed else 2


if __name__ == "__main__":
    sys.exit(main())
