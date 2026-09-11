"""Unit and contract tests for the same-run latency and profiling harness.

Tests the harness requirements defined in Issue #3198 (Epic #2974):
- Percentile calculations
- Same-process interleaved trial execution
- Batch shape handling
- Latency graduation gate verification
"""

from __future__ import annotations

import unittest

from src.training.model_classifier.modality_routing_classifier.same_run_profile_harness import (
    DummyBenchmarkEngine,
    calculate_percentiles,
    format_profile_report,
    generate_synthetic_batch,
    get_peak_rss_mb,
    run_same_process_interleaved_benchmark,
)


class PercentileCalculationTest(unittest.TestCase):
    def test_percentiles_monotonicity(self):
        # 100 values from 1ms to 100ms
        latencies_ns = [i * 1_000_000 for i in range(1, 101)]
        p = calculate_percentiles(latencies_ns)

        self.assertAlmostEqual(p.min_ms, 1.0)
        self.assertAlmostEqual(p.max_ms, 100.0)
        self.assertAlmostEqual(p.p50_ms, 50.5, delta=1.0)
        self.assertAlmostEqual(p.p90_ms, 90.1, delta=1.0)
        self.assertAlmostEqual(p.p99_ms, 99.0, delta=1.0)
        self.assertAlmostEqual(p.mean_ms, 50.5)
        self.assertGreater(p.stddev_ms, 0)

    def test_empty_list(self):
        p = calculate_percentiles([])
        self.assertEqual(p.p50_ms, 0.0)
        self.assertEqual(p.p99_ms, 0.0)


class SyntheticBatchTest(unittest.TestCase):
    def test_batch_dimensions(self):
        batch = generate_synthetic_batch(batch_size=4, seq_length=32)
        self.assertEqual(len(batch), 4)
        for text in batch:
            self.assertTrue(len(text.split()) >= 30)


class SameRunBenchmarkTest(unittest.TestCase):
    def test_interleaved_benchmark_execution(self):
        base_engine = DummyBenchmarkEngine("Base", base_delay_ms=2.0, per_item_ms=0.2)
        cand_engine = DummyBenchmarkEngine("Cand", base_delay_ms=1.0, per_item_ms=0.1)

        report = run_same_process_interleaved_benchmark(
            base_engine=base_engine,
            cand_engine=cand_engine,
            batch_sizes=[1, 2],
            seq_length=16,
            warmup_runs=2,
            eval_runs=5,
            required_p99_reduction_pct=20.0,
        )

        self.assertEqual(len(report.comparisons), 2)
        b1_comp = report.comparisons[0]
        self.assertEqual(b1_comp.batch_size, 1)
        self.assertGreater(b1_comp.baseline.latency.p99_ms, b1_comp.candidate.latency.p99_ms)
        self.assertTrue(report.gate.passed)
        self.assertGreater(report.gate.observed_p99_reduction_pct, 20.0)

        # Check formatted report renders cleanly
        text = format_profile_report(report)
        self.assertIn("SAME-RUN LATENCY & PROFILING HARNESS", text)
        self.assertIn("PASSED", text)

    def test_gate_fails_when_candidate_slower(self):
        # Base is faster than Cand
        base_engine = DummyBenchmarkEngine("Base", base_delay_ms=1.0, per_item_ms=0.1)
        cand_engine = DummyBenchmarkEngine("Cand", base_delay_ms=2.0, per_item_ms=0.2)

        report = run_same_process_interleaved_benchmark(
            base_engine=base_engine,
            cand_engine=cand_engine,
            batch_sizes=[1],
            seq_length=16,
            warmup_runs=2,
            eval_runs=5,
            required_p99_reduction_pct=20.0,
        )

        self.assertFalse(report.gate.passed)
        self.assertLess(report.gate.observed_p99_reduction_pct, 0.0)


class MemoryTrackingTest(unittest.TestCase):
    def test_rss_measurement(self):
        rss = get_peak_rss_mb()
        self.assertGreater(rss, 0.0)


if __name__ == "__main__":
    unittest.main()
