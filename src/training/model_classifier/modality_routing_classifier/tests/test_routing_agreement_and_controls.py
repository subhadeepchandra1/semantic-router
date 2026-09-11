"""Unit and contract tests for routing agreement and fixed-policy controls evaluator.

Tests the evaluation criteria defined in Issue #3198 (Epic #2974):
- Agreement rate calculation
- Disagreement attribution (candidate win vs baseline win vs both incorrect)
- Fixed policy controls (always-cheapest, always-strongest, best fixed split at matched cost)
- Graduation gate enforcement
"""

from __future__ import annotations

import unittest

from src.training.model_classifier.modality_routing_classifier.evaluate_routing_agreement_and_controls import (
    DEFAULT_MODALITY_COSTS,
    compute_classification_metrics,
    compute_fixed_policy_controls,
    compute_routing_agreement,
    evaluate_graduation_gate,
    evaluate_routing_experiment,
    find_best_fixed_split,
)


class RoutingAgreementTest(unittest.TestCase):
    def test_perfect_agreement(self):
        y_true = ["AR", "DIFFUSION", "BOTH", "AR"]
        y_base = ["AR", "DIFFUSION", "BOTH", "AR"]
        y_cand = ["AR", "DIFFUSION", "BOTH", "AR"]

        summary = compute_routing_agreement(y_true, y_base, y_cand)
        self.assertEqual(summary.total_samples, 4)
        self.assertEqual(summary.agree_count, 4)
        self.assertEqual(summary.disagree_count, 0)
        self.assertAlmostEqual(summary.agree_rate, 1.0)
        self.assertEqual(summary.net_value, 0)
        self.assertEqual(summary.both_correct, 4)

    def test_disagreement_attribution(self):
        # Sample 1: Cand correct (DIFFUSION), Base wrong (AR) -> Cand Win
        # Sample 2: Base correct (AR), Cand wrong (BOTH) -> Base Win
        # Sample 3: Both agree and correct (BOTH) -> Agreement
        # Sample 4: Both wrong (True is AR, Cand says BOTH, Base says DIFFUSION) -> Disagreement, both wrong
        y_true = ["DIFFUSION", "AR", "BOTH", "AR"]
        y_base = ["AR", "AR", "BOTH", "DIFFUSION"]
        y_cand = ["DIFFUSION", "BOTH", "BOTH", "BOTH"]

        summary = compute_routing_agreement(y_true, y_base, y_cand)
        self.assertEqual(summary.total_samples, 4)
        self.assertEqual(summary.agree_count, 1)  # Only sample 3
        self.assertEqual(summary.disagree_count, 3)
        self.assertEqual(summary.cand_wins, 1)  # Sample 1
        self.assertEqual(summary.base_wins, 1)  # Sample 2
        self.assertEqual(summary.both_incorrect, 1)  # Sample 4
        self.assertEqual(summary.net_value, 0)


class ClassificationMetricsTest(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = ["AR", "AR", "DIFFUSION", "BOTH"]
        y_pred = ["AR", "DIFFUSION", "DIFFUSION", "BOTH"]
        labels = ["AR", "DIFFUSION", "BOTH"]

        summary = compute_classification_metrics(y_true, y_pred, labels)
        self.assertEqual(summary.total_samples, 4)
        self.assertAlmostEqual(summary.accuracy, 0.75)
        # AR: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=0.6667
        # DIFFUSION: TP=1, FP=1, FN=0 -> Prec=0.5, Rec=1.0, F1=0.6667
        # BOTH: TP=1, FP=0, FN=0 -> Prec=1.0, Rec=1.0, F1=1.0
        self.assertAlmostEqual(summary.per_class_precision["AR"], 1.0)
        self.assertAlmostEqual(summary.per_class_recall["DIFFUSION"], 1.0)
        self.assertAlmostEqual(summary.per_class_f1["BOTH"], 1.0)


class FixedPolicyControlsTest(unittest.TestCase):
    def test_best_fixed_split_within_budget(self):
        labels = ["AR", "DIFFUSION", "BOTH"]
        costs = {"AR": 1.0, "DIFFUSION": 5.0, "BOTH": 6.0}
        # 60% AR, 30% DIFFUSION, 10% BOTH
        y_true = ["AR"] * 60 + ["DIFFUSION"] * 30 + ["BOTH"] * 10

        # With a budget of 1.0, policy must choose 100% AR
        alloc, acc, cost = find_best_fixed_split(y_true, labels, costs, target_budget=1.0)
        self.assertAlmostEqual(alloc["AR"], 1.0)
        self.assertAlmostEqual(acc, 0.60)
        self.assertAlmostEqual(cost, 1.0)

        # With a higher budget, e.g. 5.0, cost cannot exceed 5.0
        alloc_high, acc_high, cost_high = find_best_fixed_split(
            y_true, labels, costs, target_budget=5.0
        )
        self.assertLessEqual(cost_high, 5.0001)

    def test_controls_generation(self):
        labels = ["AR", "DIFFUSION", "BOTH"]
        costs = DEFAULT_MODALITY_COSTS
        y_true = ["AR"] * 50 + ["DIFFUSION"] * 30 + ["BOTH"] * 20

        controls = compute_fixed_policy_controls(
            y_true, labels, costs, target_budget=2.5
        )
        control_names = [c.name for c in controls]
        self.assertIn("always-cheapest", control_names)
        self.assertIn("always-strongest", control_names)
        self.assertIn("best-fixed-split-matched-cost", control_names)

        cheapest = next(c for c in controls if c.name == "always-cheapest")
        self.assertAlmostEqual(cheapest.expected_accuracy, 0.50)
        self.assertAlmostEqual(cheapest.expected_cost_per_request, 1.0)


class GraduationGateTest(unittest.TestCase):
    def test_candidate_passes_gate(self):
        # Ground truth: 10 samples
        y_true = ["AR", "AR", "AR", "AR", "DIFFUSION", "DIFFUSION", "DIFFUSION", "BOTH", "BOTH", "BOTH"]
        # Baseline: 8/10 correct
        y_base = ["AR", "AR", "AR", "AR", "DIFFUSION", "DIFFUSION", "BOTH", "BOTH", "BOTH", "AR"]
        # Candidate: 9/10 correct (wins on sample 7 where base predicted BOTH)
        y_cand = ["AR", "AR", "AR", "AR", "DIFFUSION", "DIFFUSION", "DIFFUSION", "BOTH", "BOTH", "AR"]

        report = evaluate_routing_experiment(
            y_true=y_true,
            y_base=y_base,
            y_cand=y_cand,
            labels=["AR", "DIFFUSION", "BOTH"],
            tolerance_macro_f1=0.05,
            min_agreement_rate=0.80,
            min_control_accuracy_delta=0.01,
        )

        self.assertTrue(report.gate.passed)
        self.assertGreaterEqual(report.agreement.agree_rate, 0.80)
        self.assertGreater(report.agreement.net_value, 0)
        self.assertGreater(report.best_fixed_split_delta_accuracy, 0)

    def test_candidate_fails_when_losing_disagreements(self):
        y_true = ["AR", "DIFFUSION", "BOTH", "AR", "DIFFUSION"]
        # Baseline is 100% correct
        y_base = ["AR", "DIFFUSION", "BOTH", "AR", "DIFFUSION"]
        # Candidate is wrong on 2 samples
        y_cand = ["AR", "AR", "BOTH", "AR", "AR"]

        report = evaluate_routing_experiment(
            y_true=y_true,
            y_base=y_base,
            y_cand=y_cand,
            labels=["AR", "DIFFUSION", "BOTH"],
        )

        self.assertFalse(report.gate.passed)
        self.assertFalse(report.gate.checks["positive_disagreement_attribution"])


if __name__ == "__main__":
    unittest.main()
