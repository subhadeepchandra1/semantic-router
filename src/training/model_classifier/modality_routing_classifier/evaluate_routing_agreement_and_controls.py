#!/usr/bin/env python3
"""
Routing Agreement and Fixed-Policy Control Evaluator for Router-Native Models.

Implements the evaluation requirements specified in Issue #3198 (Epic #2974):
1. Routing Agreement Metric:
   - Evaluates per-request agreement between candidate and baseline.
   - Attribution on disagreements: assesses whether student or baseline matched
     ground truth when their decisions diverge.
2. Fixed-Policy Controls at Matched Cost:
   - Always-Cheapest (e.g. 100% AR)
   - Always-Strongest (e.g. 100% BOTH)
   - Best Fixed Split at Matched Cost (optimal static probability mixture under the
     candidate's empirical average request cost).
3. Automated Graduation Gate Check against formal acceptance criteria.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODALITY_LABELS = ["AR", "DIFFUSION", "BOTH"]

# Default operational costs per modality (relative computational cost)
# AR: 1.0 (text autoregressive token generation)
# DIFFUSION: 5.0 (image generation GPU steps)
# BOTH: 6.0 (text explanation + image synthesis)
DEFAULT_MODALITY_COSTS = {
    "AR": 1.0,
    "DIFFUSION": 5.0,
    "BOTH": 6.0,
}


@dataclass
class ClassificationSummary:
    total_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]


@dataclass
class AgreementSummary:
    total_samples: int
    agree_count: int
    agree_rate: float
    disagree_count: int
    disagree_rate: float
    cand_wins: int
    base_wins: int
    both_incorrect: int
    both_correct: int
    net_value: int
    cand_win_rate_on_disagreements: float


@dataclass
class FixedPolicyControl:
    name: str
    description: str
    expected_accuracy: float
    expected_cost_per_request: float
    allocation: Dict[str, float]


@dataclass
class GateEvaluationResult:
    passed: bool
    reasons: List[str]
    checks: Dict[str, bool]


@dataclass
class EvaluationReport:
    baseline_metrics: ClassificationSummary
    candidate_metrics: ClassificationSummary
    agreement: AgreementSummary
    controls: List[FixedPolicyControl]
    candidate_cost_per_request: float
    baseline_cost_per_request: float
    best_fixed_split_delta_accuracy: float
    gate: GateEvaluationResult


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> ClassificationSummary:
    """Compute accuracy, precision, recall, and F1 (macro and weighted)."""
    n = len(y_true)
    if n == 0:
        return ClassificationSummary(
            total_samples=0,
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            per_class_precision={},
            per_class_recall={},
            per_class_f1={},
            per_class_support={},
        )

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / n

    class_counts = Counter(y_true)
    precisions: Dict[str, float] = {}
    recalls: Dict[str, float] = {}
    f1s: Dict[str, float] = {}
    supports: Dict[str, int] = {}

    for lbl in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[lbl] = prec
        recalls[lbl] = rec
        f1s[lbl] = f1
        supports[lbl] = class_counts.get(lbl, 0)

    num_classes = len(labels)
    macro_f1 = sum(f1s[lbl] for lbl in labels) / num_classes if num_classes > 0 else 0.0

    weighted_f1 = (
        sum(f1s[lbl] * supports[lbl] for lbl in labels) / n if n > 0 else 0.0
    )

    return ClassificationSummary(
        total_samples=n,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_precision=precisions,
        per_class_recall=recalls,
        per_class_f1=f1s,
        per_class_support=supports,
    )


def compute_routing_agreement(
    y_true: Sequence[str],
    y_base: Sequence[str],
    y_cand: Sequence[str],
) -> AgreementSummary:
    """
    Compute per-request agreement and attribute disagreements against ground truth.
    """
    n = len(y_true)
    if n == 0:
        return AgreementSummary(
            total_samples=0,
            agree_count=0,
            agree_rate=0.0,
            disagree_count=0,
            disagree_rate=0.0,
            cand_wins=0,
            base_wins=0,
            both_incorrect=0,
            both_correct=0,
            net_value=0,
            cand_win_rate_on_disagreements=0.0,
        )

    agree_count = 0
    cand_wins = 0
    base_wins = 0
    both_incorrect = 0
    both_correct = 0

    for yt, yb, yc in zip(y_true, y_base, y_cand):
        if yb == yc:
            agree_count += 1
            if yc == yt:
                both_correct += 1
            else:
                both_incorrect += 1
        else:
            if yc == yt and yb != yt:
                cand_wins += 1
            elif yb == yt and yc != yt:
                base_wins += 1
            else:
                both_incorrect += 1

    disagree_count = n - agree_count
    agree_rate = agree_count / n
    disagree_rate = disagree_count / n

    net_value = cand_wins - base_wins
    decisive_disagreements = cand_wins + base_wins
    cand_win_rate = (
        cand_wins / decisive_disagreements if decisive_disagreements > 0 else 0.5
    )

    return AgreementSummary(
        total_samples=n,
        agree_count=agree_count,
        agree_rate=agree_rate,
        disagree_count=disagree_count,
        disagree_rate=disagree_rate,
        cand_wins=cand_wins,
        base_wins=base_wins,
        both_incorrect=both_incorrect,
        both_correct=both_correct,
        net_value=net_value,
        cand_win_rate_on_disagreements=cand_win_rate,
    )


def find_best_fixed_split(
    y_true: Sequence[str],
    labels: Sequence[str],
    costs: Mapping[str, float],
    target_budget: float,
    resolution: int = 100,
) -> Tuple[Dict[str, float], float, float]:
    """
    Find the optimal static allocation probability p over labels maximizing expected
    accuracy subject to sum(p_i * cost_i) <= target_budget and sum(p_i) == 1.

    Expected accuracy of static policy p is sum(p_i * prior_i),
    where prior_i = P(Y = i) in the evaluation ground truth.
    """
    n = len(y_true)
    if n == 0:
        return {lbl: 1.0 / len(labels) for lbl in labels}, 0.0, 0.0

    counts = Counter(y_true)
    priors = {lbl: counts.get(lbl, 0) / n for lbl in labels}

    # If 3 classes (e.g. AR, DIFFUSION, BOTH), we can do an exact grid search
    # over the probability simplex.
    best_acc = -1.0
    best_allocation: Dict[str, float] = {}
    best_cost = 0.0

    num_classes = len(labels)
    if num_classes == 3:
        l0, l1, l2 = labels[0], labels[1], labels[2]
        c0, c1, c2 = costs[l0], costs[l1], costs[l2]
        u0, u1, u2 = priors[l0], priors[l1], priors[l2]

        for i in range(resolution + 1):
            p0 = i / resolution
            remaining = resolution - i
            for j in range(remaining + 1):
                p1 = j / resolution
                p2 = (remaining - j) / resolution

                exp_cost = p0 * c0 + p1 * c1 + p2 * c2
                if exp_cost <= target_budget + 1e-6:
                    exp_acc = p0 * u0 + p1 * u1 + p2 * u2
                    if exp_acc > best_acc:
                        best_acc = exp_acc
                        best_cost = exp_cost
                        best_allocation = {l0: p0, l1: p1, l2: p2}
    else:
        # Fallback greedy LP knapsack approach for arbitrary number of classes:
        # Sort classes by efficiency (prior_i / cost_i)
        class_efficiency = sorted(
            labels,
            key=lambda l: (priors[l] / costs[l] if costs[l] > 0 else float("inf")),
            reverse=True,
        )
        # Check pure strategies first
        for l in labels:
            if costs[l] <= target_budget:
                if priors[l] > best_acc:
                    best_acc = priors[l]
                    best_cost = costs[l]
                    best_allocation = {lbl: (1.0 if lbl == l else 0.0) for lbl in labels}

        # If pure strategy doesn't exhaust budget, mix between cheapest and highest utility
        cheapest_label = min(labels, key=lambda l: costs[l])
        highest_label = max(labels, key=lambda l: priors[l])
        if costs[cheapest_label] <= target_budget < costs[highest_label]:
            denom = costs[highest_label] - costs[cheapest_label]
            alpha = (target_budget - costs[cheapest_label]) / denom if denom > 0 else 0.0
            alpha = max(0.0, min(1.0, alpha))
            exp_acc = (1 - alpha) * priors[cheapest_label] + alpha * priors[highest_label]
            if exp_acc > best_acc:
                best_acc = exp_acc
                best_cost = target_budget
                best_allocation = {
                    cheapest_label: (1.0 - alpha),
                    highest_label: alpha,
                }
                for l in labels:
                    if l not in best_allocation:
                        best_allocation[l] = 0.0

    if not best_allocation:
        # Fallback to cheapest label if budget is tight
        cheapest = min(labels, key=lambda l: costs[l])
        best_allocation = {lbl: (1.0 if lbl == cheapest else 0.0) for lbl in labels}
        best_acc = priors.get(cheapest, 0.0)
        best_cost = costs[cheapest]

    return best_allocation, best_acc, best_cost


def compute_fixed_policy_controls(
    y_true: Sequence[str],
    labels: Sequence[str],
    costs: Mapping[str, float],
    target_budget: float,
) -> List[FixedPolicyControl]:
    """Compute always-cheapest, always-strongest, and best-fixed-split at matched cost."""
    n = len(y_true)
    counts = Counter(y_true)
    priors = {lbl: counts.get(lbl, 0) / n if n > 0 else 0.0 for lbl in labels}

    cheapest_label = min(labels, key=lambda l: costs[l])
    strongest_label = max(labels, key=lambda l: costs[l])

    controls = [
        FixedPolicyControl(
            name="always-cheapest",
            description=f"Static policy routing 100% of requests to cheapest route ({cheapest_label}).",
            expected_accuracy=priors[cheapest_label],
            expected_cost_per_request=costs[cheapest_label],
            allocation={lbl: (1.0 if lbl == cheapest_label else 0.0) for lbl in labels},
        ),
        FixedPolicyControl(
            name="always-strongest",
            description=f"Static policy routing 100% of requests to strongest route ({strongest_label}).",
            expected_accuracy=priors[strongest_label],
            expected_cost_per_request=costs[strongest_label],
            allocation={lbl: (1.0 if lbl == strongest_label else 0.0) for lbl in labels},
        ),
    ]

    best_alloc, best_acc, best_cost = find_best_fixed_split(
        y_true, labels, costs, target_budget
    )
    controls.append(
        FixedPolicyControl(
            name="best-fixed-split-matched-cost",
            description=(
                f"Optimal static probability mixture subject to average cost <= {target_budget:.3f}."
            ),
            expected_accuracy=best_acc,
            expected_cost_per_request=best_cost,
            allocation=best_alloc,
        )
    )

    return controls


def evaluate_graduation_gate(
    base_summary: ClassificationSummary,
    cand_summary: ClassificationSummary,
    agreement: AgreementSummary,
    best_fixed_split: FixedPolicyControl,
    tolerance_macro_f1: float = 0.01,
    min_agreement_rate: float = 0.90,
    min_control_accuracy_delta: float = 0.03,
) -> GateEvaluationResult:
    """
    Check formal graduation gate criteria:
    1. Macro F1 parity with baseline (within tolerance).
    2. Routing agreement >= min_agreement_rate.
    3. Positive net value on disagreement (cand_wins > base_wins).
    4. Outperforms best cost-matched fixed split control by at least min_control_accuracy_delta.
    """
    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    # Check 1: Macro F1 Parity
    f1_delta = cand_summary.macro_f1 - base_summary.macro_f1
    parity_passed = f1_delta >= -tolerance_macro_f1
    checks["macro_f1_parity"] = parity_passed
    if not parity_passed:
        reasons.append(
            f"Candidate Macro F1 ({cand_summary.macro_f1:.4f}) lags baseline "
            f"({base_summary.macro_f1:.4f}) by {abs(f1_delta):.4f} (tolerance is {tolerance_macro_f1:.4f})."
        )

    # Check 2: Agreement Rate
    agreement_passed = agreement.agree_rate >= min_agreement_rate
    checks["routing_agreement_rate"] = agreement_passed
    if not agreement_passed:
        reasons.append(
            f"Routing agreement rate ({agreement.agree_rate:.2%}) is below threshold "
            f"({min_agreement_rate:.2%})."
        )

    # Check 3: Net Disagreement Attribution
    disagree_net_passed = (
        agreement.disagree_count == 0 or agreement.net_value >= 0
    )
    checks["positive_disagreement_attribution"] = disagree_net_passed
    if not disagree_net_passed:
        reasons.append(
            f"Candidate loses more disagreements ({agreement.base_wins}) than it wins "
            f"({agreement.cand_wins}); net value is {agreement.net_value}."
        )

    # Check 4: Outperform Fixed Split Control
    acc_delta = cand_summary.accuracy - best_fixed_split.expected_accuracy
    control_passed = acc_delta >= min_control_accuracy_delta
    checks["beat_fixed_split_control"] = control_passed
    if not control_passed:
        reasons.append(
            f"Candidate accuracy ({cand_summary.accuracy:.4f}) fails to beat best cost-matched "
            f"fixed split ({best_fixed_split.expected_accuracy:.4f}) by required delta "
            f"({acc_delta:.4f} vs required {min_control_accuracy_delta:.4f})."
        )

    overall_passed = all(checks.values())
    if overall_passed:
        reasons.append("All graduation gate criteria successfully cleared.")

    return GateEvaluationResult(
        passed=overall_passed,
        reasons=reasons,
        checks=checks,
    )


def evaluate_routing_experiment(
    y_true: Sequence[str],
    y_base: Sequence[str],
    y_cand: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    costs: Optional[Mapping[str, float]] = None,
    tolerance_macro_f1: float = 0.01,
    min_agreement_rate: float = 0.90,
    min_control_accuracy_delta: float = 0.03,
) -> EvaluationReport:
    """Main evaluation entrypoint for Issue #3198."""
    if len(y_true) != len(y_base) or len(y_true) != len(y_cand):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_base={len(y_base)}, y_cand={len(y_cand)}"
        )

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_base) | set(y_cand)))

    if costs is None:
        costs = {lbl: DEFAULT_MODALITY_COSTS.get(lbl, 1.0) for lbl in labels}

    n = len(y_true)
    cand_cost = sum(costs.get(yc, 1.0) for yc in y_cand) / n if n > 0 else 0.0
    base_cost = sum(costs.get(yb, 1.0) for yb in y_base) / n if n > 0 else 0.0

    base_summary = compute_classification_metrics(y_true, y_base, labels)
    cand_summary = compute_classification_metrics(y_true, y_cand, labels)
    agreement = compute_routing_agreement(y_true, y_base, y_cand)
    controls = compute_fixed_policy_controls(y_true, labels, costs, cand_cost)

    best_fixed = next(
        c for c in controls if c.name == "best-fixed-split-matched-cost"
    )
    delta_control = cand_summary.accuracy - best_fixed.expected_accuracy

    gate = evaluate_graduation_gate(
        base_summary,
        cand_summary,
        agreement,
        best_fixed,
        tolerance_macro_f1=tolerance_macro_f1,
        min_agreement_rate=min_agreement_rate,
        min_control_accuracy_delta=min_control_accuracy_delta,
    )

    return EvaluationReport(
        baseline_metrics=base_summary,
        candidate_metrics=cand_summary,
        agreement=agreement,
        controls=controls,
        candidate_cost_per_request=cand_cost,
        baseline_cost_per_request=base_cost,
        best_fixed_split_delta_accuracy=delta_control,
        gate=gate,
    )


def format_report_as_text(report: EvaluationReport) -> str:
    """Format evaluation report into a human-readable CLI summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("  ROUTER EXPERIMENT EVALUATION & FIXED-POLICY CONTROLS REPORT  ")
    lines.append("=" * 70)

    lines.append("\n[1] CLASSIFICATION PERFORMANCE SUMMARY")
    lines.append(f"{'Metric':<25} {'Baseline':<18} {'Candidate':<18} {'Delta':<10}")
    lines.append("-" * 70)
    acc_delta = report.candidate_metrics.accuracy - report.baseline_metrics.accuracy
    f1_delta = report.candidate_metrics.macro_f1 - report.baseline_metrics.macro_f1
    lines.append(
        f"{'Accuracy':<25} {report.baseline_metrics.accuracy:<18.4f} "
        f"{report.candidate_metrics.accuracy:<18.4f} {acc_delta:+0.4f}"
    )
    lines.append(
        f"{'Macro F1':<25} {report.baseline_metrics.macro_f1:<18.4f} "
        f"{report.candidate_metrics.macro_f1:<18.4f} {f1_delta:+0.4f}"
    )
    lines.append(
        f"{'Weighted F1':<25} {report.baseline_metrics.weighted_f1:<18.4f} "
        f"{report.candidate_metrics.weighted_f1:<18.4f} "
        f"{report.candidate_metrics.weighted_f1 - report.baseline_metrics.weighted_f1:+0.4f}"
    )
    lines.append(
        f"{'Avg Cost/Request':<25} {report.baseline_cost_per_request:<18.4f} "
        f"{report.candidate_cost_per_request:<18.4f} "
        f"{report.candidate_cost_per_request - report.baseline_cost_per_request:+0.4f}"
    )

    lines.append("\n[2] ROUTING AGREEMENT & DISAGREEMENT ATTRIBUTION")
    lines.append(f"  Total Requests Evaluated:  {report.agreement.total_samples}")
    lines.append(
        f"  Routing Agreement Rate:    {report.agreement.agree_rate:.2%} ({report.agreement.agree_count}/{report.agreement.total_samples})"
    )
    lines.append(
        f"  Routing Disagreement Rate: {report.agreement.disagree_rate:.2%} ({report.agreement.disagree_count}/{report.agreement.total_samples})"
    )
    if report.agreement.disagree_count > 0:
        lines.append("  Disagreement Breakdown:")
        lines.append(
            f"    - Candidate matched Ground Truth (Cand Win): {report.agreement.cand_wins} "
            f"({report.agreement.cand_wins/report.agreement.disagree_count:.1%})"
        )
        lines.append(
            f"    - Baseline matched Ground Truth (Base Win):  {report.agreement.base_wins} "
            f"({report.agreement.base_wins/report.agreement.disagree_count:.1%})"
        )
        lines.append(
            f"    - Both models incorrect:                   {report.agreement.both_incorrect} "
            f"({report.agreement.both_incorrect/report.agreement.disagree_count:.1%})"
        )
        lines.append(
            f"    - Net Disagreement Value (Cand - Base):     {report.agreement.net_value:+d}"
        )

    lines.append("\n[3] FIXED-POLICY CONTROLS (Empirical Cost Budget: {:.4f})".format(report.candidate_cost_per_request))
    lines.append(f"{'Policy Control':<30} {'Expected Cost':<15} {'Expected Accuracy':<18}")
    lines.append("-" * 70)
    for c in report.controls:
        lines.append(
            f"{c.name:<30} {c.expected_cost_per_request:<15.4f} {c.expected_accuracy:<18.4f}"
        )
    lines.append(
        f"{'Candidate Learned Router':<30} {report.candidate_cost_per_request:<15.4f} "
        f"{report.candidate_metrics.accuracy:<18.4f}"
    )
    lines.append(
        f"\n  Net Routing Value over Best Fixed Split: {report.best_fixed_split_delta_accuracy:+0.4f} "
        f"({'CLEAR GAIN' if report.best_fixed_split_delta_accuracy > 0 else 'NO ROUTING VALUE'})"
    )

    lines.append("\n[4] GRADUATION GATE VERDICT")
    status = "PASSED [GRADUATION READY]" if report.gate.passed else "FAILED [GATE BLOCKED]"
    lines.append(f"  Verdict: {status}")
    lines.append("  Check Criteria:")
    for k, v in report.gate.checks.items():
        lines.append(f"    - {k:<35}: {'PASS' if v else 'FAIL'}")
    lines.append("  Notes:")
    for r in report.gate.reasons:
        lines.append(f"    * {r}")
    lines.append("=" * 70)
    return "\n".join(lines)


def load_jsonl_column(filepath: Union[str, Path], col_name: str) -> List[str]:
    """Load a single column/field from a JSONL file."""
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if col_name in data:
                values.append(str(data[col_name]))
            else:
                raise KeyError(f"Key '{col_name}' missing on line {idx+1} of {filepath}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Routing Agreement & Cost-Matched Fixed Policy Controls (Issue #3198)"
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        help="JSONL file containing 'label' (ground truth), 'baseline_pred', and 'candidate_pred'",
    )
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth JSONL file")
    parser.add_argument("--baseline-preds", type=str, help="Path to baseline predictions JSONL file")
    parser.add_argument("--candidate-preds", type=str, help="Path to candidate predictions JSONL file")
    parser.add_argument("--label-key", type=str, default="label_name", help="Key for label name")
    parser.add_argument("--pred-key", type=str, default="pred", help="Key for prediction")
    parser.add_argument("--output-json", type=str, help="Optional output JSON report path")
    parser.add_argument("--tolerance-f1", type=float, default=0.01, help="Max allowed Macro F1 drop")
    parser.add_argument("--min-agreement", type=float, default=0.90, help="Min routing agreement rate")
    parser.add_argument(
        "--min-control-delta",
        type=float,
        default=0.03,
        help="Min accuracy advantage over best fixed split",
    )

    args = parser.parse_args()

    if args.eval_file:
        y_true = load_jsonl_column(args.eval_file, args.label_key)
        y_base = load_jsonl_column(args.eval_file, "baseline_pred")
        y_cand = load_jsonl_column(args.eval_file, "candidate_pred")
    elif args.ground_truth and args.baseline_preds and args.candidate_preds:
        y_true = load_jsonl_column(args.ground_truth, args.label_key)
        y_base = load_jsonl_column(args.baseline_preds, args.pred_key)
        y_cand = load_jsonl_column(args.candidate_preds, args.pred_key)
    else:
        logger.error("Must provide either --eval-file OR (--ground-truth, --baseline-preds, --candidate-preds)")
        return 1

    report = evaluate_routing_experiment(
        y_true=y_true,
        y_base=y_base,
        y_cand=y_cand,
        tolerance_macro_f1=args.tolerance_f1,
        min_agreement_rate=args.min_agreement,
        min_control_accuracy_delta=args.min_control_delta,
    )

    text_summary = format_report_as_text(report)
    print(text_summary)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Saved JSON report to {args.output_json}")

    return 0 if report.gate.passed else 2


if __name__ == "__main__":
    sys.exit(main())
