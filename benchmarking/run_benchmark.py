"""Automated benchmark runner.

Runs the full CrossModalConsistencyPipeline against the synthetic fixture set
(app/../benchmarking/attack_fixtures.py) and reports precision/recall/F1 per
attack category, plus overall accuracy. Writes a Markdown report to
benchmarking/report.md.

Usage:
    python -m benchmarking.run_benchmark [--n-per-category 10] [--use-pretrained]

By default this runs in heuristic-only mode (`--use-pretrained` off) so it
works without torch/transformers/internet access -- useful as a CI smoke test
and as a quick sanity check after touching detector thresholds. Pass
`--use-pretrained` to exercise the ViT/CLIP-backed path once you have the
real dependencies installed and (optionally) a fine-tuned checkpoint.

IMPORTANT: the fixtures here are procedurally generated, not real photos or
real deepfakes -- they validate that the *harness and fusion logic* behave
correctly, not that the detectors generalize to real adversarial content.
Point `build_fixture_set` at a real labeled dataset (FaceForensics++, DFDC,
CASIA v2) for meaningful accuracy numbers.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarking.attack_fixtures import build_fixture_set
from app.pipeline.consistency_pipeline import CrossModalConsistencyPipeline


def run(n_per_category: int = 10, use_pretrained_models: bool = False) -> dict:
    pipeline = CrossModalConsistencyPipeline(use_pretrained_models=use_pretrained_models)
    fixtures = build_fixture_set(n_per_category=n_per_category)

    per_category = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    rows = []

    for f in fixtures:
        verdict = pipeline.run(f.image_bytes, caption=f.caption)
        predicted_authentic = verdict.verdict == "authentic"
        actual_authentic = f.expected_authentic

        bucket = per_category[f.category]
        if predicted_authentic and actual_authentic:
            bucket["tp"] += 1
        elif not predicted_authentic and not actual_authentic:
            bucket["tn"] += 1
        elif predicted_authentic and not actual_authentic:
            bucket["fp"] += 1
        else:
            bucket["fn"] += 1

        rows.append({
            "name": f.name,
            "category": f.category,
            "expected_authentic": actual_authentic,
            "predicted_authentic": predicted_authentic,
            "verdict": verdict.verdict,
            "unified_score": verdict.unified_score,
        })

    report = {"per_category": {}, "rows": rows}
    total_correct = 0
    for category, counts in per_category.items():
        tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
        n = tp + tn + fp + fn
        # For attack categories (non-"clean"), "positive" = correctly caught as fake.
        if category == "clean":
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
        else:
            precision = tn / (tn + fn) if (tn + fn) else float("nan")
            recall = tn / (tn + fp) if (tn + fp) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
        accuracy = (tp + tn) / n if n else float("nan")
        total_correct += tp + tn
        report["per_category"][category] = {
            "n": n, "accuracy": round(accuracy, 4),
            "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None,
            "f1": round(f1, 4) if f1 == f1 else None,
        }

    report["overall_accuracy"] = round(total_correct / len(fixtures), 4) if fixtures else None
    report["n_fixtures"] = len(fixtures)
    report["use_pretrained_models"] = use_pretrained_models
    return report


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# Benchmark Report",
        "",
        f"- Fixtures: {report['n_fixtures']} (synthetic, see `attack_fixtures.py`)",
        f"- Pretrained models: {report['use_pretrained_models']}",
        f"- Overall accuracy: **{report['overall_accuracy']}**",
        "",
        "| Category | N | Accuracy | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for category, m in report["per_category"].items():
        lines.append(f"| {category} | {m['n']} | {m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} |")
    lines.append("")
    lines.append("_Synthetic fixtures validate the fusion/harness logic, not real-world generalization. "
                  "Re-run against a real labeled dataset for meaningful numbers._")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-category", type=int, default=10)
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--out", type=str, default="benchmarking/report.md")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    report = run(n_per_category=args.n_per_category, use_pretrained_models=args.use_pretrained)
    write_markdown_report(report, Path(args.out))
    print(f"Overall accuracy: {report['overall_accuracy']}")
    for category, m in report["per_category"].items():
        print(f"  {category:24s} acc={m['accuracy']} precision={m['precision']} recall={m['recall']} f1={m['f1']}")
    print(f"Report written to {args.out}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
