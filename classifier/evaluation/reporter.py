import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from classifier.evaluation.metrics import DatasetEvaluationResult

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Exports benchmark results to Markdown, CSV, and JSON report files."""

    @classmethod
    def export_all(
        cls,
        results: Dict[str, DatasetEvaluationResult],
        output_dir: str,
        iou_threshold: float = 0.50,
    ) -> Dict[str, str]:
        """
        Exports all reports (Markdown, CSV, JSON) into output_dir.

        Returns:
            Dict mapping report type -> output file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        md_path = out_path / "benchmark_summary.md"
        csv_path = out_path / "benchmark_summary.csv"
        class_csv_path = out_path / "per_class_summary.csv"
        json_path = out_path / "evaluation_results.json"

        cls.export_markdown(results, str(md_path), iou_threshold=iou_threshold)
        cls.export_csv(results, str(csv_path), str(class_csv_path))
        cls.export_json(results, str(json_path))

        logger.info(f"[EvaluationReporter] Successfully generated all evaluation reports in: {out_path}")
        return {
            "markdown": str(md_path),
            "csv_summary": str(csv_path),
            "csv_classes": str(class_csv_path),
            "json": str(json_path),
        }

    @classmethod
    def export_markdown(
        cls,
        results: Dict[str, DatasetEvaluationResult],
        file_path: str,
        iou_threshold: float = 0.50,
    ) -> str:
        """Generates a clean Markdown comparison report."""
        lines = [
            "# Vision Model Benchmark Summary",
            "",
            f"**Matching Criteria**: IoU $\\ge$ {iou_threshold:.0%} Overlap + Correct Food Class Label",
            "",
            "## 1. Overall Model Comparison",
            "",
            "| Model | Extracted (Preds) | Ground Truth | Matches (TP) | False Pos | Missed (FN) | Precision | Recall | F1 Score | Latency (ms) | FPS |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        # Find best performers
        best_f1 = max((r.f1_score for r in results.values()), default=0.0)
        best_prec = max((r.precision for r in results.values()), default=0.0)
        best_rec = max((r.recall for r in results.values()), default=0.0)
        fastest_ms = min((r.latency_ms_per_image for r in results.values() if r.latency_ms_per_image > 0), default=0.0)

        for name, r in results.items():
            prec_str = f"**{r.precision:.1%}**" if r.precision == best_prec and best_prec > 0 else f"{r.precision:.1%}"
            rec_str = f"**{r.recall:.1%}**" if r.recall == best_rec and best_rec > 0 else f"{r.recall:.1%}"
            f1_str = f"**{r.f1_score:.1%}**" if r.f1_score == best_f1 and best_f1 > 0 else f"{r.f1_score:.1%}"
            lat_str = f"**{r.latency_ms_per_image:.1f}**" if r.latency_ms_per_image == fastest_ms else f"{r.latency_ms_per_image:.1f}"

            lines.append(
                f"| **{name}** | {r.total_extracted} | {r.total_in_gt} | {r.total_matches} | "
                f"{r.total_false_positives} | {r.total_false_negatives} | {prec_str} | {rec_str} | "
                f"{f1_str} | {lat_str} ms | {r.fps:.1f} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Per-Class Performance Breakdown",
            "",
        ])

        for name, r in results.items():
            lines.extend([
                f"### Model: {name}",
                "",
                "| Food Item | Ground Truth | Extracted | Matches (TP) | False Pos | Missed (FN) | Precision | Recall | F1 Score |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for cls_name, cm in r.per_class.items():
                lines.append(
                    f"| `{cls_name}` | {cm.total_in_gt} | {cm.extracted} | {cm.matches} | "
                    f"{cm.false_positives} | {cm.false_negatives} | {cm.precision:.1%} | {cm.recall:.1%} | {cm.f1_score:.1%} |"
                )
            lines.append("")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return file_path

    @classmethod
    def export_csv(
        cls,
        results: Dict[str, DatasetEvaluationResult],
        summary_path: str,
        class_summary_path: str,
    ) -> None:
        """Exports metrics as CSV files for easy ingestion."""
        # 1. Summary CSV
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name",
                "total_images",
                "total_extracted",
                "total_in_gt",
                "total_matches",
                "total_false_positives",
                "total_false_negatives",
                "precision",
                "recall",
                "f1_score",
                "macro_precision",
                "macro_recall",
                "macro_f1_score",
                "latency_ms_per_image",
                "fps",
            ])
            for name, r in results.items():
                writer.writerow([
                    name,
                    r.total_images,
                    r.total_extracted,
                    r.total_in_gt,
                    r.total_matches,
                    r.total_false_positives,
                    r.total_false_negatives,
                    r.precision,
                    r.recall,
                    r.f1_score,
                    r.macro_precision,
                    r.macro_recall,
                    r.macro_f1_score,
                    r.latency_ms_per_image,
                    r.fps,
                ])

        # 2. Per-Class CSV
        with open(class_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name",
                "class_name",
                "total_in_gt",
                "extracted",
                "matches",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1_score",
            ])
            for name, r in results.items():
                for cls_name, cm in r.per_class.items():
                    writer.writerow([
                        name,
                        cls_name,
                        cm.total_in_gt,
                        cm.extracted,
                        cm.matches,
                        cm.false_positives,
                        cm.false_negatives,
                        cm.precision,
                        cm.recall,
                        cm.f1_score,
                    ])

    @classmethod
    def export_json(cls, results: Dict[str, DatasetEvaluationResult], file_path: str) -> str:
        """Serializes full results including per-image detections to JSON."""
        serializable = {name: r.model_dump() for name, r in results.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        return file_path
