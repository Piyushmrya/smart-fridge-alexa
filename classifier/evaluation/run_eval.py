"""
Command-line runner for the Bounding Box Evaluation Framework.

Usage:
    python -m classifier.evaluation.run_eval
    python -m classifier.evaluation.run_eval --config classifier/evaluation/eval_config.json
    python -m classifier.evaluation.run_eval --model YOLO-World-Small --iou-threshold 0.5
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from classifier.evaluation.config import EvalConfig
from classifier.evaluation.evaluator import BenchmarkRunner, DatasetLoader
from classifier.evaluation.reporter import EvaluationReporter
from classifier.evaluation.visualizer import generate_benchmark_visualizations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluation_runner")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "eval_config.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Precision/Recall/F1 Bounding Box Evaluation across vision models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to evaluation JSON configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter evaluation to a specific model by name (e.g. 'YOLO-World-Small')",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Override IoU threshold for matching (default 0.50)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Override confidence threshold for all evaluated models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and visualizations",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable generating annotated visualization images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Loading evaluation configuration from: {args.config}")
    cfg = EvalConfig.from_json_file(args.config)

    # Apply CLI overrides if provided
    if args.iou_threshold is not None:
        cfg.matching.iou_threshold = args.iou_threshold
    if args.conf_threshold is not None:
        for m in cfg.models:
            m.confidence_threshold = args.conf_threshold
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.no_viz:
        cfg.save_visualizations = False

    # Filter to single model if requested
    if args.model is not None:
        filtered = [m for m in cfg.models if m.name.lower() == args.model.lower() or m.detector_type.lower() == args.model.lower()]
        if not filtered:
            available = [m.name for m in cfg.models]
            logger.error(f"Model '{args.model}' not found in config! Available models: {available}")
            sys.exit(1)
        cfg.models = filtered

    logger.info(f"Starting evaluation across {len(cfg.models)} model(s)...")
    logger.info(f"Matching IoU threshold: {cfg.matching.iou_threshold:.0%}")
    logger.info(f"Dataset images directory: {cfg.dataset.images_dir}")
    logger.info(f"Annotations file: {cfg.dataset.annotations_path}")

    # Load dataset
    dataset = DatasetLoader.load(cfg.dataset)
    if not dataset:
        logger.error("No valid images/annotations loaded. Exiting.")
        sys.exit(1)

    runner = BenchmarkRunner(cfg)
    results = runner.run()

    # Generate visualizations if enabled
    if cfg.save_visualizations:
        for model_name, res in results.items():
            logger.info(f"Generating visual debug overlays for {model_name}...")
            generate_benchmark_visualizations(
                model_name=model_name,
                dataset=dataset,
                eval_results=res.per_image,
                output_dir=cfg.output_dir,
            )

    # Export reports
    report_paths = EvaluationReporter.export_all(
        results=results,
        output_dir=cfg.output_dir,
        iou_threshold=cfg.matching.iou_threshold,
    )

    # Print clean summary table to stdout
    print("\n" + "=" * 80)
    print("                    EVALUATION BENCHMARK RESULTS")
    print(f"       IoU Match Threshold: {cfg.matching.iou_threshold:.0%}  |  Total Images: {len(dataset)}")
    print("=" * 80)
    print(f"{'Model':<20} | {'Extr':<5} | {'GT':<5} | {'Match':<5} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Latency':<9}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<20} | {r.total_extracted:<5} | {r.total_in_gt:<5} | {r.total_matches:<5} | "
            f"{r.precision:<7.1%} | {r.recall:<7.1%} | {r.f1_score:<7.1%} | {r.latency_ms_per_image:<6.1f}ms"
        )
    print("=" * 80)
    print(f"\n📊 Markdown Summary: {report_paths['markdown']}")
    print(f"📁 Full Results JSON: {report_paths['json']}")
    print(f"📈 Per-Class CSV:     {report_paths['csv_classes']}")
    if cfg.save_visualizations:
        print(f"🖼️  Visual Overlays:   {os.path.join(cfg.output_dir, 'visualizations')}")
    print()


if __name__ == "__main__":
    main()
