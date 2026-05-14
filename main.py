# main.py

import argparse

from src.run_grounding_dino import run_inference
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        default="data/annotations.json",
    )

    parser.add_argument(
        "--predictions",
        default="outputs/predictions.json",
    )

    parser.add_argument(
        "--report",
        default="outputs/evaluation_report.json",
    )

    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--eval_threshold", type=float, default=0.3)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_inference", action="store_true")

    args = parser.parse_args()

    if not args.skip_inference:
        run_inference(
            annotation_path=args.annotations,
            output_path=args.predictions,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )

    report = evaluate(
        prediction_path=args.predictions,
        output_path=args.report,
        threshold=args.eval_threshold,
    )

    print("=" * 80)
    print("Evaluation finished.")
    print(f"FSAR: {report['false_situation_activation_rate']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()