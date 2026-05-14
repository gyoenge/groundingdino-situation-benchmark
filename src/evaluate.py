from collections import defaultdict

from src.io_utils import load_json, save_json
from src.metrics import (
    max_score,
    situation_discrimination_score,
    prompt_specificity_gap,
    is_false_situation_activation,
)


def evaluate(prediction_path: str, output_path: str, threshold: float = 0.3):
    predictions = load_json(prediction_path)

    prompt_results = []
    grouped = defaultdict(list)

    false_activation_count = 0
    negative_situation_count = 0

    for item in predictions:
        score = max_score(item["predictions"])

        result = {
            "image_id": item["image_id"],
            "prompt_id": item["prompt_id"],
            "prompt": item["prompt"],
            "type": item["type"],
            "expected": item["expected"],
            "score": score,
            "activated": score >= threshold,
        }

        prompt_results.append(result)
        grouped[item["image_id"]].append(result)

        if item["type"] == "situation" and item["expected"] == "negative":
            negative_situation_count += 1
            if is_false_situation_activation(item["expected"], score, threshold):
                false_activation_count += 1

    fsar = (
        false_activation_count / negative_situation_count
        if negative_situation_count > 0
        else 0.0
    )

    image_level_results = []

    for image_id, items in grouped.items():
        positive_situations = [
            x for x in items
            if x["type"] == "situation" and x["expected"] == "positive"
        ]
        negative_situations = [
            x for x in items
            if x["type"] == "situation" and x["expected"] == "negative"
        ]
        compositions = [
            x for x in items
            if x["type"] == "composition"
        ]

        sds_values = []
        for pos in positive_situations:
            for neg in negative_situations:
                sds_values.append(
                    situation_discrimination_score(pos["score"], neg["score"])
                )

        psg_values = []
        for sit in positive_situations:
            for comp in compositions:
                psg_values.append(
                    prompt_specificity_gap(sit["score"], comp["score"])
                )

        image_level_results.append({
            "image_id": image_id,
            "mean_sds": sum(sds_values) / len(sds_values) if sds_values else None,
            "mean_psg": sum(psg_values) / len(psg_values) if psg_values else None,
        })

    report = {
        "threshold": threshold,
        "false_situation_activation_rate": fsar,
        "prompt_results": prompt_results,
        "image_level_results": image_level_results,
    }

    save_json(report, output_path)
    return report