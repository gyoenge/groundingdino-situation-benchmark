from pathlib import Path
from PIL import Image

from src.io_utils import load_json, save_json


def predict_with_grounding_dino(image_path: str, prompt: str):
    """
    TODO: 여기에 실제 Grounding DINO inference 연결.

    Return format:
    [
        {
            "bbox": [x1, y1, x2, y2],
            "score": 0.73,
            "phrase": "person"
        }
    ]
    """

    # placeholder
    return []


def run_inference(annotation_path: str, output_path: str):
    samples = load_json(annotation_path)
    results = []

    for sample in samples:
        image_path = sample["image_path"]

        for prompt_item in sample["prompts"]:
            prompt = prompt_item["prompt"]

            predictions = predict_with_grounding_dino(
                image_path=image_path,
                prompt=prompt,
            )

            results.append({
                "image_id": sample["image_id"],
                "prompt_id": prompt_item["prompt_id"],
                "prompt": prompt,
                "type": prompt_item["type"],
                "expected": prompt_item["expected"],
                "predictions": predictions,
            })

    save_json(results, output_path)