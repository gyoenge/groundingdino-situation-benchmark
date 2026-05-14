import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUNDING_DINO_ROOT = PROJECT_ROOT / "GroundingDINO"

sys.path.append(str(GROUNDING_DINO_ROOT))

import torch
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)

from src.io_utils import load_json, save_json


def resolve_path(path: str | Path, root_dir: Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(root_dir / path)


def to_xyxy(box, image_width: int, image_height: int):
    """
    GroundingDINO predict output box format:
    normalized cxcywh
    -> absolute xyxy
    """
    cx, cy, w, h = box.tolist()

    x1 = (cx - w / 2) * image_width
    y1 = (cy - h / 2) * image_height
    x2 = (cx + w / 2) * image_width
    y2 = (cy + h / 2) * image_height

    return [
        float(max(0, x1)),
        float(max(0, y1)),
        float(min(image_width, x2)),
        float(min(image_height, y2)),
    ]


DEFAULT_CONFIG_PATH = (
    GROUNDING_DINO_ROOT
    / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

DEFAULT_CHECKPOINT_PATH = (
    GROUNDING_DINO_ROOT
    / "weights/groundingdino_swint_ogc.pth"
)


class GroundingDINOInferencer:
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = load_model(
            model_config_path=str(config_path),
            model_checkpoint_path=str(checkpoint_path),
            device=self.device,
        )

    def predict_one(
        self,
        image_path: str,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        height, width = image_source.shape[:2]

        results = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            results.append({
                "bbox": to_xyxy(box, width, height),
                "score": float(logit.item()),
                "phrase": str(phrase),
            })

        return results


def run_inference(
    annotation_path: str,
    output_path: str,
    config_path: str,
    checkpoint_path: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
    root_dir: str | Path = ".",
):
    root_dir = Path(root_dir).resolve()

    annotation_path = resolve_path(annotation_path, root_dir)
    output_path = resolve_path(output_path, root_dir)
    config_path = resolve_path(config_path, root_dir)
    checkpoint_path = resolve_path(checkpoint_path, root_dir)

    samples = load_json(annotation_path)

    inferencer = GroundingDINOInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    results = []

    for sample in samples:
        image_path = resolve_path(sample["image_path"], root_dir)

        for prompt_item in sample["prompts"]:
            prompt = prompt_item["prompt"]

            predictions = inferencer.predict_one(
                image_path=image_path,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            results.append({
                "image_id": sample["image_id"],
                "prompt_id": prompt_item["prompt_id"],
                "prompt": prompt,
                "type": prompt_item["type"],
                "expected": prompt_item["expected"],
                "predictions": predictions,
            })

            print(
                f"[DONE] {sample['image_id']} | "
                f"{prompt_item['prompt_id']} | "
                f"{prompt} | n={len(predictions)}"
            )

    save_json(results, output_path)
    print(f"[SAVE] predictions -> {output_path}")
