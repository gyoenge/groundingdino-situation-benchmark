import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, unrealistic, extra limbs, distorted hands, "
    "bad anatomy, watermark, text"
)


SCENARIOS = [
    {
        "category": "dangerous_knife",
        "image_prompt": (
            "A realistic surveillance-style photo of a man threatening another person "
            "with a clearly visible knife in a dark alley, tense situation, natural lighting"
        ),
        "prompts": [
            ("person", "object", "positive"),
            ("knife", "object", "positive"),
            ("person with knife", "composition", "positive"),
            ("person holding knife", "relation", "positive"),
            ("a man threatening someone with a knife", "situation", "positive"),
            ("a chef cooking with a knife", "situation", "negative"),
        ],
    },
    {
        "category": "cooking_knife",
        "image_prompt": (
            "A realistic photo of a chef cooking with a knife in a kitchen, "
            "cutting vegetables on a cutting board, safe non-dangerous context"
        ),
        "prompts": [
            ("person", "object", "positive"),
            ("knife", "object", "positive"),
            ("person with knife", "composition", "positive"),
            ("person holding knife", "relation", "positive"),
            ("a chef cooking with a knife", "situation", "positive"),
            ("a man threatening someone with a knife", "situation", "negative"),
        ],
    },
    {
        "category": "person_dog",
        "image_prompt": (
            "A realistic outdoor photo of a person standing beside a dog in a park, "
            "calm everyday scene"
        ),
        "prompts": [
            ("person", "object", "positive"),
            ("dog", "object", "positive"),
            ("person and dog", "composition", "positive"),
            ("dog sitting beside person", "relation", "positive"),
            ("person attacking another person", "situation", "negative"),
        ],
    },
]


class SDXLGenerator:
    def __init__(self, model_id: str, device: str):
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self.device = device
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(device)

        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]


class GroundingDinoAutoAnnotator:
    def __init__(
        self,
        model_id: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)

    @torch.inference_mode()
    def detect(self, image: Image.Image, query: str) -> List[Dict]:
        inputs = self.processor(
            images=image,
            text=query,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        target_sizes = [image.size[::-1]]

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )

        result = results[0]
        objects = []

        for score, label, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"],
        ):
            label = str(label).lower().strip()
            norm_label = normalize_label(label)

            if norm_label is None:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]

            objects.append({
                "label": norm_label,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score.item()),
            })

        return keep_top_per_label(objects)


def normalize_label(label: str):
    if "knife" in label:
        return "knife"
    if any(x in label for x in ["person", "man", "woman", "human", "people", "chef"]):
        return "person"
    if "dog" in label:
        return "dog"
    return None


def keep_top_per_label(objects: List[Dict], max_per_label: int = 2):
    grouped = {}

    for obj in objects:
        grouped.setdefault(obj["label"], []).append(obj)

    kept = []
    for label, items in grouped.items():
        items = sorted(items, key=lambda x: x["confidence"], reverse=True)
        kept.extend(items[:max_per_label])

    return kept


def build_benchmark_prompts(prompt_specs):
    prompts = []

    for idx, (prompt, prompt_type, expected) in enumerate(prompt_specs, start=1):
        prompts.append({
            "prompt_id": f"p{idx:03d}",
            "prompt": prompt,
            "type": prompt_type,
            "expected": expected,
        })

    return prompts


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--sdxl_model_id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--grounding_dino_model_id",
        default="IDEA-Research/grounding-dino-base",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    print("[INFO] Loading SDXL...")
    generator = SDXLGenerator(
        model_id=args.sdxl_model_id,
        device=args.device,
    )

    print("[INFO] Loading Grounding DINO auto annotator...")
    annotator = GroundingDinoAutoAnnotator(
        model_id=args.grounding_dino_model_id,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    annotations = []

    for i in tqdm(range(args.num_samples), desc="Generating benchmark dataset"):
        scenario = random.choice(SCENARIOS)

        image_id = f"sample_{i + 1:06d}"
        image_name = f"{image_id}.jpg"
        image_path = image_dir / image_name

        image = generator.generate(
            prompt=scenario["image_prompt"],
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + i,
        )

        image.save(image_path)

        detected_objects = annotator.detect(
            image=image,
            query="person . knife . dog ."
        )

        sample = {
            "image_id": image_id,
            "image_path": str(image_path.resolve().relative_to(PROJECT_ROOT)),
            "category": scenario["category"],
            "generation_prompt": scenario["image_prompt"],
            "prompts": build_benchmark_prompts(scenario["prompts"]),
            "annotations": [
                {
                    "label": obj["label"],
                    "bbox": obj["bbox"],
                }
                for obj in detected_objects
            ],
            "pseudo_annotation_meta": [
                {
                    "label": obj["label"],
                    "bbox": obj["bbox"],
                    "confidence": obj["confidence"],
                }
                for obj in detected_objects
            ],
        }

        annotations.append(sample)

    annotation_path = output_dir / "annotations.json"

    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved annotations: {annotation_path}")
    print(f"[DONE] Saved images: {image_dir}")


if __name__ == "__main__":
    main()
