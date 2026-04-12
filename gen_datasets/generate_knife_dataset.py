import os
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# =========================
# Prompt / Dataset Design
# =========================

DANGEROUS_ACTIONS = [
    "holding",
    "swinging",
    "pointing",
    "attacking with",
    "threatening with",
    "chasing someone with",
]

NEUTRAL_ACTIONS = [
    "cutting food with",
    "cooking with",
    "preparing ingredients with",
]

HARD_CASE_ACTIONS = [
    "hiding",
    "concealing",
    "partially showing",
]

ENVIRONMENTS = [
    "street",
    "kitchen",
    "restaurant",
    "dark alley",
    "indoor room",
    "subway station",
    "parking lot",
    "office",
]

PERSON_TYPES = [
    "man",
    "woman",
    "teenager",
    "person",
]

OUTFITS = [
    "wearing a hoodie",
    "wearing a suit",
    "wearing casual clothes",
]

FACE_VISIBILITY = [
    "face visible",
    "face partially occluded",
]

CAMERA_VIEWS = [
    "close-up shot",
    "medium shot",
    "long shot",
    "side view",
    "top-down view",
]

LIGHTING = [
    "daylight",
    "night",
    "low light",
    "backlight",
]

OCCLUSIONS = [
    "knife clearly visible",
    "knife partially hidden",
    "hand covering part of the knife",
]

NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, unrealistic, extra limbs, distorted hands, "
    "deformed body, bad anatomy, duplicated objects, watermark, text"
)


@dataclass
class SampleSpec:
    category: str
    action_phrase: str
    environment: str
    person_type: str
    outfit: str
    face_visibility: str
    camera_view: str
    lighting: str
    occlusion: str
    caption: str
    prompt: str


@dataclass
class DetectionObject:
    label: str
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    confidence: float


@dataclass
class AnnotationRecord:
    image: str
    width: int
    height: int
    caption: str
    category: str
    objects: List[Dict]


def choose_category() -> str:
    """Balanced distribution:
    dangerous 40%, neutral 40%, hidden 20%
    """
    r = random.random()
    if r < 0.4:
        return "dangerous"
    if r < 0.8:
        return "neutral"
    return "hidden"


def sample_action(category: str) -> str:
    if category == "dangerous":
        return random.choice(DANGEROUS_ACTIONS)
    if category == "neutral":
        return random.choice(NEUTRAL_ACTIONS)
    return random.choice(HARD_CASE_ACTIONS)


def build_caption(
    person_type: str,
    action_phrase: str,
    environment: str,
) -> str:
    if action_phrase == "hiding":
        return f"A {person_type} is hiding a knife behind their back in a {environment}"
    if action_phrase == "concealing":
        return f"A {person_type} is concealing a knife in a pocket in a {environment}"
    if action_phrase == "partially showing":
        return f"A {person_type} is showing a partially visible knife in a {environment}"

    return f"A {person_type} is {action_phrase} a knife in a {environment}"


def build_prompt(spec: SampleSpec) -> str:
    return (
        f"A realistic photo of a {spec.person_type}, {spec.outfit}, {spec.face_visibility}, "
        f"{spec.action_phrase} a knife in a {spec.environment}, "
        f"{spec.camera_view}, {spec.lighting}, {spec.occlusion}, "
        f"full body, natural scene, high detail, realistic surveillance-style photo"
    )


def generate_sample_spec() -> SampleSpec:
    category = choose_category()
    action_phrase = sample_action(category)
    environment = random.choice(ENVIRONMENTS)
    person_type = random.choice(PERSON_TYPES)
    outfit = random.choice(OUTFITS)
    face_visibility = random.choice(FACE_VISIBILITY)
    camera_view = random.choice(CAMERA_VIEWS)
    lighting = random.choice(LIGHTING)
    occlusion = random.choice(OCCLUSIONS)

    caption = build_caption(person_type, action_phrase, environment)

    spec = SampleSpec(
        category=category,
        action_phrase=action_phrase,
        environment=environment,
        person_type=person_type,
        outfit=outfit,
        face_visibility=face_visibility,
        camera_view=camera_view,
        lighting=lighting,
        occlusion=occlusion,
        caption=caption,
        prompt="",  # filled below
    )
    spec.prompt = build_prompt(spec)
    return spec


# =========================
# Image Generation
# =========================

class SDXLGenerator:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
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
        negative_prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]


# =========================
# Grounding DINO Detection
# =========================

class KnifeDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    @torch.inference_mode()
    def detect_knife(self, image: Image.Image) -> List[DetectionObject]:
        text_prompt = "knife."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]

        detections: List[DetectionObject] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            detections.append(
                DetectionObject(
                    label=str(label),
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_xywh=[x1, y1, w, h],
                    confidence=float(score.item()),
                )
            )

        return detections


# =========================
# Saving Utilities
# =========================

def ensure_dirs(base_dir: str) -> Dict[str, str]:
    paths = {
        "images": os.path.join(base_dir, "images"),
        "labels": os.path.join(base_dir, "labels"),
        "annotations": os.path.join(base_dir, "annotations"),
        "meta": os.path.join(base_dir, "meta"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def xyxy_to_yolo(
    bbox_xyxy: List[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def save_yolo_label(
    label_path: str,
    detections: List[DetectionObject],
    img_w: int,
    img_h: int,
    class_map: Dict[str, int],
) -> None:
    lines = []
    for det in detections:
        if det.label not in class_map:
            continue
        class_id = class_map[det.label]
        cx, cy, w, h = xyxy_to_yolo(det.bbox_xyxy, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_annotation(
    json_path: str,
    image_name: str,
    caption: str,
    category: str,
    detections: List[DetectionObject],
    img_w: int,
    img_h: int,
) -> None:
    record = AnnotationRecord(
        image=image_name,
        width=img_w,
        height=img_h,
        caption=caption,
        category=category,
        objects=[
            {
                "label": det.label,
                "bbox_xyxy": det.bbox_xyxy,
                "bbox_xywh": det.bbox_xywh,
                "confidence": det.confidence,
            }
            for det in detections
        ],
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, ensure_ascii=False)


def save_metadata(meta_path: str, spec: SampleSpec) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(spec), f, indent=2, ensure_ascii=False)


# =========================
# Filtering / Validation
# =========================

def filter_best_knife_detections(
    detections: List[DetectionObject],
    min_conf: float = 0.25,
    max_boxes: int = 3,
) -> List[DetectionObject]:
    valid = [d for d in detections if d.confidence >= min_conf and "knife" in d.label.lower()]
    valid.sort(key=lambda x: x.confidence, reverse=True)
    return valid[:max_boxes]


def reject_sample(
    detections: List[DetectionObject],
    min_boxes: int = 1,
) -> bool:
    return len(detections) < min_boxes


# =========================
# Main Pipeline
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--sdxl_model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--grounding_dino_model_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--min_conf", type=float, default=0.25)
    parser.add_argument("--max_boxes", type=int, default=3)

    parser.add_argument(
        "--class_name",
        type=str,
        default="knife",
        help="Single-class detection target",
    )
    parser.add_argument(
        "--save_failed",
        action="store_true",
        help="Save images even if no knife bbox is detected",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    paths = ensure_dirs(args.output_dir)
    class_map = {args.class_name: 0}

    print("Loading SDXL pipeline...")
    generator = SDXLGenerator(
        model_id=args.sdxl_model_id,
        device=args.device,
        dtype=torch.float16 if "cuda" in args.device else torch.float32,
    )

    print("Loading Grounding DINO...")
    detector = KnifeDetector(
        model_id=args.grounding_dino_model_id,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    kept = 0
    failed = 0
    target = args.num_samples
    pbar = tqdm(total=target, desc="Generating dataset")

    trial_idx = 0
    while kept < target:
        trial_idx += 1
        sample_idx = kept

        spec = generate_sample_spec()

        image = generator.generate(
            prompt=spec.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + trial_idx,
        )

        detections = detector.detect_knife(image)
        detections = filter_best_knife_detections(
            detections,
            min_conf=args.min_conf,
            max_boxes=args.max_boxes,
        )

        image_name = f"image_{sample_idx:06d}.jpg"
        label_name = f"image_{sample_idx:06d}.txt"
        json_name = f"image_{sample_idx:06d}.json"
        meta_name = f"image_{sample_idx:06d}.meta.json"

        if reject_sample(detections):
            failed += 1
            if args.save_failed:
                failed_name = f"failed_{failed:06d}.jpg"
                image.save(os.path.join(paths["images"], failed_name))
            continue

        image_path = os.path.join(paths["images"], image_name)
        label_path = os.path.join(paths["labels"], label_name)
        json_path = os.path.join(paths["annotations"], json_name)
        meta_path = os.path.join(paths["meta"], meta_name)

        image.save(image_path)

        save_yolo_label(
            label_path=label_path,
            detections=detections,
            img_w=image.width,
            img_h=image.height,
            class_map=class_map,
        )

        save_json_annotation(
            json_path=json_path,
            image_name=image_name,
            caption=spec.caption,
            category=spec.category,
            detections=detections,
            img_w=image.width,
            img_h=image.height,
        )

        save_metadata(meta_path, spec)

        kept += 1
        pbar.update(1)

    pbar.close()

    dataset_yaml = {
        "path": os.path.abspath(args.output_dir),
        "train": "images",
        "val": "images",
        "names": {0: args.class_name},
    }
    with open(os.path.join(args.output_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml_text = [
            f"path: {dataset_yaml['path']}",
            f"train: {dataset_yaml['train']}",
            f"val: {dataset_yaml['val']}",
            "names:",
            f"  0: {args.class_name}",
        ]
        f.write("\n".join(yaml_text))

    print(f"Done. saved={kept}, failed={failed}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

