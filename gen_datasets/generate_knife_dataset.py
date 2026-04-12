import os
import json
import random
import shutil
import logging
import argparse
from time import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# =========================
# Logging
# =========================

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    log_path = os.path.join(
        output_dir,
        "logs",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    logger = logging.getLogger("knife_dataset")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(f"Log file: {log_path}")
    return logger


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
    split: str


@dataclass
class DetectionObject:
    label: str
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    confidence: float


@dataclass
class AnnotationRecord:
    image: str
    split: str
    width: int
    height: int
    caption: str
    category: str
    objects: List[Dict]


def choose_category() -> str:
    r = random.random()
    if r < 0.4:
        return "dangerous"
    if r < 0.8:
        return "neutral"
    return "hidden"


def choose_split(val_ratio: float) -> str:
    return "val" if random.random() < val_ratio else "train"


def sample_action(category: str) -> str:
    if category == "dangerous":
        return random.choice(DANGEROUS_ACTIONS)
    if category == "neutral":
        return random.choice(NEUTRAL_ACTIONS)
    return random.choice(HARD_CASE_ACTIONS)


def build_caption(person_type: str, action_phrase: str, environment: str) -> str:
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


def generate_sample_spec(val_ratio: float) -> SampleSpec:
    category = choose_category()
    action_phrase = sample_action(category)
    environment = random.choice(ENVIRONMENTS)
    person_type = random.choice(PERSON_TYPES)
    outfit = random.choice(OUTFITS)
    face_visibility = random.choice(FACE_VISIBILITY)
    camera_view = random.choice(CAMERA_VIEWS)
    lighting = random.choice(LIGHTING)
    occlusion = random.choice(OCCLUSIONS)
    split = choose_split(val_ratio)

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
        prompt="",
        split=split,
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

class MultiClassDetector:
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
    def detect(self, image: Image.Image, query: str) -> List[DetectionObject]:
        inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
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
                    label=str(label).strip().lower(),
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
        "images_train": os.path.join(base_dir, "images", "train"),
        "images_val": os.path.join(base_dir, "images", "val"),
        "labels_train": os.path.join(base_dir, "labels", "train"),
        "labels_val": os.path.join(base_dir, "labels", "val"),
        "annotations_train": os.path.join(base_dir, "annotations", "train"),
        "annotations_val": os.path.join(base_dir, "annotations", "val"),
        "meta_train": os.path.join(base_dir, "meta", "train"),
        "meta_val": os.path.join(base_dir, "meta", "val"),
        "visualizations_train": os.path.join(base_dir, "visualizations", "train"),
        "visualizations_val": os.path.join(base_dir, "visualizations", "val"),
        "failed": os.path.join(base_dir, "failed"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def xyxy_to_yolo(bbox_xyxy: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
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
    split: str,
    caption: str,
    category: str,
    detections: List[DetectionObject],
    img_w: int,
    img_h: int,
) -> None:
    record = AnnotationRecord(
        image=image_name,
        split=split,
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


def save_visualization(
    vis_path: str,
    image: Image.Image,
    detections: List[DetectionObject],
    color_map: Dict[str, Tuple[int, int, int]],
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for det in detections:
        color = color_map.get(det.label, (255, 255, 0))
        x1, y1, x2, y2 = det.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        text = f"{det.label} {det.confidence:.2f}"
        tx = x1 + 4
        ty = max(0, y1 - 18)
        draw.rectangle([tx, ty, tx + 180, ty + 18], fill=color)
        draw.text((tx + 3, ty + 1), text, fill=(0, 0, 0))

    canvas.save(vis_path)


# =========================
# Filtering / Validation
# =========================

def normalize_label(label: str) -> Optional[str]:
    label = label.strip().lower()

    if "knife" in label:
        return "knife"

    person_aliases = [
        "person",
        "man",
        "woman",
        "teenager",
        "human",
        "people",
    ]
    if any(alias in label for alias in person_aliases):
        return "person"

    return None


def filter_detections(
    detections: List[DetectionObject],
    min_conf: float,
    max_boxes_per_class: int,
) -> List[DetectionObject]:
    grouped: Dict[str, List[DetectionObject]] = {"person": [], "knife": []}

    for det in detections:
        norm_label = normalize_label(det.label)
        if norm_label is None:
            continue
        if det.confidence < min_conf:
            continue

        det.label = norm_label
        grouped[norm_label].append(det)

    for key in grouped:
        grouped[key].sort(key=lambda x: x.confidence, reverse=True)
        grouped[key] = grouped[key][:max_boxes_per_class]

    return grouped["person"] + grouped["knife"]


def reject_sample(
    detections: List[DetectionObject],
    require_person: bool = True,
    require_knife: bool = True,
) -> bool:
    labels = [d.label for d in detections]
    if require_person and "person" not in labels:
        return True
    if require_knife and "knife" not in labels:
        return True
    return False


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
    parser.add_argument("--max_boxes_per_class", type=int, default=3)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_failed", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--max_visualizations_per_split", type=int, default=50)

    return parser.parse_args()


def write_dataset_yaml(output_dir: str) -> None:
    yaml_text = [
        f"path: {os.path.abspath(output_dir)}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: person",
        "  1: knife",
    ]
    with open(os.path.join(output_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_text))


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.output_dir)

    logger.info("===== START DATASET GENERATION =====")
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Validation ratio: {args.val_ratio}")
    logger.info(f"Device: {args.device}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    paths = ensure_dirs(args.output_dir)
    class_map = {"person": 0, "knife": 1}
    color_map = {"person": (0, 255, 0), "knife": (255, 0, 0)}

    logger.info("Loading SDXL pipeline...")
    generator = SDXLGenerator(
        model_id=args.sdxl_model_id,
        device=args.device,
        dtype=torch.float16 if "cuda" in args.device else torch.float32,
    )

    logger.info("Loading Grounding DINO...")
    detector = MultiClassDetector(
        model_id=args.grounding_dino_model_id,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    kept = 0
    failed = 0
    trial_idx = 0
    vis_count = {"train": 0, "val": 0}

    pbar = tqdm(total=args.num_samples, desc="Generating dataset")

    while kept < args.num_samples:
        trial_idx += 1
        sample_idx = kept
        start_t = time()

        spec = generate_sample_spec(val_ratio=args.val_ratio)
        logger.info(f"[Trial {trial_idx}] split={spec.split} prompt={spec.prompt}")

        try:
            image = generator.generate(
                prompt=spec.prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + trial_idx,
            )
        except Exception as e:
            failed += 1
            logger.error(f"Image generation failed: {e}")
            continue

        try:
            raw_detections = detector.detect(image, query="person . knife .")
        except Exception as e:
            failed += 1
            logger.error(f"Detection failed: {e}")
            continue

        logger.info(f"Raw detections: {len(raw_detections)}")
        detections = filter_detections(
            raw_detections,
            min_conf=args.min_conf,
            max_boxes_per_class=args.max_boxes_per_class,
        )

        if reject_sample(detections, require_person=True, require_knife=True):
            failed += 1
            logger.warning("Rejected sample: missing person or knife")
            if args.save_failed:
                failed_name = f"failed_{failed:06d}.jpg"
                image.save(os.path.join(paths["failed"], failed_name))
            continue

        image_name = f"image_{sample_idx:06d}.jpg"
        label_name = f"image_{sample_idx:06d}.txt"
        json_name = f"image_{sample_idx:06d}.json"
        meta_name = f"image_{sample_idx:06d}.meta.json"

        if spec.split == "train":
            image_path = os.path.join(paths["images_train"], image_name)
            label_path = os.path.join(paths["labels_train"], label_name)
            json_path = os.path.join(paths["annotations_train"], json_name)
            meta_path = os.path.join(paths["meta_train"], meta_name)
            vis_path = os.path.join(paths["visualizations_train"], image_name)
        else:
            image_path = os.path.join(paths["images_val"], image_name)
            label_path = os.path.join(paths["labels_val"], label_name)
            json_path = os.path.join(paths["annotations_val"], json_name)
            meta_path = os.path.join(paths["meta_val"], meta_name)
            vis_path = os.path.join(paths["visualizations_val"], image_name)

        image.save(image_path)
        save_yolo_label(label_path, detections, image.width, image.height, class_map)
        save_json_annotation(
            json_path=json_path,
            image_name=image_name,
            split=spec.split,
            caption=spec.caption,
            category=spec.category,
            detections=detections,
            img_w=image.width,
            img_h=image.height,
        )
        save_metadata(meta_path, spec)

        if args.save_visualizations and vis_count[spec.split] < args.max_visualizations_per_split:
            save_visualization(vis_path, image, detections, color_map)
            vis_count[spec.split] += 1

        for det in detections:
            logger.info(f"Detected {det.label}: bbox={det.bbox_xyxy}, conf={det.confidence:.3f}")

        kept += 1
        elapsed = time() - start_t
        logger.info(f"Saved sample {image_name} | split={spec.split} | elapsed={elapsed:.2f}s")
        pbar.update(1)

    pbar.close()
    write_dataset_yaml(args.output_dir)

    logger.info(f"Done. saved={kept}, failed={failed}, trials={trial_idx}")
    logger.info(f"Success rate: {kept / max(1, trial_idx):.3f}")
    logger.info("===== END =====")


if __name__ == "__main__":
    main()