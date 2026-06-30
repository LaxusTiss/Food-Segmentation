from __future__ import annotations

from collections import Counter

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


MODEL_CKPT = "facebook/mask2former-swin-tiny-ade-semantic"
WEIGHT_PATH = "mask2former_best_model.pth"
NUM_CLASSES = 104
IMAGE_SIZE = 256
IGNORE_INDEX = 255

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FOODSEG_LABELS = [
    "background",
    "candy",
    "egg tart",
    "french fries",
    "chocolate",
    "biscuit",
    "popcorn",
    "pudding",
    "ice cream",
    "cheese butter",
    "cake",
    "wine",
    "milkshake",
    "coffee",
    "juice",
    "milk",
    "tea",
    "almond",
    "red beans",
    "cashew",
    "dried cranberries",
    "soy",
    "walnut",
    "peanut",
    "egg",
    "apple",
    "date",
    "apricot",
    "avocado",
    "banana",
    "strawberry",
    "cherry",
    "blueberry",
    "raspberry",
    "mango",
    "olives",
    "peach",
    "lemon",
    "pear",
    "fig",
    "pineapple",
    "grape",
    "kiwi",
    "melon",
    "orange",
    "watermelon",
    "steak",
    "pork",
    "chicken duck",
    "sausage",
    "fried meat",
    "lamb",
    "sauce",
    "crab",
    "fish",
    "shellfish",
    "shrimp",
    "soup",
    "bread",
    "corn",
    "hamburg",
    "pizza",
    "hanamaki baozi",
    "wonton dumplings",
    "pasta",
    "noodles",
    "rice",
    "pie",
    "tofu",
    "eggplant",
    "potato",
    "garlic",
    "cauliflower",
    "tomato",
    "kelp",
    "seaweed",
    "spring onion",
    "rape",
    "ginger",
    "okra",
    "lettuce",
    "pumpkin",
    "cucumber",
    "white radish",
    "carrot",
    "asparagus",
    "bamboo shoots",
    "broccoli",
    "celery stick",
    "cilantro mint",
    "snow peas",
    "cabbage",
    "bean sprouts",
    "onion",
    "pepper",
    "green beans",
    "French beans",
    "king oyster mushroom",
    "shiitake",
    "enoki mushroom",
    "oyster mushroom",
    "white button mushroom",
    "salad",
    "other ingredients",
]

ID2LABEL = {idx: label for idx, label in enumerate(FOODSEG_LABELS)}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def build_palette(num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


processor = Mask2FormerImageProcessor(
    ignore_index=IGNORE_INDEX,
    reduce_labels=False,
    do_resize=True,
    size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
)

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    MODEL_CKPT,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# strict=False: checkpoint fine-tuned chỉ thay đổi vài layer đầu ra (class_predictor...),
# phần backbone (Swin) có thể có một vài key đặt tên khác phiên bản transformers -> không nên
# bắt buộc khớp 100% nếu không sẽ lỗi như log gặp phải (Missing/Unexpected keys).
state_dict = load_state_dict(WEIGHT_PATH)
load_result = model.load_state_dict(state_dict, strict=False)

if load_result.missing_keys:
    print(f"[warning] {len(load_result.missing_keys)} missing keys khi load checkpoint, ví dụ: {load_result.missing_keys[:5]}")
if load_result.unexpected_keys:
    print(f"[warning] {len(load_result.unexpected_keys)} unexpected keys khi load checkpoint, ví dụ: {load_result.unexpected_keys[:5]}")

# Cảnh báo riêng nếu các layer đầu ra (quan trọng cho kết quả dự đoán) bị thiếu hoàn toàn
critical_substrings = ("class_predictor", "query_feat", "query_embed", "criterion")
critical_missing = [k for k in load_result.missing_keys if any(s in k for s in critical_substrings)]
if critical_missing:
    print(f"[ERROR] Các layer quan trọng cho kết quả dự đoán bị thiếu trong checkpoint: {critical_missing}")
    print("Model có thể chạy được nhưng kết quả phân đoạn sẽ KHÔNG chính xác vì chưa load được trọng số đã fine-tune.")

model.to(DEVICE)
model.eval()

PALETTE = build_palette(NUM_CLASSES)


def summarize_mask(mask: np.ndarray, top_k: int = 8) -> list[dict[str, float | str]]:
    pixels = mask.reshape(-1)
    pixels = pixels[(pixels > 0) & (pixels < NUM_CLASSES)]
    total = int(pixels.size)
    if total == 0:
        return []

    rows = []
    for class_id, count in Counter(pixels.tolist()).most_common(top_k):
        rows.append(
            {
                "label": ID2LABEL.get(class_id, f"Class_{class_id}"),
                "percent": round(count * 100 / total, 2),
            }
        )
    return rows


def predict(image: Image.Image):
    if image is None:
        return None, None, []

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)

    target_size = image.size[::-1]
    pred = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[target_size],
    )[0]

    mask = pred.cpu().numpy()
    safe_mask = np.clip(mask, 0, NUM_CLASSES - 1)
    color_mask = PALETTE[safe_mask]
    image_np = np.array(image)
    overlay = (0.55 * image_np + 0.45 * color_mask).astype(np.uint8)

    return Image.fromarray(overlay), Image.fromarray(color_mask), summarize_mask(mask)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Food image"),
    outputs=[
        gr.Image(type="pil", label="Overlay"),
        gr.Image(type="pil", label="Mask"),
        gr.JSON(label="Top labels"),
    ],
    title="Food Segmentation - Mask2Former",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()