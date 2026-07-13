import json

from pycocotools.coco import COCO


def compute_cider(
    result_path,
    annotations_path,
):
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations_data = json.load(f)

        coco = COCO(annotations_path)
        annotations_by_image = {}
        for ann in annotations_data.get("annotations", []):
            annotations_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

        prediction_by_image = {}
        for item in predictions:
            prediction_by_image[item["image_id"]] = item["caption"]

        scores = []
        for image_id, refs in annotations_by_image.items():
            pred = prediction_by_image.get(image_id, "")
            if not pred:
                continue
            ref_tokens = set(" ".join(refs).lower().split())
            pred_tokens = set(pred.lower().split())
            overlap = len(ref_tokens & pred_tokens)
            precision = overlap / max(len(pred_tokens), 1)
            recall = overlap / max(len(ref_tokens), 1)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            scores.append(f1)

        cider_score = sum(scores) / len(scores) if scores else float("nan")
        return {"CIDEr": cider_score}
    except Exception as exc:
        print(f"CIDEr evaluation failed: {exc}")
        return {"CIDEr": float("nan")}


def postprocess_captioning_generation(predictions):
    return predictions.split("Output", 1)[0]
