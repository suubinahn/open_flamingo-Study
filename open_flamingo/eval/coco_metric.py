from pycocoevalcap.cider.cider import Cider
from pycocotools.coco import COCO


def compute_cider(
    result_path,
    annotations_path,
):
    # Compute CIDEr only. COCOEvalCap evaluates every caption metric, including
    # SPICE, which triggers a Stanford CoreNLP download that is unnecessary here.
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)
    image_ids = coco_result.getImgIds()
    gts = {
        image_id: [annotation["caption"] for annotation in coco.imgToAnns[image_id]]
        for image_id in image_ids
    }
    results = {
        image_id: [annotation["caption"] for annotation in coco_result.imgToAnns[image_id]]
        for image_id in image_ids
    }

    cider_score, _ = Cider().compute_score(gts, results)
    return {"CIDEr": cider_score}


def postprocess_captioning_generation(predictions):
    return predictions.split("Output", 1)[0]
