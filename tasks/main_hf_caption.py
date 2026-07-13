# -*- coding: utf-8 -*-

import ssl
import certifi


def create_certifi_ssl_context(*args, **kwargs):
    """Windows 인증서 저장소 대신 certifi CA 번들을 사용합니다."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=certifi.where())
    return context


ssl.create_default_context = create_certifi_ssl_context

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageDraw

from inference import generate_text
from open_flamingo.eval.coco_metric import compute_cider
from utils.image_utils import prepare_images
from utils.model_loader import load_model


# ============================================================
# Experiment configuration
# ============================================================

DATASET_NAME = "lmms-lab/COCO-Caption"
DATASET_SPLIT = "val"
NUM_DATASET_SAMPLES = 200

NUM_SHOTS = 2
SEED = 42
EMBEDDING_BATCH_SIZE = 16
DEFAULT_SAMPLES_PER_MODE = 3

# 첫 실행 때 생성되며, 이후 실행에서는 재사용됩니다.
EMBEDDING_CACHE_PATH = Path(
    "cache/coco_caption_val_200_vitl14_embeddings.pt"
)
RESULTS_DIR = Path("results")
PREVIEW_SIZE = (320, 320)
LABEL_HEIGHT = 36


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example-mode",
        choices=["similarity", "fixed", "random", "all"],
        default="similarity",
    )
    parser.add_argument(
        "--samples-per-mode",
        type=int,
        default=DEFAULT_SAMPLES_PER_MODE,
    )
    return parser.parse_args()


def get_mode_result_dir(mode):
    return RESULTS_DIR / mode


def get_mode_eval_paths(mode):
    mode_dir = get_mode_result_dir(mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    return (
        mode_dir / "coco_caption_annotations.json",
        mode_dir / "coco_caption_predictions.json",
        mode_dir / "cider_summary.csv",
        mode_dir / "cider_summary.txt",
    )


def build_prompt(caption1, caption2):
    """
    두 개의 이미지-캡션 예시를 사용해 query 이미지의 캡션을 생성하는
    OpenFlamingo few-shot prompt를 만듭니다.
    """

    return (
        "Describe the image in one short English sentence. "
        "Focus on the main object and scene.\n"
        f"<image>Output: {caption1}<|endofchunk|>"
        f"<image>Output: {caption2}<|endofchunk|>"
        "<image>Output:"
    )


def build_image_embeddings(
    model,
    image_processor,
    dataset,
    device,
    batch_size=EMBEDDING_BATCH_SIZE,
):
    """
    데이터셋의 모든 이미지를 OpenFlamingo 내부 CLIP vision encoder로
    임베딩합니다.

    반환값의 각 행은 이미지 한 장에 대응하는 L2-normalized 벡터입니다.
    정규화된 벡터끼리의 내적은 cosine similarity와 같습니다.
    """

    embeddings = []
    total_images = len(dataset)

    model.vision_encoder.eval()

    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)

        print(
            f"Building embeddings: {end}/{total_images}",
            end="\r",
        )

        batch_samples = dataset[start:end]

        # COCO 이미지를 RGB로 통일한 뒤 CLIP 입력 텐서로 변환합니다.
        image_tensors = [
            image_processor(sample["image"].convert("RGB"))
            for sample in batch_samples
        ]

        image_tensor = torch.stack(image_tensors).to(device)

        with torch.inference_mode():
            # OpenFlamingo 내부 구현과 동일하게 token features를 사용합니다.
            # 반환값 [1]: (batch_size, num_visual_tokens, feature_dim)
            token_features = model.vision_encoder(image_tensor)[1]

            # 여러 visual token을 평균 내 이미지당 하나의 벡터로 만듭니다.
            image_features = token_features.mean(dim=1)

            # cosine similarity 검색을 위해 L2 normalize 합니다.
            image_features = F.normalize(
                image_features,
                dim=-1,
            )

        # GPU 메모리를 계속 점유하지 않도록 CPU에 저장합니다.
        embeddings.append(image_features.cpu())

    print()

    return torch.cat(embeddings, dim=0)


def load_or_build_embeddings(
    model,
    image_processor,
    dataset,
    device,
):
    """
    저장된 임베딩 캐시가 있고 데이터셋 크기가 같으면 재사용합니다.
    그렇지 않으면 새로 만들고 cache 폴더에 저장합니다.
    """

    if EMBEDDING_CACHE_PATH.exists():
        cached_data = torch.load(
            EMBEDDING_CACHE_PATH,
            map_location="cpu",
        )

        if cached_data["num_images"] == len(dataset):
            print(
                f"Loading cached embeddings: {EMBEDDING_CACHE_PATH}"
            )
            return cached_data["embeddings"]

    print("\nBuilding image embeddings...")
    embeddings = build_image_embeddings(
        model=model,
        image_processor=image_processor,
        dataset=dataset,
        device=device,
    )

    EMBEDDING_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "num_images": len(dataset),
            "embeddings": embeddings,
        },
        EMBEDDING_CACHE_PATH,
    )

    print(f"Embeddings saved: {EMBEDDING_CACHE_PATH}")

    return embeddings


def find_similar_examples_by_image(
    embeddings,
    query_index,
    num_examples=NUM_SHOTS,
):
    """
    query 이미지와 cosine similarity가 높은 이미지를 예시로 선택합니다.

    query의 정답 캡션을 사용하지 않으므로, 기존 문자열 기반 검색에서
    발생했던 정답 정보 누수 없이 유사한 이미지를 찾습니다.
    """

    query_embedding = embeddings[query_index]

    # embeddings는 이미 L2 normalize되어 있으므로 내적 = cosine similarity
    similarity_scores = embeddings @ query_embedding

    # query 자기 자신은 예시 후보에서 제외합니다.
    similarity_scores[query_index] = float("-inf")

    top_results = torch.topk(
        similarity_scores,
        k=num_examples,
    )

    return (
        top_results.indices.tolist(),
        top_results.values.tolist(),
    )


def save_image_comparison(
    example1,
    example1_index,
    example2,
    example2_index,
    query,
    query_index,
    mode="similarity",
):
    """Save the retrieved examples and query image in one contact sheet."""

    mode_dir = get_mode_result_dir(mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        (example1["image"], f"Example 1 (index {example1_index})"),
        (example2["image"], f"Example 2 (index {example2_index})"),
        (query["image"], f"Query (index {query_index})"),
    ]

    panel_width, panel_height = PREVIEW_SIZE
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + LABEL_HEIGHT),
        "white",
    )
    drawer = ImageDraw.Draw(canvas)

    for panel_index, (image, label) in enumerate(panels):
        preview = image.convert("RGB").copy()
        preview.thumbnail(PREVIEW_SIZE, Image.LANCZOS)

        x_offset = panel_index * panel_width
        x_image = x_offset + (panel_width - preview.width) // 2
        y_image = (panel_height - preview.height) // 2
        canvas.paste(preview, (x_image, y_image))
        drawer.text((x_offset + 8, panel_height + 10), label, fill="black")

    output_path = mode_dir / f"query_{query_index:05d}_comparison.png"
    canvas.save(output_path)
    return output_path


def save_coco_caption_evaluation_files(dataset, predictions, mode="similarity"):
    """COCO 형식의 annotation/prediction JSON 파일을 저장합니다."""
    mode_dir = get_mode_result_dir(mode)
    mode_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = mode_dir / "coco_caption_annotations.json"
    predictions_path = mode_dir / "coco_caption_predictions.json"

    annotations = []
    for image_id, sample in enumerate(dataset):
        for caption in sample["answer"]:
            annotations.append(
                {
                    "image_id": image_id,
                    "id": len(annotations) + 1,
                    "caption": caption,
                    "iscrowd": 0,
                }
            )

    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "images": [
                    {"id": image_id, "file_name": f"{image_id:05d}.jpg"}
                    for image_id in range(len(dataset))
                ],
                "annotations": annotations,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    prediction_records = []
    for item in predictions:
        prediction_records.append(
            {
                "image_id": item["image_id"],
                "id": item["id"],
                "caption": item["caption"],
            }
        )

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2)

    return annotations_path, predictions_path


def save_evaluation_summary(dataset, predictions, cider_score, step=None, mode="similarity"):
    """중간/최종 평가 결과를 CSV와 텍스트로 저장합니다."""
    annotations_path, predictions_path, summary_csv_path, summary_txt_path = get_mode_eval_paths(mode)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    if step is None:
        step = len(predictions)

    row = {
        "step": step,
        "num_samples": len(predictions),
        "cider": round(cider_score, 4),
        "annotations_path": str(annotations_path),
        "predictions_path": str(predictions_path),
    }

    file_exists = summary_csv_path.exists()
    with summary_csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "num_samples", "cider", "annotations_path", "predictions_path"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("CIDEr evaluation summary\n")
        f.write(f"Total evaluated samples: {len(predictions)}\n")
        f.write(f"Current CIDEr: {cider_score:.2f}\n")
        f.write(f"Annotations: {annotations_path}\n")
        f.write(f"Predictions: {predictions_path}\n")
        f.write(f"CSV summary: {summary_csv_path}\n")


def evaluate_current_predictions(dataset, predictions, mode="similarity"):
    """현재까지 누적된 예측들에 대해 CIDEr를 계산하고 결과를 저장합니다."""
    if not predictions:
        return None

    annotations_path, predictions_path = save_coco_caption_evaluation_files(
        dataset=dataset,
        predictions=predictions,
        mode=mode,
    )
    cider_scores = compute_cider(
        str(predictions_path),
        str(annotations_path),
    )
    cider_score = cider_scores.get("CIDEr", float("nan")) * 100.0
    save_evaluation_summary(
        dataset=dataset,
        predictions=predictions,
        cider_score=cider_score,
        step=len(predictions),
        mode=mode,
    )

    return cider_score


def select_examples_by_mode(
    embeddings,
    query_index,
    dataset,
    num_examples=NUM_SHOTS,
    mode="similarity",
    rng=None,
):
    if mode == "similarity":
        example_indices, similarity_scores = find_similar_examples_by_image(
            embeddings=embeddings,
            query_index=query_index,
            num_examples=num_examples,
        )
        return example_indices, similarity_scores

    if mode == "fixed":
        candidate_indices = [
            idx for idx in range(min(10, len(dataset))) if idx != query_index
        ]
        if len(candidate_indices) < num_examples:
            candidate_indices = [
                idx for idx in range(len(dataset)) if idx != query_index
            ]
        example_indices = candidate_indices[:num_examples]
        return example_indices, [None] * len(example_indices)

    if mode == "random":
        candidate_indices = [
            idx for idx in range(len(dataset)) if idx != query_index
        ]
        if len(candidate_indices) < num_examples:
            example_indices = candidate_indices[:]
        else:
            example_indices = rng.sample(candidate_indices, num_examples)
        return example_indices, [None] * len(example_indices)

    raise ValueError(f"Unsupported example mode: {mode}")


def main():

    args = parse_args()

    print("=" * 60)
    print("OpenFlamingo + HuggingFace COCO Caption")
    print("=" * 60)

    # 매 실행마다 다른 query 순서를 사용합니다.
    rng = random.Random()

    # --------------------------------------------------------
    # 모델 로드
    # --------------------------------------------------------
    print("\nLoading model...")

    model, image_processor, tokenizer, device = load_model()

    print("Model loaded!")

    # --------------------------------------------------------
    # 데이터셋 로드
    # --------------------------------------------------------
    print("\nLoading dataset...")

    dataset_stream = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )


    # 전체 COCO split을 내려받지 않고 앞의 일부 샘플만 사용합니다.
    dataset = list(dataset_stream.take(NUM_DATASET_SAMPLES))

    print(f"Dataset loaded: {len(dataset)} streamed samples")

    # --------------------------------------------------------
    # 이미지 임베딩 준비
    # --------------------------------------------------------
    image_embeddings = load_or_build_embeddings(
        model=model,
        image_processor=image_processor,
        dataset=dataset,
        device=device,
    )

    print("Image embeddings ready!")

    modes = [args.example_mode] if args.example_mode != "all" else ["similarity", "fixed", "random"]

    for example_mode in modes:
        predictions = []
        mode_dir = get_mode_result_dir(example_mode)
        print("\n" + "=" * 60)
        print(f"Running example mode: {example_mode}")
        print(f"Results directory: {mode_dir}")
        print("=" * 60)

        # --------------------------------------------------------
        # 반복 추론
        # --------------------------------------------------------
        while True:
            if len(predictions) >= args.samples_per_mode:
                print(
                    f"Reached {args.samples_per_mode} samples for {example_mode}; moving on."
                )
                break

            print("\n" + "=" * 60)
            print("Enter : Next query")
            print("q     : Quit")
            print("=" * 60)

            command = input(">> ").strip().lower()

            if command == "q":
                print("Program terminated.")
                break

            # 재현 가능한 방식으로 query 하나를 선택합니다.
            query_index = rng.randrange(len(dataset))
            query = dataset[query_index]

            # query의 정답은 검색에 쓰지 않고, 결과 비교에만 사용합니다.
            ground_truth_captions = query["answer"]

            print("\n========== Query ==========")
            print(f"Index: {query_index}")
            print("Ground truth captions:")

            for index, caption in enumerate(
                ground_truth_captions,
                start=1,
            ):
                print(f"{index}. {caption}")

            # ----------------------------------------------------
            # query 이미지와 유사한 support 이미지 2장 선택
            # ----------------------------------------------------
            example_indices, similarity_scores = select_examples_by_mode(
                embeddings=image_embeddings,
                query_index=query_index,
                dataset=dataset,
                num_examples=NUM_SHOTS,
                mode=example_mode,
                rng=rng,
            )

            example1 = dataset[example_indices[0]]
            example2 = dataset[example_indices[1]]

            print("\n========== Retrieved Examples ==========")

            if example_mode == "similarity":
                print(
                    f"Example 1 index: {example_indices[0]} "
                    f"(similarity: {similarity_scores[0]:.4f})"
                )
                print(
                    f"Example 2 index: {example_indices[1]} "
                    f"(similarity: {similarity_scores[1]:.4f})"
                )
            else:
                print(f"Example 1 index: {example_indices[0]}")
                print(f"Example 2 index: {example_indices[1]}")

            print(example1["answer"][0])
            print(example2["answer"][0])

            comparison_path = save_image_comparison(
                example1=example1,
                example1_index=example_indices[0],
                example2=example2,
                example2_index=example_indices[1],
                query=query,
                query_index=query_index,
                mode=example_mode,
            )

            print(f"\nImage comparison saved: {comparison_path}")

            # ----------------------------------------------------
            # few-shot prompt 생성
            # ----------------------------------------------------
            prompt = build_prompt(
                example1["answer"][0],
                example2["answer"][0],
            )

            print("\n========== Prompt ==========")
            print(prompt)

            # OpenFlamingo 입력 순서와 prompt의 <image> 순서를 맞춥니다.
            images = [
                example1["image"],
                example2["image"],
                query["image"],
            ]

            vision_x = prepare_images(
                image_processor,
                images,
            )

            # ----------------------------------------------------
            # 캡션 생성
            # ----------------------------------------------------
            prediction = generate_text(
                model=model,
                tokenizer=tokenizer,
                vision_x=vision_x,
                prompt=prompt,
                device=device,
                max_new_tokens=16,
                num_beams=4,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )

            print("\n" + "=" * 60)
            print("Prediction")
            print("=" * 60)
            print(prediction)

            predictions.append(
                {
                    "image_id": query_index,
                    "id": len(predictions) + 1,
                    "caption": prediction.strip(),
                }
            )

            cider_score = evaluate_current_predictions(
                dataset=dataset,
                predictions=predictions,
                mode=example_mode,
            )
            if cider_score is not None:
                print(f"\nCurrent cumulative CIDEr: {cider_score:.2f}")
                _, _, summary_csv_path, summary_txt_path = get_mode_eval_paths(example_mode)
                print(f"Saved summary to: {summary_csv_path}")
                print(f"Saved text report to: {summary_txt_path}")

            print("\n" + "=" * 60)
            print("Ground Truth")
            print("=" * 60)

            for index, caption in enumerate(
                ground_truth_captions,
                start=1,
            ):
                print(f"{index}. {caption}")

        if predictions:
            final_cider_score = evaluate_current_predictions(
                dataset=dataset,
                predictions=predictions,
                mode=example_mode,
            )

            print("\n" + "=" * 60)
            print(f"CIDEr Evaluation ({example_mode})")
            print("=" * 60)
            print(f"Evaluated samples: {len(predictions)}")
            print(f"Final CIDEr: {final_cider_score:.2f}")
            annotations_path, predictions_path, summary_csv_path, summary_txt_path = get_mode_eval_paths(example_mode)
            print(f"Annotations: {annotations_path}")
            print(f"Predictions: {predictions_path}")
            print(f"CSV summary: {summary_csv_path}")
            print(f"Text summary: {summary_txt_path}")


if __name__ == "__main__":
    main()
