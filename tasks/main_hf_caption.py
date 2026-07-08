import random
from difflib import SequenceMatcher

from datasets import load_dataset

from utils.model_loader import load_model
from utils.image_utils import prepare_images
from inference import generate_text


def build_prompt(caption1, caption2):

    return (
        f"<image>Output:{caption1}<|endofchunk|>"
        f"<image>Output:{caption2}<|endofchunk|>"
        "<image>Output:"
    )


def find_similar_examples(dataset, query_caption, query_index, num_examples=2):

    scores = []

    query_caption = query_caption.lower()

    for i, sample in enumerate(dataset):

        # Query 자기 자신 제외
        if i == query_index:
            continue

        candidate = sample["answer"][0].lower()

        score = SequenceMatcher(
            None,
            query_caption,
            candidate,
        ).ratio()

        scores.append((score, i))

    scores.sort(reverse=True)

    indices = [idx for _, idx in scores[:num_examples]]

    return indices


def main():

    print("=" * 60)
    print("OpenFlamingo + HuggingFace COCO Caption")
    print("=" * 60)

    # ----------------------------------------------------
    # 모델 로드 (한 번만)
    # ----------------------------------------------------
    print("\nLoading Model...")

    model, image_processor, tokenizer, device = load_model()

    print("Model Loaded!")

    # ----------------------------------------------------
    # Dataset 로드 (한 번만)
    # ----------------------------------------------------
    print("\nLoading Dataset...")

    dataset = load_dataset(
        "lmms-lab/COCO-Caption",
        split="val",
    )

    print("Dataset Loaded!")

    # ----------------------------------------------------
    # 반복 추론
    # ----------------------------------------------------
    while True:

        print("\n" + "=" * 60)
        print("Enter : Next Query")
        print("q     : Quit")
        print("=" * 60)

        cmd = input(">> ").strip().lower()

        if cmd == "q":
            print("프로그램을 종료합니다.")
            break

        # ----------------------------------------------------
        # 랜덤 Query 선택
        # ----------------------------------------------------
        query_index = random.randint(0, len(dataset) - 1)

        query = dataset[query_index]

        query_caption = query["answer"][0]

        print("\n========== Query ==========")
        print(query_caption)

        # ----------------------------------------------------
        # 비슷한 Example 선택
        # ----------------------------------------------------
        indices = find_similar_examples(
            dataset,
            query_caption,
            query_index,
        )

        example1 = dataset[indices[0]]
        example2 = dataset[indices[1]]

        print("\n========== Example 1 ==========")
        print(example1["answer"][0])

        print("\n========== Example 2 ==========")
        print(example2["answer"][0])

        # ----------------------------------------------------
        # Prompt 생성
        # ----------------------------------------------------
        prompt = build_prompt(
            example1["answer"][0],
            example2["answer"][0],
        )

        print("\n========== Prompt ==========")
        print(prompt)

        # ----------------------------------------------------
        # 이미지 준비
        # ----------------------------------------------------
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
        # 추론
        # ----------------------------------------------------
        prediction = generate_text(
            model=model,
            tokenizer=tokenizer,
            vision_x=vision_x,
            prompt=prompt,
            device=device,
        )

        # ----------------------------------------------------
        # 결과 출력
        # ----------------------------------------------------
        print("\n============================================================")
        print("Prediction")
        print("============================================================")

        print(prediction)

        print("\n============================================================")
        print("Ground Truth")
        print("============================================================")

        for i, caption in enumerate(query["answer"], start=1):
            print(f"{i}. {caption}")


if __name__ == "__main__":
    main()