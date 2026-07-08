from utils.model_loader import load_model
from utils.image_utils import (
    load_image_from_path,
    prepare_images,
)
from inference import generate_text


def build_prompt(
    example_question,
    example_answer1,
    example_answer2,
    query_question,
):
    """
    Few-shot VQA Prompt 생성
    """

    return (
        f"<image>Question: {example_question}\n"
        f"Answer: {example_answer1}<|endofchunk|>"
        f"<image>Question: {example_question}\n"
        f"Answer: {example_answer2}<|endofchunk|>"
        f"<image>Question: {query_question}\n"
        "Answer:"
    )


def load_examples():

    print("\n========== Few-shot Example 설정 ==========\n")

    example1_path = input("Example 1 이미지 경로 : ").strip()
    example2_path = input("Example 2 이미지 경로 : ").strip()

    example_question = input("\nExample Question : ").strip()

    example_answer1 = input("Example 1 Answer : ").strip()
    example_answer2 = input("Example 2 Answer : ").strip()

    example_images = [
        load_image_from_path(example1_path),
        load_image_from_path(example2_path),
    ]

    return (
        example_images,
        example_question,
        example_answer1,
        example_answer2,
    )


def main():

    print("=" * 60)
    print("OpenFlamingo Few-shot VQA Demo")
    print("=" * 60)

    # ===================================================
    # 모델은 단 한 번만 로드
    # ===================================================
    model, image_processor, tokenizer, device = load_model()

    print("\n모델 준비 완료!")

    (
        example_images,
        example_question,
        example_answer1,
        example_answer2,
    ) = load_examples()

    while True:

        print("\n----------------------------------------")
        print("Query 이미지 입력")
        print("(change : Example 변경)")
        print("(q : 종료)")
        print("----------------------------------------")

        command = input(">> ").strip()

        if command.lower() == "q":
            print("\n프로그램 종료")
            break

        if command.lower() == "change":

            (
                example_images,
                example_question,
                example_answer1,
                example_answer2,
            ) = load_examples()

            continue

        query_path = command

        query_question = input(
            "Query Question : "
        ).strip()

        try:

            query_image = load_image_from_path(query_path)

            images = [
                example_images[0],
                example_images[1],
                query_image,
            ]

            vision_x = prepare_images(
                image_processor,
                images,
            )

            prompt = build_prompt(
                example_question,
                example_answer1,
                example_answer2,
                query_question,
            )

            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                vision_x=vision_x,
                prompt=prompt,
                device=device,
            )

            print("\n========== RESULT ==========\n")
            print(result)

        except Exception as e:

            print(f"\n오류 발생 : {e}")


if __name__ == "__main__":
    main()