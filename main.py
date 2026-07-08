from utils.model_loader import load_model
from utils.image_utils import (
    load_image_from_path,
    prepare_images,
)
from inference import generate_text


def build_prompt(example1_caption, example2_caption):
    return (
        f"<image>An image of {example1_caption}<|endofchunk|>"
        f"<image>An image of {example2_caption}<|endofchunk|>"
        "<image>An image of"
    )


def load_example_images():

    print("\n========== Example 설정 ==========")

    example1_path = input("Example 1 이미지 경로 : ").strip()
    example2_path = input("Example 2 이미지 경로 : ").strip()

    example1_caption = input("Example 1 설명 : ").strip()
    example2_caption = input("Example 2 설명 : ").strip()

    example_images = [
        load_image_from_path(example1_path),
        load_image_from_path(example2_path),
    ]

    prompt = build_prompt(
        example1_caption,
        example2_caption,
    )

    return example_images, prompt


def main():

    print("=" * 60)
    print("OpenFlamingo Interactive Demo")
    print("=" * 60)

    # ======================================================
    # 모델은 프로그램 시작 시 단 한 번만 로드
    # ======================================================
    model, image_processor, tokenizer, device = load_model()

    print("\n모델 준비 완료!")

    # 처음 Example 설정
    example_images, prompt = load_example_images()

    while True:

        print("\n------------------------------------------")
        print("Query 이미지 경로 입력")
        print("(change : Example 변경)")
        print("(q : 종료)")
        print("------------------------------------------")

        query_path = input(">> ").strip()

        if query_path.lower() == "q":
            print("프로그램을 종료합니다.")
            break

        if query_path.lower() == "change":

            try:
                example_images, prompt = load_example_images()
                print("\nExample가 변경되었습니다.")
            except Exception as e:
                print(f"\nExample 변경 실패 : {e}")

            continue

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

            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                vision_x=vision_x,
                prompt=prompt,
                device=device,
            )

            print("\n========== RESULT ==========")
            print(result)

        except Exception as e:
            print(f"\n오류 발생 : {e}")


if __name__ == "__main__":
    main()