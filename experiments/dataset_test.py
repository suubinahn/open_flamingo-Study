from datasets import load_dataset

print("=" * 60)
print("Loading COCO Caption Dataset...")
print("=" * 60)

# Dataset 다운로드
dataset = load_dataset(
    "lmms-lab/COCO-Caption",
    split="val",
)

print("\nDataset Loaded!\n")

print(dataset)

print("\n" + "=" * 60)
print("Sample 0")
print("=" * 60)

sample = dataset[0]

# 하나씩 꺼내기
image = sample["image"]
question = sample["question"]
answers = sample["answer"]

print("\n========== Extract ==========\n")

print("Question")
print(question)

print("\nGround Truth")

for i, ans in enumerate(answers, 1):
    print(f"{i}. {ans}")

print("\nImage Type")
print(type(image))

print("\nImage Size")
print(image.size)

# 전체 샘플 출력
print(sample)

print("\n" + "=" * 60)
print("Data Type")
print("=" * 60)

print(f"sample          : {type(sample)}")
print(f"image           : {type(sample['image'])}")
print(f"question        : {type(sample['question'])}")
print(f"answer          : {type(sample['answer'])}")

print("\n" + "=" * 60)
print("Image Info")
print("=" * 60)

print(f"File Name       : {sample['file_name']}")
print(f"Image Size      : {sample['width']} x {sample['height']}")
print(f"Image Object    : {sample['image']}")

print("\n" + "=" * 60)
print("Question")
print("=" * 60)

print(sample["question"])

print("\n" + "=" * 60)
print("Ground Truth Captions")
print("=" * 60)

for idx, caption in enumerate(sample["answer"], start=1):
    print(f"{idx}. {caption}")