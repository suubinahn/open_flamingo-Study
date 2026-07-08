from PIL import Image
import requests

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch

print("Step 1: Creating model...")

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

print("Step 2: Downloading checkpoint...")

checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "checkpoint.pt"
)

print("Checkpoint:", checkpoint_path)

model.load_state_dict(torch.load(checkpoint_path), strict=False)

# 이미지 로드
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        stream=True,
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True,
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True,
    ).raw
)

# 이미지 전처리
vision_x = [
    image_processor(demo_image_one).unsqueeze(0),
    image_processor(demo_image_two).unsqueeze(0),
    image_processor(query_image).unsqueeze(0),
]

vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

# 텍스트 Prompt
tokenizer.padding_side = "left"

lang_x = tokenizer(
    [
        "<image>An image of two cats.<|endofchunk|>"
        "<image>An image of a bathroom sink.<|endofchunk|>"
        "<image>An image of"
    ],
    return_tensors="pt",
)

# GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

vision_x = vision_x.to(device)

lang_x = {
    k: v.to(device)
    for k, v in lang_x.items()
}

# Generate
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

# 출력
print(tokenizer.decode(generated_text[0]))

# print("Model loaded successfully!")