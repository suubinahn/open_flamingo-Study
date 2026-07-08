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

print("Model loaded successfully!")