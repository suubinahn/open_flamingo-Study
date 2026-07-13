import torch

from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

from config import (
    CLIP_VISION_ENCODER,
    CLIP_PRETRAINED,
    LANGUAGE_MODEL,
    CHECKPOINT,
    CHECKPOINT_FILE,
    CROSS_ATTN_EVERY_N_LAYERS,
)

# ==========================
# Model Cache
# ==========================
_model = None
_image_processor = None
_tokenizer = None
_device = None


def load_model():
    """
    OpenFlamingo 모델 생성 및 체크포인트 로드
    (이미 로드되어 있으면 캐시된 모델 반환)
    """

    global _model
    global _image_processor
    global _tokenizer
    global _device

    # 이미 모델이 메모리에 있으면 그대로 반환
    if _model is not None:
        print("Using cached model.")
        return (
            _model,
            _image_processor,
            _tokenizer,
            _device,
        )

    print("Creating OpenFlamingo model...")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=CLIP_VISION_ENCODER,
        clip_vision_encoder_pretrained=CLIP_PRETRAINED,
        lang_encoder_path=LANGUAGE_MODEL,
        tokenizer_path=LANGUAGE_MODEL,
        cross_attn_every_n_layers=CROSS_ATTN_EVERY_N_LAYERS,
    )

    print("Downloading / Loading checkpoint...")

    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT,
        filename=CHECKPOINT_FILE,
    )

    model.load_state_dict(
        torch.load(checkpoint_path),
        strict=False,
    )

    # The available 6 GB GPU cannot hold the full OpenFlamingo 3B model.
    # Keep inference on CPU unless the loading strategy is changed (for
    # example, with quantization or CPU/GPU offloading).
    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # ==========================
    # Cache 저장
    # ==========================
    _model = model
    _image_processor = image_processor
    _tokenizer = tokenizer
    _device = device

    return (
        _model,
        _image_processor,
        _tokenizer,
        _device,
    )
