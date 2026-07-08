import torch
import requests

from PIL import Image


def load_image_from_url(url):
    """
    URL에서 이미지를 불러온다.
    """
    return Image.open(
        requests.get(url, stream=True).raw
    )


def load_image_from_path(path):
    """
    로컬 이미지를 불러온다.
    """
    return Image.open(path)


def prepare_images(image_processor, images):
    """
    여러 장의 이미지를 OpenFlamingo 입력 형태로 변환
    """

    vision_x = [
        image_processor(image).unsqueeze(0)
        for image in images
    ]

    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    return vision_x