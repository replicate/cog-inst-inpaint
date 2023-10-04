import os
import tarfile
import subprocess
from typing import Tuple

from PIL import Image
from torchvision.transforms import ToTensor


to_tensor = ToTensor()


def preprocess_image(
    image: Image, resize_shape: Tuple[int, int] = (256, 256), center_crop=True
):
    pil_image = image

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        pil_image = image.crop((left, top, right, bottom))

    pil_image = pil_image.resize(resize_shape)
    
    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0) * 2 - 1
    return tensor_image, pil_image


def download_model(url, dest):
    if not os.path.exists("/src/tmp.tar"):
        print("Downloading weights...")
        try:
            output = subprocess.check_output(["pget", "-x", url, "/src/tmp"])
            print(output)
        except subprocess.CalledProcessError as e:
            # If download fails, clean up and re-raise exception
            print(e.output)
            raise e
    
    os.rename("/src/tmp/", dest)
