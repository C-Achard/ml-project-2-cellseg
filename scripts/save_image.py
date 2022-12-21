from tifffile import imread, imwrite
import numpy as np
from pathlib import Path


def save_image_as_type(path):
    image = imread(path)
    image = image.astype(np.uint8)
    imwrite(path, image)


if __name__ == "__main__":
    repo_path = Path(__file__).resolve().parents[1]
    print(f"REPO PATH : {repo_path}")
    path = repo_path / "dataset/visual_tif/lab_sem/testing_im.tif"
    save_image_as_type(path)
    image = imread(path)
    print(image.dtype)
