import numpy as np
import napari
from pathlib import Path
from tifffile import imread, imwrite
from skimage.morphology import remove_small_objects

"""
New code to label axons in the TRAILMAP dataset
Author : Cyril Achard
"""

def normalize(image):
    return (image - image.mean()) / image.std()


def labeL_axons(volume_directory):
    print(volume_directory)
    images_filepaths = sorted(
        [str(file) for file in Path(volume_directory).glob("*.tiff")]
    )
    print(f"Images : {images_filepaths}")
    images = [imread(image) for image in images_filepaths]
    images = [normalize(im) for im in images]

    labels = [np.where(im > 0.99, True, False) for im in images]
    labels = [remove_small_objects(lab, 70) for lab in labels]
    labels = [np.where(lab == 1, 2, 0) for lab in labels]

    save_path = Path(volume_directory) / "results"
    save_path.mkdir(exist_ok=True)

    [
        imwrite(str(save_path / f"label_{i}.tif"), lab.astype(np.float32))
        for i, lab in enumerate(labels)
    ]
    [
        imwrite(str(save_path / f"volume_{i}.tif"), im.astype(np.float32))
        for i, im in enumerate(images)
    ]

    view = napari.viewer.Viewer()

    [view.add_labels(lab) for lab in labels]
    [view.add_image(im) for im in images]

    view.window.resize(3000, 1400)
    napari.run()


if __name__ == "__main__":
    repo_path = Path(__file__).resolve().parents[1]
    print(f"REPO PATH : {repo_path}")
    # path = repo_path / "dataset/axons/training/training-set/volumes"
    path = repo_path / "dataset/axons/validation/validation-set/volumes"

    labeL_axons(str(path))
