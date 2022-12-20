import tifffile as tiff
import numpy as np
import random
import os

"""
New code by Yves Paychère
Adds artifacts to images to make training more challenging
"""


def add_artefact(
    path_image, path_label, path_artefact_image, path_out, min_x=0, seed=1
):
    """
    Add artefact to image
    :param path_image: path to image
    :param path_label: path to label
    :param path_artefact_image: path to artefact image
    :param path_out: path to output image
    :param min_x: minimum x position to place the artefact
    :param seed: seed for random number generator
    """
    image = tiff.imread(path_image)
    artefact = tiff.imread(path_artefact_image)
    label = tiff.imread(path_label)
    random.seed(seed)
    # find the background intensity of the image
    background = np.median(image[label == 0])
    # set the background of the artefact to the background of the image
    artefact[artefact == 0] = background

    # if the artefact is bigger than the image, we crop it
    if (
        artefact.shape[0] > image.shape[0] - min_x
        or artefact.shape[1] > image.shape[1]
        or artefact.shape[2] > image.shape[2]
    ):
        artefact = artefact[
            0 : image.shape[0] - min_x, 0 : image.shape[1], 0 : image.shape[2]
        ]

    # randomly select a position in the image
    x = random.randint(min_x, image.shape[0] - artefact.shape[0])
    y = random.randint(0, image.shape[1] - artefact.shape[1])
    z = random.randint(0, image.shape[2] - artefact.shape[2])
    out = np.copy(image)
    out[
        x : x + artefact.shape[0], y : y + artefact.shape[1], z : z + artefact.shape[2]
    ] = artefact
    # where the neurons are labelled, we use the original image
    out[label > 0] = image[label > 0]
    tiff.imwrite(path_out, out)
    print("Artefact added to image: " + path_image)


def add_artefacts(
    path_image, path_label, paths_artefact_image, path_out, min_x=0, seed=1
):
    """
    Add multiple artefacts to image
    :param path_image: path to image
    :param path_label: path to label
    :param paths_artefact_image: list of paths to artefact images
    :param min_x: minimum x position to place the artefact
    :param path_out: path to output image
    """
    for path_artefact_image in paths_artefact_image:
        add_artefact(path_image, path_label, path_artefact_image, path_out, min_x, seed)
        path_image = path_out
        seed += 1


def add_artefacts_to_folder(
    path_folder_image,
    path_folder_label,
    paths_artefact_image,
    path_out,
    min_x=0,
    seed=1,
):
    """
    Add multiple artefacts to all images in a folder
    :param path_folder_image: path to folder with images
    :param path_folder_label: path to folder with labels
    :param paths_artefact_image: list of paths to artefact images
    :param path_out: path to output folder
    :param min_x: minimum x position to place the artefact
    :param seed: seed for random number generator
    """
    images_path = os.listdir(path_folder_image)
    images_path = [f for f in images_path if f.endswith(".tif")]
    images_path.sort()
    labels_path = os.listdir(path_folder_label)
    labels_path = [f for f in labels_path if f.endswith(".tif")]
    labels_path.sort()
    for i in range(len(images_path)):
        path_image = os.path.join(path_folder_image, images_path[i])
        path_label = os.path.join(path_folder_label, labels_path[i])
        out_name = str(images_path[i]).replace(".tif", "_with_artefact.tif")
        path_out_image = os.path.join(path_out, out_name)
        paths_artefact_image_random = random.sample(
            paths_artefact_image, random.randint(1, len(paths_artefact_image))
        )
        add_artefacts(
            path_image,
            path_label,
            paths_artefact_image_random,
            path_out_image,
            min_x,
            seed,
        )
        seed += 1
