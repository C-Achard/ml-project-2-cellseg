import numpy as np
from tifffile import imread
from tifffile import imwrite
from pathlib import Path
import scipy.ndimage as ndimage
import os
import napari
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from post_processing import binary_watershed


def map_labels(labels, artefacts):
    """Map the artefacts labels to the neurons labels.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    artefacts : ndarray
        Label image with artefacts labelled as mulitple values.
    Returns
    -------
    map_labels_existing: numpy array
        The label value of the artefact and the label value of the neurone associated or the neurons associated
    new_labels: list
        The labels of the artefacts that are not labelled in the neurons
    """
    map_labels_existing = []
    new_labels = []

    for i in np.unique(artefacts):
        if i == 0:
            continue
        indexes = labels[artefacts == i]
        # find the most common label in the indexes
        unique, counts = np.unique(indexes, return_counts=True)
        unique = np.flip(unique[np.argsort(counts)])
        counts = np.flip(counts[np.argsort(counts)])
        if unique[0] != 0:
            map_labels_existing.append(np.array([i, unique[np.argmax(counts)]]))
        elif (
            counts[0] < np.sum(counts) * 2 / 3.0
        ):  # the artefact is connected to multiple neurons
            total = 0
            ii = 1
            while total < np.size(indexes) / 3.0:
                total = np.sum(counts[1 : ii + 1])
                ii += 1
            map_labels_existing.append(np.append([i], unique[1 : ii + 1]))
        else:
            new_labels.append(i)

    return map_labels_existing, new_labels


def make_artefact_labels(
    image,
    labels,
    threshold_artefact_brightness_percent=40,
    threshold_artefact_size_percent=1,
    contrast_power=20,
    label_value=2,
    do_multi_label=False,
):
    """Detect pseudo nucleus.
    Parameters
    ----------
    image : ndarray
        Image.
    labels : ndarray
        Label image.
    threshold_artefact_brightness_percent : int, optional
        Threshold for artefact brightness.
    threshold_artefact_size_percent : int, optional
        Threshold for artefact size, if the artefcact is smaller than this percentage of the neurons it will be removed.
    contrast_power : int, optional
        Power for contrast enhancement.

    Returns
    -------
    ndarray
        Label image with pseudo nucleus labelled with 1 value per artefact.
    """

    neurons = np.array(labels > 0)
    non_neurons = np.array(labels == 0)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # calculate the percentile of the intensity of all the pixels that are labeled as neurons
    # check if the neurons are not empty
    if np.sum(neurons) > 0:
        threshold = np.percentile(image[neurons], threshold_artefact_brightness_percent)
    else:
        # take the percentile of the non neurons if the neurons are empty
        threshold = np.percentile(image[non_neurons], 90)

    # modify the contrast of the image accoring to the threshold with a tanh function and map the values to [0,1]

    image_contrasted = np.tanh((image - threshold) * contrast_power)
    image_contrasted = (image_contrasted - np.min(image_contrasted)) / (
        np.max(image_contrasted) - np.min(image_contrasted)
    )

    artefacts = binary_watershed(
        image_contrasted, thres_seeding=0.9, thres_small=30, thres_objects=0.4
    )

    # evaluate where the artefacts are connected to the neurons
    # map the artefacts label to the neurons label
    map_labels_existing, new_labels = map_labels(labels, artefacts)

    # remove the artefacts that are connected to the neurons
    for i in map_labels_existing:
        artefacts[artefacts == i[0]] = 0
    # remove all the pixels of the neurons from the artefacts
    artefacts = np.where(labels > 0, 0, artefacts)

    # remove the artefacts that are too small
    # calculate the percentile of the size of the neurons
    if np.sum(neurons) > 0:
        sizes = ndimage.sum_labels(labels > 0, labels, np.unique(labels))
        neurone_size_percentile = np.percentile(sizes, threshold_artefact_size_percent)
    else:
        # find the size of each connected component
        sizes = ndimage.sum_labels(labels > 0, labels, np.unique(labels))
        # remove the smallest connected components
        neurone_size_percentile = np.percentile(sizes, 95)

    artefacts = select_artefacts_by_size(
        artefacts, min_size=neurone_size_percentile, is_labeled=True
    )

    # relable with the label value if the artefacts are not multi label
    if not do_multi_label:
        artefacts = np.where(artefacts > 0, label_value, artefacts)

    return artefacts


def select_artefacts_by_size(artefacts, min_size, is_labeled=False):
    """Select artefacts by size.
    Parameters
    ----------
    artefacts : ndarray
        Label image with artefacts labelled as 1.
    min_size : int, optional
        Minimum size of artefacts to keep
    Returns
    -------
    ndarray
        Label image with artefacts labelled and small artefacts removed.
    """
    if not is_labeled:
        # find all the connected components in the artefacts image
        labels = ndimage.label(artefacts)[0]
    else:
        labels = artefacts

    # remove the small components
    for i in np.unique(labels):
        if i != 0:
            if np.sum(labels == i) < min_size:
                artefacts = np.where(labels == i, 0, artefacts)

    return artefacts


def create_artefact_labels(
    image_path,
    labels_path,
    output_path,
    threshold_artefact_brightness_percent=40,
    threshold_artefact_size_percent=1,
    contrast_power=20,
):
    """Create a new label image with artefacts labelled as 2 and neurons labelled as 1.
    Parameters
    ----------
    image_path : str
        Path to image file.
    labels_path : str
        Path to label image file with each neurons labelled as a different value.
    output_path : str
        Path to save the output label image file.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurons.
    contrast_power : int, optional
        Power for contrast enhancement.
    """
    image = imread(image_path)
    labels = imread(labels_path)

    artefacts = make_artefact_labels(
        image,
        labels,
        threshold_artefact_brightness_percent,
        threshold_artefact_size_percent,
        contrast_power=contrast_power,
        label_value=2,
        do_multi_label=False,
    )
    neurons_artefacts_labels = np.where(labels > 0, 1, artefacts)

    imwrite(output_path, neurons_artefacts_labels)


def visualize_images(paths):
    """Visualize images.
    Parameters
    ----------
    paths : list
        List of paths to images to visualize.
    """
    viewer = napari.Viewer()
    for path in paths:
        viewer.add_image(imread(path), name=os.path.basename(path))
    # wait for the user to close the viewer
    napari.run()


def create_artefact_labels_from_folder(
    path,
    do_visualize=False,
    threshold_artefact_brightness_percent=40,
    threshold_artefact_size_percent=1,
    contrast_power=20,
):
    """Create a new label image with artefacts labelled as 2 and neurons labelled as 1 for all images in a folder. The images created are stored in a folder artefact_neurons.
    Parameters
    ----------
    path : str
        Path to folder with images in folder volumes and labels in folder lab_sem. The images are expected to have the same alphabetical order in both folders.
    do_visualize : bool, optional
        If True, the images will be visualized.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurons.
    contrast_power : int, optional
        Power for contrast enhancement.
    """
    # find all the images in the folder and create a list
    path_labels = [f for f in os.listdir(path + "/labels") if f.endswith(".tif")]
    path_images = [f for f in os.listdir(path + "/volumes") if f.endswith(".tif")]
    # sort the list
    path_labels.sort()
    path_images.sort()
    # create the output folder
    os.makedirs(path + "/artefact_neurons", exist_ok=True)
    # create the artefact labels
    for i in range(len(path_labels)):
        print(path_labels[i])
        # consider that the images and the labels have names in the same alphabetical order
        create_artefact_labels(
            path + "/volumes/" + path_images[i],
            path + "/labels/" + path_labels[i],
            path + "/artefact_neurons/" + path_labels[i],
            threshold_artefact_brightness_percent,
            threshold_artefact_size_percent,
            contrast_power,
        )
        if do_visualize:
            visualize_images(
                [
                    path + "/volumes/" + path_images[i],
                    path + "/labels/" + path_labels[i],
                    path + "/artefact_neurons/" + path_labels[i],
                ]
            )


if __name__ == "__main__":

    repo_path = Path(__file__).resolve().parents[1]
    print(f"REPO PATH : {repo_path}")
    paths = [
        "dataset/cropped_visual/train",
        "dataset/cropped_visual/val",
        "dataset/somatomotor",
        "dataset/visual_tif",
    ]
    for data_path in paths:
        path = str(repo_path / data_path)
        print(path)
        create_artefact_labels_from_folder(
            path,
            do_visualize=False,
            threshold_artefact_brightness_percent=40,
            threshold_artefact_size_percent=1,
            contrast_power=20,
        )
