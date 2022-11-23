import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from tifffile import imwrite
import scipy.ndimage as ndimage


def make_artefact_labels(image, labels, artefact_threshold_adjustment=0):
    """Make a new label image with artefacts labelled as 1.
    Parameters
    ----------
    image : ndarray
        Image data.
    labels : ndarray
        Label image data.
    artefact_threshold_adjustment : float, optional
        Threshold adjustment parameter for artefact detection, it will multiply the standard deviation of the pixels value, by default 0
    Returns
    -------
    ndarray
        Label image with artefacts labelled as artefact_label.
    """
    neurones = np.array(labels > 0)
    non_neurones = np.array(labels == 0)

    # calculate mean intensity of all the pixels that are labeled as neurones
    mean = np.mean(image[neurones])

    # calculate standard deviation of the intensity of all the pixels that are labeled as neurones
    std = np.std(image[neurones])

    # calculate the threshold
    threshold = mean + artefact_threshold_adjustment * std
    # print("threshold", threshold, "mean", mean, "std", std)

    if threshold < 0:
        threshold = 0
        print("Warning: threshold for artefact detection is negative, setting to 0")

    # take all the pixels that are above the threshold and that are not labeled as neurones
    artefacts = np.where(image > threshold, 1, 0)
    artefacts = np.where(non_neurones, artefacts, 0)

    # calculate the mean size of the neurones
    neurone_size = np.mean(ndimage.sum(neurones, labels, range(np.max(labels) + 1)))
    # calculate the standard deviation of the neurone size
    neurone_size_std = np.std(ndimage.sum(neurones, labels, range(np.max(labels) + 1)))
    # print("mean neurone size: ", neurone_size, "std: ", neurone_size_std)

    # select artefacts by size
    artefacts = select_artefacts_by_size(
        artefacts, neurone_size - 0.5 * neurone_size_std
    )

    return artefacts


def select_artefacts_by_size(artefacts, min_size):
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
        Label image with artefacts labelled as 1 and small artefacts removed.
    """
    # find all the connected components in the artefacts image
    labels, nlabels = ndimage.label(artefacts)

    # find the size of each connected component
    sizes = ndimage.sum(artefacts, labels, range(nlabels + 1))

    # remove the small connected components
    mask_sizes = sizes < min_size
    remove_pixel = mask_sizes[
        labels
    ]  # where the mask is true and the label is not zero the pixel is removed
    labels[remove_pixel] = 0

    # relabel the image
    labels, nlabels = ndimage.label(labels)

    return labels


# load the dataset

im = "dataset/visual_tif/volumes/images.tif"
image = imread(im)

labeled_im = "dataset/visual_tif/labels/testing_im.tif"
labels = imread(labeled_im)

artefacts = make_artefact_labels(image, labels, artefact_threshold_adjustment=-1)

# save the artefact image
imwrite("dataset/visual_tif/artefact.tif", artefacts)
