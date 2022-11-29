import numpy as np
from tifffile import imread
from tifffile import imwrite
import scipy.ndimage as ndimage
import os
import napari


def make_artefact_labels(
    image,
    labels,
    threshold_artefact_brightness_percent=50,
    threshold_artefact_size_percent=50,
    label_value=2,
    do_multi_label=False,
):
    """Make a new label image with artefacts labelled as 1.
    Parameters
    ----------
    image : ndarray
        Image data.
    labels : ndarray
        Label image data.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurones.
    label_value : int, optional
        The value to use for the artefact label.
    do_multi_label : bool, optional
        If True, the artefacts will be labelled with a different value for each artefact. If False, all artefacts will be labelled with the label_value.
    Returns
    -------
    ndarray
        Label image with artefacts labelled as artefact_label.
    """
    neurones = np.array(labels > 0)
    non_neurones = np.array(labels == 0)

    # calculate the percentile of the intensity of all the pixels that are labeled as neurones
    #check if the neurones are not empty
    if np.sum(neurones) > 0:
        threshold = np.percentile(image[neurones], threshold_artefact_brightness_percent)
    else:
        #take the percentile of the non neurones if the neurones are empty
        threshold = np.percentile(image[non_neurones], 90)

    # take all the pixels that are above the threshold and that are not labeled as neurones
    artefacts = np.where(image > threshold, 1, 0)
    artefacts = np.where(non_neurones, artefacts, 0)

    # calculate the percentile of the size of the neurones
    if np.sum(neurones) > 0:
        neurone_size_percentile = np.percentile(
            ndimage.sum(neurones, labels, range(np.max(labels) + 1)),
            threshold_artefact_size_percent,
        )
    else:
        # find all the connected components in the artefacts image
        labels, nlabels = ndimage.label(artefacts)
        # find the size of each connected component
        sizes = ndimage.sum(artefacts, labels, range(nlabels + 1))
        #remove the 50% smallest connected components
        neurone_size_percentile = np.percentile(sizes, 95)

    # select artefacts by size
    artefacts = select_artefacts_by_size(artefacts, neurone_size_percentile)

    # label the artefacts
    if not do_multi_label:
        artefacts = np.where(artefacts > 0, label_value, 0)

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
        Label image with artefacts labelled and small artefacts removed.
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

def create_artefact_labels(
    image_path,
    labels_path,
    output_path,
    threshold_artefact_brightness_percent=20,
    threshold_artefact_size_percent=10,
):
    """Create a new label image with artefacts labelled as 2 and neurones labelled as 1.
    Parameters
    ----------
    image_path : str
        Path to image file.
    labels_path : str
        Path to label image file.
    output_path : str
        Path to save the output label image file.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurones.
    """
    image = imread(image_path)
    labels = imread(labels_path)

    artefacts = make_artefact_labels(
        image,
        labels,
        threshold_artefact_brightness_percent,
        threshold_artefact_size_percent,
        label_value=2,
        do_multi_label=False,
    )
    neurones_artefacts_labels = np.where(labels > 0, 1, artefacts)

    imwrite(output_path, neurones_artefacts_labels)

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
    #wait for the user to close the viewer
    napari.run()

def create_artefact_labels_from_folder(
    path,
    do_visualize=False,
    threshold_artefact_brightness_percent=20,
    threshold_artefact_size_percent=10,
):
    """Create a new label image with artefacts labelled as 2 and neurones labelled as 1 for all images in a folder. The images created are stored in a folder artefact_neurones.
    Parameters
    ----------
    path : str
        Path to folder with images in folder volumes and labels in folder lab_sem. The images are expected to have the same alphabetical order in both folders.
    do_visualize : bool, optional
        If True, the images will be visualized.
    threshold_artefact_brightness_percent : int, optional
        The artefacts need to be as least as bright as this percentage of the neurone's pixels.
    threshold_artefact_size : int, optional
        The artefacts need to be at least as big as this percentage of the neurones.
    """
    #find all the images in the folder and create a list
    path_labels = [f for f in os.listdir(path+"/lab_sem") if f.endswith(".tif")]
    path_images = [f for f in os.listdir(path+"/volumes") if f.endswith(".tif")]
    #sort the list
    path_labels.sort()
    path_images.sort()
    #create the output folder
    os.makedirs(path+"/artefact_neurones", exist_ok=True)
    #create the artefact labels
    for i in range(len(path_labels)):
        print(path_labels[i])
        #consider that the images and the labels have names in the same alphabetical order
        create_artefact_labels(path+"/volumes/"+path_images[i], path+"/lab_sem/"+path_labels[i], path+"/artefact_neurones/"+path_labels[i], threshold_artefact_brightness_percent, threshold_artefact_size_percent)
        if do_visualize:
            visualize_images([path+"/volumes/"+path_images[i], path+"/lab_sem/"+path_labels[i], path+"/artefact_neurones/"+path_labels[i]])

"""

im = "dataset/visual_tif/volumes/images.tif"
labeled_im = "dataset/visual_tif/labels/testing_im.tif"


image = imread(im)


labels = imread(labeled_im)

artefacts = make_artefact_labels(
    image,
    labels,
    threshold_artefact_brightness_percent=20,
    threshold_artefact_size_percent=10,
    label_value=2,
    do_multi_label=False,
)

# save the artefact image
imwrite("dataset/visual_tif/artefact.tif", artefacts)
create_artefact_labels(im, labeled_im, "dataset/visual_tif/artefact_neurones.tif")
"""

paths=["dataset/cropped_visual/train","dataset/cropped_visual/val","dataset/somatomotor","dataset/visual_tif"]

for path in paths:
    print(path)
    create_artefact_labels_from_folder(path, do_visualize=False)


#create_artefact_labels("dataset/somatomotor/volumes/c3images.tif", "dataset/somatomotor/lab_sem/c3labels.tif", "dataset/somatomotor/artefact_neurones/c3labels.tif", threshold_artefact_brightness_percent=20, threshold_artefact_size_percent=2)
#visualize_images(["dataset/somatomotor/volumes/c3images.tif", "dataset/somatomotor/lab_sem/c3labels.tif", "dataset/somatomotor/artefact_neurones/c3labels.tif"])
