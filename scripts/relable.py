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
import time
import warnings
from napari.qt.threading import thread_worker
from tqdm import tqdm


def relable(label, save_path, go_fast=False):
    """relable the image labelled with different label for each neuron and save it in the save_path location
    Parameters
    ----------
    label : np.array
        the label image
    save_path : str
        the path to save the relabled image
    """
    value_label = 0
    new_labels = np.zeros_like(label)
    map_labels_existing = []
    unique_label = np.unique(label)
    for i_label in tqdm(range(len(unique_label)), desc="relabeling", ncols=100):
        i = unique_label[i_label]
        if i == 0:
            continue
        if go_fast:
            new_label, to_add = ndimage.label(label == i)
            map_labels_existing.append(
                [i, list(range(value_label + 1, value_label + to_add + 1))]
            )

        else:
            # catch the warning of the watershed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_label = binary_watershed(label == i)
                unique = np.unique(new_label)
                to_add = unique[-1]
                map_labels_existing.append([i, unique[1:] + value_label])

        new_label[new_label != 0] += value_label
        new_labels += new_label
        value_label += to_add



    imwrite(save_path, new_labels)
    return map_labels_existing


def modify_viewer(old_label, new_label, args):
    """modify the viewer to show the relabeling
    Parameters
    ----------
    map_labels_existing : list
        the list of the relabeling
    """
    if args == "hide new label":
        new_label.visible = False
    elif args == "show new label":
        new_label.visible = True
    else:
        old_label.selected_label = args[0]
        new_label.selected_label = args[1]


@thread_worker
def to_show(map_labels_existing, delay=0.5):
    """modify the viewer to show the relabeling
    Parameters
    ----------
    map_labels_existing : list
        the list of the relabeling
    """
    time.sleep(4)
    for i in map_labels_existing:
        yield "hide new label"
        yield [i[0], i[1][0]]
        time.sleep(delay)
        yield "show new label"
        for j in i[1]:
            yield [i[0], j]
            time.sleep(delay)


def create_connected_widget(old_label, new_label, map_labels_existing, delay=0.5):
    """Builds a widget that can control a function in another thread."""

    worker = to_show(map_labels_existing, delay)
    worker.start()
    worker.yielded.connect(lambda arg: modify_viewer(old_label, new_label, arg))


def visualize_map(map_labels_existing, label_path, relable_path, delay=0.5):
    """visualize the map of the relabeling
    Parameters
    ----------
    map_labels_existing : list
        the list of the relabeling
    """
    label = imread(label_path)
    relable = imread(relable_path)

    viewer = napari.Viewer()

    old_label = viewer.add_labels(label, num_colors=1)
    new_label = viewer.add_labels(relable, num_colors=1)
    old_label.colormap.colors = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    new_label.colormap.colors = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])

    viewer.dims.ndisplay = 3
    viewer.camera.angles = (180, 3, 50)
    viewer.camera.zoom = 1

    old_label.show_selected_label = True
    new_label.show_selected_label = True

    create_connected_widget(old_label, new_label, map_labels_existing, delay=delay)
    napari.run()


def relable_folder(folder_path, end_of_new_name):
    """relable the image labelled with different label for each neuron and save it in the save_path location
    Parameters
    ----------
    folder_path : str
        the path to the folder containing the label images
    save_path : str
        the path to save the relabled image
    """
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            label = imread(os.path.join(folder_path, file))
            relable(
                label, os.path.join(folder_path, file[:-4] + end_of_new_name + ".tif")
            )


if __name__ == "__main__":

    repo_path = Path(__file__).resolve().parents[1]
    file_path = os.path.join(
        repo_path, "dataset", "visual_tif", "labels", "testing_im.tif"
    )

    label = imread(file_path)
    map = relable(label, file_path[:-4] + "_relable.tif")
    visualize_map(map, file_path, file_path[:-4] + "_relable.tif", delay=0.3)
