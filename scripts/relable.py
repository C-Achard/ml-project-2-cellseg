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
import scripts.make_artefact_labels as make_artefact_labels
import time
import warnings
from napari.qt.threading import thread_worker
from tqdm import tqdm
import threading


def relable_non_unique_i(label, save_path, go_fast=False):
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

def add_label(old_label,artefact,new_label_path,i_labels_to_add):
    """add the label to the label image
    Parameters
    ----------
    old_label : np.array
        the label image
    artefact : np.array
        the artefact image that contains some neurons
    new_label_path : str
        the path to save the new label image
    """
    new_label = old_label.copy()
    max_label = np.max(old_label)
    for i,i_label in enumerate(i_labels_to_add):
        new_label[artefact == i_label] = i+max_label+1
    imwrite(new_label_path, new_label)
returns=[]
def ask_labels(unique_artefact):
    global returns
    returns=[]
    i_labels_to_add_tmp = input("Which label do you want to add ? (separated by a comma):")
    i_labels_to_add_tmp = [int(i) for i in i_labels_to_add_tmp.split(",")]
    for i in i_labels_to_add_tmp:
        if i == 0:
            print("0 is not a valid label")
            # delete the 0
            i_labels_to_add_tmp.remove(i)
    # test if all index are negative
    if all(i < 0 for i in i_labels_to_add_tmp):
        print("all labels are negative-> will add all the labels except the one you gave")
        i_labels_to_add = list(unique_artefact)
        for i in i_labels_to_add_tmp:
            if np.abs(i) in i_labels_to_add:
                i_labels_to_add.remove(np.abs(i))
            else:
                print("the label", np.abs(i), "is not in the label image")
        i_labels_to_add_tmp = i_labels_to_add
    else:
        # remove the negative index
        for i in i_labels_to_add_tmp:
            if i < 0:
                i_labels_to_add_tmp.remove(i)
                print("ignore the negative label", i," since not all the labels are negative")
            if i not in unique_artefact:
                print("the label", i, "is not in the label image")
                i_labels_to_add_tmp.remove(i)
            
    returns=[i_labels_to_add_tmp]

def relable(image_path,label_path, go_fast=False, check_for_unicity=True,delay=0.3):
    """relable the image labelled with different label for each neuron and save it in the save_path location
    Parameters
    ----------
    label_path : str
        the path to the label image
    """
    global returns

    label = imread(label_path)
    initial_label_path = label_path
    if check_for_unicity:
        # check if the label are unique
        new_label_path = label_path[:-4] + "_relable_unique.tif"
        map_labels_existing = relable_non_unique_i(label, new_label_path, go_fast=go_fast)
        print("visualize the relabled image in white the previous labels and in red the new labels")
        visualize_map(map_labels_existing, label_path, new_label_path,delay=delay)
        label_path = new_label_path
    #detect artefact
    print("detection of artefact (in process)")
    image=imread(image_path)
    artefact=make_artefact_labels.make_artefact_labels(image,imread(label_path),do_multi_label=True)
    print("detection of artefact (done)")
    #ask the user if the artefact are not neurons
    i_labels_to_add=[]
    loop=True
    unique_artefact=list(np.unique(artefact))
    while loop:
        #visualize the artefact and ask the user which label to add to the label image
        t=threading.Thread(target=ask_labels,args=(unique_artefact,))
        t.start()
        artefact_copy=np.where(np.isin(artefact,i_labels_to_add),0,artefact)
        viewer=napari.view_image(image)
        viewer.add_labels(artefact_copy)
        napari.run()
        t.join()
        i_labels_to_add_tmp=returns[0]
        #check if the selected labels are neurones
        for i in i_labels_to_add:
            if i not in i_labels_to_add_tmp:
                i_labels_to_add_tmp.append(i)
        artefact_copy=np.where(np.isin(artefact,i_labels_to_add_tmp),artefact,0)
        print("these labels will be added")
        viewer=napari.view_image(image)
        viewer.add_labels(artefact_copy)
        napari.run()
        revert = input("Do you want to revert? (y/n)")
        if revert != "y":
            i_labels_to_add=i_labels_to_add_tmp
            for i in i_labels_to_add:
                if i in unique_artefact:
                    unique_artefact.remove(i)
        loop=input("Do you want to add more labels? (y/n)") == "y"
    #add the label to the label image
    new_label_path = initial_label_path[:-4] + "_new_label.tif"
    add_label(imread(label_path),artefact,new_label_path,i_labels_to_add)
    #store the artefact remaining
    new_artefact_path = initial_label_path[:-4] + "_artefact.tif"
    artefact=np.where(np.isin(artefact,i_labels_to_add),0,artefact)
    imwrite(new_artefact_path, artefact)

        

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
    time.sleep(2)
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


def relable_non_unique_i_folder(folder_path, end_of_new_name):
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
            relable_non_unique_i(
                label, os.path.join(folder_path, file[:-4] + end_of_new_name + ".tif")
            )


if __name__ == "__main__":

    repo_path = Path(__file__).resolve().parents[1]
    file_path = os.path.join(
        repo_path, "dataset", "visual_tif", "labels", "testing_im.tif"
    )
    image_path = os.path.join(repo_path, "dataset", "visual_tif", "volumes", "images.tif")

    relable(image_path,file_path, check_for_unicity=True, go_fast=False)
