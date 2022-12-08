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
import threading
import warnings
import io


def relable(label, save_path, go_fast=True, verbose=True):
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
    for i in np.unique(label):
        if i == 0:
            continue
        if go_fast:
            new_label, to_add = ndimage.label(label == i)
            map_labels_existing.append(
                [i, list(range(value_label, value_label + to_add + 1))]
            )

        else:
            new_label = binary_watershed(label == i)
            unique = np.unique(new_label)
            to_add = unique[-1]
            map_labels_existing.append([i, unique[1:] + value_label])

        new_label[new_label != 0] += value_label
        new_labels += new_label
        value_label += to_add

        if verbose:
            print(
                "label", i, "relabled", value_label, "will be added to the next label"
            )

    imwrite(save_path, new_labels)
    return map_labels_existing

def modify_viewer(map_labels_existing, viewer):
    """modify the viewer to show the relabeling
    Parameters
    ----------
    map_labels_existing : list
        the list of the relabeling
    viewer : napari.Viewer
        the napari viewer
    """
    print("start")
    time.sleep(4)

    view_label = viewer.layers[0]
    view_relabel = viewer.layers[1]
    view_label.show_selected_label = True
    view_relabel.show_selected_label = True
    for i in map_labels_existing:
        view_label.selected_label = i[0]
        view_relabel.visible = False
        time.sleep(0.5)
        for j in i[1]:
            view_relabel.selected_label = j
            view_relabel.visible = True
            time.sleep(0.5)
    
    # now restore stdout function
    sys.stdout = sys.__stdout__


def visualize_map(map_labels_existing, label_path, relable_path):
    """visualize the map of the relabeling
    Parameters
    ----------
    map_labels_existing : list
        the list of the relabeling
    """
    label = imread(label_path)
    relable = imread(relable_path)

    viewer = napari.Viewer()

    viewer.add_labels(label)
    viewer.add_labels(relable)

    viewer.dims.ndisplay = 3
    viewer.camera.angles = (180, 3, 50)
    viewer.camera.zoom =1

    warnings.filterwarnings("ignore", message="parent")
    text_trap = io.StringIO()
    sys.stdout = text_trap
    sys.stderr = text_trap
    sys.std
    
    modify_viewer(map_labels_existing, viewer)

    t = threading.Thread(target=modify_viewer, args=(map_labels_existing, viewer))
    t.daemon = True
    t.start()

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
    map=relable(label,file_path[:-4]+"_relable.tif",go_fast=True)
    """
    map = [
        [1, [0, 1, 2, 3]],
        [2, [3, 4, 5, 6, 7, 8]],
        [3, [8, 9, 10, 11, 12]],
        [4, [12, 13, 14]],
        [5, [14, 15]],
        [6, [15, 16, 17, 18]],
        [7, [18, 19, 20, 21]],
        [8, [21, 22, 23, 24, 25, 26]],
        [9, [26, 27, 28, 29, 30, 31]],
        [10, [31, 32, 33, 34]],
        [11, [34, 35, 36, 37, 38, 39]],
        [12, [39, 40, 41, 42, 43]],
        [13, [43, 44, 45, 46, 47, 48]],
        [14, [48, 49, 50, 51]],
        [15, [51, 52, 53, 54, 55]],
        [16, [55, 56, 57, 58, 59, 60]],
        [17, [60, 61, 62, 63, 64, 65]],
        [18, [65, 66, 67, 68, 69, 70]],
        [19, [70, 71, 72, 73, 74]],
        [20, [74, 75, 76]],
        [21, [76, 77]],
        [22, [77, 78, 79, 80, 81]],
        [23, [81, 82, 83, 84]],
        [24, [84, 85, 86, 87, 88]],
        [25, [88, 89, 90]],
        [26, [90, 91, 92, 93, 94, 95]],
        [27, [95, 96, 97, 98]],
        [28, [98, 99, 100, 101, 102, 103, 104, 105]],
        [29, [105, 106, 107, 108]],
        [30, [108, 109, 110, 111, 112, 113, 114, 115, 116]],
        [31, [116, 117, 118, 119]],
        [32, [119, 120]],
        [33, [120, 121, 122, 123, 124]],
        [34, [124, 125, 126, 127, 128, 129]],
        [35, [129, 130, 131, 132, 133, 134, 135, 136]],
        [36, [136, 137, 138, 139, 140, 141, 142, 143, 144, 145]],
        [37, [145, 146, 147, 148, 149]],
        [38, [149, 150, 151, 152, 153, 154]],
        [39, [154, 155, 156, 157, 158, 159]],
        [40, [159, 160, 161]],
        [41, [161, 162, 163, 164]],
        [42, [164, 165]],
        [43, [165, 166, 167, 168]],
        [44, [168, 169, 170, 171, 172]],
        [45, [172, 173, 174, 175]],
        [46, [175, 176, 177, 178, 179]],
        [47, [179, 180, 181]],
        [48, [181, 182, 183, 184]],
        [49, [184, 185, 186, 187, 188, 189, 190, 191]],
        [50, [191, 192, 193, 194, 195, 196, 197]],
        [51, [197, 198, 199, 200]],
        [52, [200, 201, 202, 203, 204, 205, 206, 207, 208]],
        [53, [208, 209]],
        [54, [209, 210, 211, 212, 213, 214, 215, 216]],
        [55, [216, 217, 218, 219]],
        [56, [219, 220]],
        [57, [220, 221]],
        [58, [221, 222, 223, 224, 225, 226]],
        [60, [226, 227, 228]],
        [61, [228, 229, 230, 231, 232]],
        [62, [232, 233, 234, 235, 236, 237, 238]],
        [63, [238, 239, 240, 241, 242, 243]],
        [64, [243, 244, 245, 246, 247, 248, 249, 250]],
        [65, [250, 251, 252, 253, 254, 255, 256]],
        [66, [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266]],
        [67, [266, 267, 268, 269, 270, 271]],
        [68, [271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283]],
        [69, [283, 284, 285, 286, 287, 288, 289]],
        [70, [289, 290, 291, 292, 293, 294, 295, 296]],
        [71, [296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306]],
        [72, [306, 307, 308, 309, 310, 311, 312, 313, 314]],
        [73, [314, 315, 316, 317, 318, 319, 320, 321, 322]],
        [74, [322, 323, 324]],
        [75, [324, 325, 326, 327, 328, 329, 330, 331]],
        [76, [331, 332, 333, 334, 335, 336, 337, 338, 339]],
        [77, [339, 340, 341, 342, 343, 344, 345, 346]],
        [78, [346, 347, 348, 349, 350, 351, 352]],
        [79, [352, 353, 354]],
        [80, [354, 355, 356, 357, 358]],
        [81, [358, 359, 360, 361, 362]],
        [82, [362, 363, 364, 365, 366]],
        [83, [366, 367, 368, 369, 370, 371]],
        [84, [371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382]],
        [85, [382, 383, 384, 385, 386, 387]],
        [86, [387, 388, 389, 390, 391]],
        [87, [391, 392, 393, 394, 395]],
        [88, [395, 396, 397, 398]],
        [89, [398, 399, 400]],
        [90, [400, 401, 402, 403, 404, 405, 406]],
        [91, [406, 407, 408, 409, 410, 411, 412, 413]],
        [92, [413, 414, 415, 416, 417]],
        [93, [417, 418, 419]],
        [94, [419, 420, 421, 422, 423, 424, 425, 426]],
        [95, [426, 427, 428, 429, 430, 431, 432]],
        [96, [432, 433, 434, 435, 436, 437]],
        [97, [437, 438, 439, 440, 441, 442, 443]],
        [98, [443, 444, 445, 446, 447, 448]],
        [99, [448, 449, 450, 451]],
        [100, [451, 452, 453, 454, 455]],
        [101, [455, 456]],
        [102, [456, 457, 458, 459, 460, 461, 462]],
        [103, [462, 463, 464, 465, 466]],
        [104, [466, 467, 468, 469, 470]],
        [105, [470, 471, 472, 473, 474, 475, 476, 477]],
        [107, [477, 478, 479, 480]],
        [108, [480, 481, 482, 483]],
        [109, [483, 484, 485, 486, 487, 488, 489]],
        [110, [489, 490, 491, 492, 493]],
        [111, [493, 494, 495, 496, 497, 498]],
        [112, [498, 499, 500, 501, 502]],
        [113, [502, 503, 504, 505, 506, 507, 508, 509, 510, 511]],
        [114, [511, 512, 513, 514, 515]],
        [115, [515, 516, 517, 518]],
        [116, [518, 519, 520, 521, 522, 523, 524]],
        [117, [524, 525, 526, 527, 528, 529, 530]],
        [118, [530, 531, 532]],
        [119, [532, 533, 534, 535]],
        [120, [535, 536, 537, 538, 539, 540]],
        [121, [540, 541]],
        [123, [541, 542]],
        [124, [542, 543, 544, 545, 546]],
        [126, [546, 547, 548]],
        [127, [548, 549, 550, 551]],
        [128, [551, 552, 553]],
        [129, [553, 554]],
        [130, [554, 555]],
        [132, [555, 556]],
        [134, [556, 557]],
        [137, [557, 558]],
    ]
    """
    
    visualize_map(map, file_path, file_path[:-4] + "_relable.tif")
