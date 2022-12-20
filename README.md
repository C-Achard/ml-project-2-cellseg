# ml-project-2-cellseg
 Project 2 for ML_course : Cell Segmentation in 3D

Authors :
Cyril Achard
Yves Paychère
Colin Hoffman

## Important

This is based on previous code from the Mathis Lab of Adaptive Motor Control, made by Cyril Achard and Maxime Vidal.
Authors and updates are indicated in the files' header comment.

## Installation procedure
We recommend using conda to install the required packages. The following commands will create a new conda environment named `cellseg` and install required packages.


- `conda create --name cellseg python=3.8`
- `conda activate cellseg`
- Install pytorch according to the instructions on the website, with python 3.8 and using pip. Use cuda if relevant, it is strongly recommended.
- `pip install -r requirements.txt`
- For visualization and labeling/evaluation : `pip install napari[all]`

## Quality check and relabelling
All the scripts and notebooks that are used to label the images and perform a quality check of the labels are in the folder label. To relabel the labels to ensure that they are unique, check their quality and add neurons missed you can use the function relabel in the file relabel.py, as shown in the main of this file. To label new images, with a basic watershed algorithm you can use the function make_labels of the file make_artefact_labels. To label all the artefacts of the dataset you can run the main of the file make_artefact_labels.py.

## Data augmentation
In order to add artefacts to the images labelled you can use the function add_artefacts_to_folder of the file data_augmentation in the folder data_augmentation, as shown in the notebook data_augmentation.ipynb in the same folder.

