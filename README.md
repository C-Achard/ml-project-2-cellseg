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