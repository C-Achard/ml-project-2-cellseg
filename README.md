# ml-project-2-cellseg
 Project 2 for ML_course : Cell Segmentation in 3D

Authors :
Cyril Achard
Yves Paychère
Colin Hoffman

## Important

This is based on previous code from the Mathis Lab of Adaptive Motor Control, made by Cyril Achard and Maxime Vidal.
Authors and updates are indicated in the files' header comment as precisely as possible.

## Installation procedure

From the Drive (**link is in the report**) :

- Copy the `dataset_clean` folder to obtain the training and validation datasets.
- Weights can be found the `weights_results` folder and can be used for inference. The `dataset_clean`folder is also needed for inference. 

We recommend using anaconda to install the required packages in an env. 
The following commands will create a new conda environment named `cellseg` and install required packages.

- `conda create --name cellseg python=3.8`
- `conda activate cellseg`
- Install pytorch according to the instructions on the website, with python 3.8 and using pip. Use cuda if relevant, it is strongly recommended even for inference.
- `pip install -r requirements.txt`

## Guided overview

In order to do obtain the result for the unsupervised model you should follow these steps. **Note that all the notebooks and python files for this guide are on models/wnet/**:

0. You should have downloaded the folder dataset_clean and added it in the main folder, You should download the file `weights_results/results_unsupervised/new_results_wnet_1500e.zip` from the Drive and extract its content into a folder that you should create `weights_results/results_unsupervised/`

1. If you want to do the inference (not recommended as it takes multiple Go of ram) You should run the notebook predict.ipynb and run the last cell for each image segmentated. To do so you should select the different layers 'Segmentation 1 image' and run the cell afterward. Then you can save the new layers directly from napari

1. You can directly download the results that are on the folder `results (output and statistics)/self supervised/` and put them on the folder `results/self supervised/`

2. To relabel the image used to evaluate the performance of the unsupervised model, you should run the script `relabel_images.py`. This will open a window showing the previously labeled neurons that were separated by the script. Once the script is finished, you can close the window and see the neurons that were previously missed. To view the image in 3D, you can click on the button with a square in the bottom left corner of the napari window, or press Ctrl+Y. You can change the view by clicking on the image and moving your mouse. On the left panel, you should have three layers: labels, potential neurons, and images. To see only the potential neurons on top of the image, you should click on the eye button of the labels layer. You will see two neurons. To find their values, you should click on the "potential neurons" layer and move your mouse over the neurons. The bottom left corner will show the label values, such as "potential neurons [21 19 9]: 13", which means that one of the label values you can add is 13. In the terminal where you run the code, you will see the prompt "Which labels do you want to add (0 to skip)? (separated by a comma):". To add the two forgotten neurons, you should enter "13,29" and press enter. Then, close the napari window and answer "n" to the two following questions. This will save the new labels.

3. To evaluate the model performances on this image you can run the notebook `evaluate_performances.ipynb`. The image used in the report is the 2nd one

4. To visualize the evaluation metric you can run the file `visualize_performances.py`. You can compare the classification of the labels between found, not found, artifacts and fused by hiding all the other layers (by clicking on the eye button) and comparing what the model found and what the ground truth is.

## Quality check and relabelling
All the scripts and notebooks that are used to label the images and perform a quality check of the labels are in the folder `label`.
To relabel the labels to ensure that they are unique, check their quality and add neurons missed you can use the function `relabel` in the file `relabel.py`, as shown in the main of this file.
To label new images with a basic watershed algorithm you can use the function `make_labels` of the file `make_artefact_labels`. To label all the artefacts of the dataset you can run the main of the file `make_artefact_labels.py`.

## Data augmentation
In order to add artefacts to the images labelled you can use the function `add_artefacts_to_folder` of the file `data_augmentation` in the folder `data_augmentation`, as shown in the notebook `data_augmentation.ipynb` in the same folder.

## Training models on axons and augmented data
You can use `train_multichannel.py` to re-train models to attempt to reproduce results. Use the dataset_clean folder on the Drive, place it in the repo path, and follow the instructions in the main of the file to reproduce results.

## Inference and evaluation
Use `infer_and_evaluate.py` to perform inference on validation data and evaluate the results. The weights are available in the `weights_results` folder on the Drive. The `dataset_clean` folder is also needed for inference.

## Unsupervised WNet model
The unsupervised WNet model is in the folder `models/wnet`. To train the model you can use `train_wnet.py` by either executing it or calling the method `train` from a notebook. 
To perform inference you can use the `PredictWNet` class from `predict.py` in a notebook. `predict.ipynb` is such a notebook and can also be used to visualize the results in napari. The weights of the model are available in the folder on the Drive. Use `config.py` to change the parameters of the model. For the time being, please don't use the CRF post-processing as it is doesn't work due to an implementation error from the library used.

## image processing
 If you want to obtain the results presented in the report for the image processing you should use the notebook `model_watershed.ipynb` that is in the main folder

## Libraries used
- MONAI
- PyTorch
- pydensecrf
- cython
- napari
- scikit-image
- tifffile
- Others (see requirements.txt)
