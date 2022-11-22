#convert tif image to numpy array

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from tifffile import imwrite

#load the dataset

testing_im = "dataset/visual_tif/volumes/images.tif"
image = imread(testing_im)

labeled_im = "dataset/visual_tif/lab_sem/testing_im.tif"
labels = imread(labeled_im)

neurones = np.array(labels > 0)

#take all the pixels of the image that are not labeled as neurones
artefact= np.where(neurones == False, image, 0)


#save the artefact image
imwrite("dataset/visual_tif/artefact.tif", artefact)

