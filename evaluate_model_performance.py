import numpy as np


def map_labels(labels, model_labels):
    """Map the model's labels to the neurons labels.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    Returns
    -------
    map_labels_existing: numpy array
        The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label correctly labelled
    map_fused_neurons: numpy array
        The neurones are considered fused if they are labelled by the same model's label, in this case we will return The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label that are in one of the fused neurones
    new_labels: list
        The labels of the model that are not labelled in the neurons, the ratio of the pixels of the model's label that are an artefact
    """
    map_labels_existing = []
    map_fused_neurons = []
    new_labels = []

    for i in np.unique(model_labels):
        if i == 0:
            continue
        indexes = labels[model_labels == i]
        # find the most common labels in the indexes
        unique, counts = np.unique(indexes, return_counts=True)
        tmp_map = []
        total_pixel_found = 0
        for ii in range(len(unique)):
            true_positive_ratio_model = counts[ii] / np.sum(counts)
            if unique[ii] == 0:
                if true_positive_ratio_model > 0.5:
                    new_labels.append([i, true_positive_ratio_model])
            else:
                ratio_pixel_found = counts[ii] / np.sum(labels == unique[ii])
                #print(i,unique[ii],ratio_pixel_found)
                if ratio_pixel_found > 0.5:
                    total_pixel_found += np.sum(counts[ii])
                    tmp_map.append(
                        [i, unique[ii], ratio_pixel_found, true_positive_ratio_model]
                    )
        if len(tmp_map) == 1:
            map_labels_existing.append(tmp_map[0])
        elif len(tmp_map) > 1:
            for ii in range(len(tmp_map)):
                tmp_map[ii][3] = total_pixel_found / np.sum(counts)
            map_fused_neurons += tmp_map
    return map_labels_existing, map_fused_neurons, new_labels


def evaluate_model_performance(labels, model_labels, do_print=True):
    """Evaluate the model performance.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    do_print : bool
        If True, print the results.
    Returns
    -------
    neuron_found : float
        The number of neurons found by the model
    neuron_fused: float
        The number of neurons fused by the model
    neuron_not_found: float
        The number of neurons not found by the model
    neuron_artefact: float
        The number of artefact that the model wrongly labelled as neurons
    mean_true_positive_ratio_model: float
        The mean (over the model's labels that correspond to one true label) of (correctly labelled pixels)/(total number of pixels of the model's label)
    mean_ratio_pixel_found: float
        The mean (over the model's labels that correspond to one true label) of (correctly labelled pixels)/(total number of pixels of the true label)
    mean_ratio_pixel_found_fused: float
        The mean (over the model's labels that correspond to multiple true label) of (correctly labelled pixels)/(total number of pixels of the true label)
    mean_true_positive_ratio_model_fused: float
        The mean (over the model's labels that correspond to multiple true label) of (correctly labelled pixels in any fused neurons of this model's label)/(total number of pixels of the model's label)
    mean_ratio_false_pixel_artefact: float
        The mean (over the model's labels that are not labelled in the neurons) of (wrongly labelled pixels)/(total number of pixels of the model's label)
    """
    map_labels_existing, map_fused_neurons, new_labels = map_labels(
        labels, model_labels
    )

    # calculate the number of neurons individually found
    neurons_found = len(map_labels_existing)
    # calculate the number of neurons fused
    neurons_fused = len(map_fused_neurons)
    # calculate the number of neurons not found
    neurons_not_found = len(np.unique(labels)) - 1 - neurons_found - neurons_fused
    # artefacts found
    artefacts_found = len(new_labels)
    if len(map_labels_existing) > 0:
        # calculate the mean true positive ratio of the model
        mean_true_positive_ratio_model = np.mean([i[3] for i in map_labels_existing])
        # calculate the mean ratio of the neurons pixels correctly labelled
        mean_ratio_pixel_found = np.mean([i[2] for i in map_labels_existing])
    else:
        mean_true_positive_ratio_model = np.nan
        mean_ratio_pixel_found = np.nan

    if len(map_fused_neurons) > 0:
        # calculate the mean ratio of the neurons pixels correctly labelled for the fused neurons
        mean_ratio_pixel_found_fused = np.mean([i[2] for i in map_fused_neurons])
        # calculate the mean true positive ratio of the model for the fused neurons
        mean_true_positive_ratio_model_fused = np.mean(
            [i[3] for i in map_fused_neurons]
        )
    else:
        mean_ratio_pixel_found_fused = np.nan
        mean_true_positive_ratio_model_fused = np.nan

    # calculate the mean false positive ratio of each artefact
    if len(new_labels) > 0:
         mean_ratio_false_pixel_artefact= np.mean([i[1] for i in new_labels])
    else:
         mean_ratio_false_pixel_artefact= np.nan
    if do_print:
        print("Neurons found: ", neurons_found)
        print("Neurons fused: ", neurons_fused)
        print("Neurons not found: ", neurons_not_found)
        print("Artefacts found: ", artefacts_found)
        print("Mean true positive ratio of the model: ", mean_true_positive_ratio_model)
        print(
            "Mean ratio of the neurons pixels correctly labelled: ",
            mean_ratio_pixel_found,
        )
        print(
            "Mean ratio of the neurons pixels correctly labelled for fused neurons: ",
            mean_ratio_pixel_found_fused,
        )
        print(
            "Mean true positive ratio of the model for fused neurons: ",
            mean_true_positive_ratio_model_fused,
        )
        print(
            "Mean ratio of false pixel in artefacts: ", mean_ratio_false_pixel_artefact
        )

    return (
        neurons_found,
        neurons_fused,
        neurons_not_found,
        artefacts_found,
        mean_true_positive_ratio_model,
        mean_ratio_pixel_found,
        mean_ratio_pixel_found_fused,
        mean_true_positive_ratio_model_fused,
        mean_ratio_false_pixel_artefact,
    )


if __name__ == "__main__":
    #Example of how to use the functions in this module.
    a = np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]])

    b = np.array([[5, 5, 0, 0], 
                  [5, 5, 2, 0], 
                  [0, 2, 2, 0], 
                  [0, 0, 2, 0]])
    evaluate_model_performance(a, b)

    c = np.array([[2, 2, 0, 0],
                  [2, 2, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]])

    d = np.array([[4, 0, 4, 0], 
                  [4, 4, 4, 0], 
                  [0, 4, 4, 0], 
                  [0, 0, 4, 0]])

    evaluate_model_performance(c, d)
    """
    from tifffile import imread
    labels=imread("dataset/visual_tif/labels/testing_im.tif")
    labels_model=imread("dataset/visual_tif/artefact_neurones/basic_model.tif")
    evaluate_model_performance(labels, labels_model)
    """