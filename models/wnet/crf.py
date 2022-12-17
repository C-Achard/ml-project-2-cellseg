import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def crf_batch(images, probs, sa, sb, sg, n_iter=5):
    """ CRF post-processing step for the W-Net, applied to a batch of images.

    Args:
        images (np.ndarray): Array of shape (N, C, H, W, D) containing the input images.
        probs (np.ndarray): Array of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
        sa (float): alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
        sb (float): beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
        sg (float): gamma standard deviation, the scale of the smoothness/gaussian kernel.

    Returns:
        np.ndarray: Array of shape (N, K, H, W, D) containing the refined class probabilities for each pixel.
    """
    
    return np.stack([crf(images[i], probs[i], sa, sb, sg, n_iter=n_iter) for i in range(images.shape[0])], axis=0)

def crf(image, prob, sa, sb, sg, n_iter=5):
    """ Implements the CRF post-processing step for the W-Net.
    Inspired by https://arxiv.org/abs/1210.5644, https://arxiv.org/abs/1606.00915 and https://arxiv.org/abs/1711.08506.
    Implemented using the pydensecrf library.

    Args:
        image (np.ndarray): Array of shape (C, H, W, D) containing the input image.
        prob (np.ndarray): Array of shape (K, H, W, D) containing the predicted class probabilities for each pixel.
        sa (float): alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
        sb (float): beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
        sg (float): gamma standard deviation, the scale of the smoothness/gaussian kernel.

    Returns:
        np.ndarray: Array of shape (K, H, W, D) containing the refined class probabilities for each pixel.
    """
    d = dcrf.DenseCRF(image.shape[1] * image.shape[2] * image.shape[3], prob.shape[0])

    # Get unary potentials from softmax probabilities
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)

    # Generate pairwise potentials
    featsGaussian = create_pairwise_gaussian(sdims=(sg, sg, sg), shape=image.shape[1:])
    featsBilateral = create_pairwise_bilateral(sdims=(sa, sa, sa), schan=[sb for i in range(image.shape[0])], img=image, chdim=image.shape[0])

    # Add pairwise potentials to the CRF
    compat = np.ones(prob.shape[0], dtype=np.float32) - np.diag([1 for i in range(prob.shape[0])], dtype=np.float32)
    d.addPairwiseEnergy(featsGaussian, compat=compat)
    d.addPairwiseEnergy(featsBilateral, compat=compat)

    # Run inference
    Q = d.inference(n_iter)

    return np.array(Q).reshape((prob.shape[0], image.shape[1], image.shape[2], image.shape[3]))