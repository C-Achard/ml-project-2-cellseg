import math
import torch
import torch.nn as nn

class SoftNCutsLoss(nn.Module):
    """Implementation of a 3D Soft N-Cuts loss based on https://arxiv.org/abs/1711.08506 and https://ieeexplore.ieee.org/document/868688.
    
    Args:
        data_shape (H, W, D): shape of the images as a tuple. 
        o_i (scalar): weight for the distance of pixels in brightness.
        o_x (scalar): weight for the distance of pixels in space.
        radius (scalar): radius of pixels for which we compute the weights
    """
    def __init__(self, data_shape, o_i=..., o_x=..., radius=None):
        super(SoftNCutsLoss, self).__init__()
        self.o_i = o_i
        self.o_x = o_x
        self.radius = radius
        self.H = data_shape[0]
        self.W = data_shape[1]
        self.D = data_shape[2]

        if self.radius is None:
            self.radius = math.min(math.max(5, math.ceil(math.min(self.H, self.W, self.D) / 20)), self.H, self.W, self.D)

        # Precompute the spatial distance of the pixels for the weights calculation, to avoid recomputing it at each iteration
        H_index = torch.tensor(range(self.H)).expand(self.H, self.H) # (H, H)
        W_index = torch.tensor(range(self.W)).expand(self.W, self.W) # (W, W)
        D_index = torch.tensor(range(self.D)).expand(self.D, self.D) # (D, D)

        distances_H = torch.subtract(H_index, H_index.T) # (H, H)
        distances_W = torch.subtract(W_index, W_index.T) # (W, W)
        distances_D = torch.subtract(D_index, D_index.T) # (D, D)

        distances_H_expanded = distances_H.view(self.H, self.H, 1, 1, 1, 1).expand(self.H, self.H, self.W, self.W, self.D, self.D) # (H, H, W, W, D, D)
        distances_W_expanded = distances_W.view(1, 1, self.W, self.W, 1, 1).expand(self.H, self.H, self.W, self.W, self.D, self.D) # (H, H, W, W, D, D)
        distances_D_expanded = distances_D.view(1, 1, 1, 1, self.D, self.D).expand(self.H, self.H, self.W, self.W, self.D, self.D) # (H, H, W, W, D, D)

        squared_distances = torch.add(torch.add(torch.pow(distances_H_expanded, 2), torch.pow(distances_W_expanded, 2)), torch.pow(distances_D_expanded, 2)) # (H, H, W, W, D, D)

        squared_distances = squared_distances.swapaxes(1, 3).swapaxes(2, 4).swapaxes(1, 4) # (H, W, D, H, W, D)
        squared_distances = squared_distances.flatten(0, 2).flatten(1, 3) # (H*W*D, H*W*D)

        # Mask to only keep the weights for the pixels in the radius
        self.mask = torch.le(squared_distances, self.radius**2) # (H*W*D, H*W*D)

        W_X = torch.exp(torch.neg(torch.div(squared_distances, self.o_x))) # (H*W*D, H*W*D)

        self.W_X = torch.mul(W_X, self.mask) # (H*W*D, H*W*D)

    def forward(self, labels, inputs):
        """Forward pass of the Soft N-Cuts loss.
        
        Args:
            labels (torch.Tensor): Tensor of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
            inputs (torch.Tensor): Tensor of shape (N, C, H, W, D) containing the input images.

        Returns:
            The Soft N-Cuts loss of shape (N,).
        """
        N = inputs.shape[0]
        C = inputs.shape[1]
        K = labels.shape[1]

        losses = []
        weights = self.get_weights(inputs)  # (N, C, H*W*D, H*W*D)

        for k in range(K):
            Ak = labels[:, k, :, :, :] # (N, H, W, D)
            flatted_Ak = Ak.view(N, -1) # (N, H*W*D)

            # Compute the numerator of the Soft N-Cuts loss for k
            flatted_Ak_unsqueeze = flatted_Ak.unsqueeze(1) # (N, 1, H*W*D)
            transposed_Ak = torch.transpose(flatted_Ak_unsqueeze, 1, 2) # (N, H*W*D, 1)
            probs = torch.bmm(transposed_Ak, flatted_Ak_unsqueeze) # (N, H*W*D, H*W*D)
            probs_unsqueeze_expanded = probs.unsqueeze(1) # (N, 1, H*W*D, H*W*D)
            numerator_elements = torch.mul(probs_unsqueeze_expanded, weights) # (N, C, H*W*D, H*W*D)
            numerator = torch.sum(numerator_elements, dim=(2, 3)) # (N, C)

            # Compute the denominator of the Soft N-Cuts loss for k
            expanded_flatted_Ak = flatted_Ak.expand(-1, self.H*self.W*self.D) # (N, H*W*D, H*W*D)
            e_f_Ak_unsqueeze_expanded = expanded_flatted_Ak.unsqueeze(1) # (N, 1, H*W*D, H*W*D)
            denominator_elements = torch.mul(e_f_Ak_unsqueeze_expanded, weights) # (N, C, H*W*D, H*W*D)
            denominator = torch.sum(denominator_elements, dim=(2, 3)) # (N, C)

            # Compute the Soft N-Cuts loss for k
            division = torch.div(numerator, torch.add(denominator, 1e-8)) # (N, C)
            loss = torch.sum(division, dim=1) # (N,)
            losses.append(loss)

        loss = torch.sum(torch.stack(losses, dim=0), dim=0) # (N,)
        
        return torch.add(torch.neg(loss), K)

    def get_weights(self, inputs):
        """Computes the weights matrix for the Soft N-Cuts loss.
        
        Args:
            inputs (torch.Tensor): Tensor of shape (N, C, H, W, D) containing the input images.

        Returns:
            The weights matrix of shape (N, C, H*W*D, H*W*D).
        """

        # Compute the brightness distance of the pixels
        flatted_inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1) # (N, C, H*W*D)
        I_diff = torch.subtract(flatted_inputs.unsqueeze(3), flatted_inputs.unsqueeze(2)) # (N, C, H*W*D, H*W*D)
        masked_I_diff = torch.mul(I_diff, self.mask) # (N, C, H*W*D, H*W*D)
        squared_I_diff = torch.pow(masked_I_diff, 2) # (N, C, H*W*D, H*W*D)

        W_I = torch.exp(torch.neg(torch.div(squared_I_diff, self.o_i))) # (N, C, H*W*D, H*W*D)
        W_I = torch.mul(W_I, self.mask) # (N, C, H*W*D, H*W*D)

        # Get the spatial distance of the pixels
        unsqueezed_W_X = self.W_X.view(1, 1, self.W_X.shape[0], self.W_X.shape[1]) # (1, 1, H*W*D, H*W*D)

        W = torch.mul(W_I, unsqueezed_W_X) # (N, C, H*W*D, H*W*D)
        return W
