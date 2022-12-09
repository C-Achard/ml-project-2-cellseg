import torch
import torch.nn as nn

class SoftNCutsLoss(nn.Module):
    """Implementation of a 3D Soft N-Cuts loss based on the 2D version from https://arxiv.org/abs/1711.08506."""
    def __init__(self):
        super(SoftNCutsLoss, self).__init__()

    def forward(self, labels, inputs):
        """Forward pass of the Soft N-Cuts loss.
        
        Args:
            labels (torch.Tensor): Tensor of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
            inputs (torch.Tensor): Tensor of shape (N, C, H, W, D) containing the input images.

        Returns:
            The Soft N-Cuts loss of shape (N,).
        """
        num_classes = labels.shape[1]

        losses = []
        weights = self.get_weights(inputs) # (N, H*W*D, H*W*D)

        for k in range(num_classes):
            Ak = labels[:, k, :, :, :] # (N, H, W, D)
            flatted_Ak = Ak.view(Ak.shape[0], -1) # (N, H*W*D)
            flatted_Ak_unsqueeze = flatted_Ak.unsqueeze(1) # (N, 1, H*W*D)
            transposed_Ak = torch.transpose(flatted_Ak_unsqueeze, 1, 2) # (N, H*W*D, 1)
            probs = torch.bmm(transposed_Ak, flatted_Ak_unsqueeze) # (N, H*W*D, H*W*D)
            numerator_elements = torch.mul(probs, weights) # (N, H*W*D, H*W*D)
            numerator = torch.sum(numerator_elements, dim=(1, 2)) # (N,)

            expanded_flatted_Ak = flatted_Ak.expand(-1, flatted_Ak.shape[2]) # (N, H*W*D, H*W*D)
            denominator_elements = torch.mul(expanded_flatted_Ak, weights) # (N, H*W*D, H*W*D)
            denominator = torch.sum(denominator_elements, dim=(1, 2)) # (N,)

            division = torch.div(numerator, torch.add(denominator, 1e-8)) # (N,)
            losses.append(division)

        loss = torch.sum(torch.stack(losses, dim=0), dim=0) # (N,)
        
        return torch.add(torch.neg(loss), num_classes)

    def get_weights(self, inputs):
        """Computes the weights matrix for the Soft N-Cuts loss.
        
        Args:
            inputs (torch.Tensor): Tensor of shape (N, C, H, W, D) containing the input images.

        Returns:
            The weights matrix of shape (N, H*W*D, H*W*D).
        """
        return ...
