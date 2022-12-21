from monai.networks.nets import SwinUNETR
from torch import sigmoid
from torch import softmax

"""
SwinUNetR model from MONAI used for multi-class segmentation
"""


def get_weights_file():
    return ""


def get_net(img_size, use_checkpoint=True, out_channels=1):
    return SwinUNETR(
        img_size,
        in_channels=1,
        out_channels=out_channels,
        feature_size=48,
        use_checkpoint=use_checkpoint,
        # drop_rate=0.3,
        # spatial_dims=3,
    )


def get_output(model, input):
    out = model(input)
    # out = sigmoid(out)
    return out


def get_validation(model, val_inputs):
    return model(val_inputs)
