from models.unet.model import UNet3D

"""
Pytorch port of TRAILMAP by Maxime Vidal
"""

def get_weights_file():
    # return "TMP_TEST_40e.pth"
    return "TRAILMAP_MS_best_metric_epoch_26.pth"


def get_net(out_channels=1):
    return UNet3D(1, out_channels=out_channels)


def get_output(model, input):
    out = model(input)

    return out


def get_validation(model, val_inputs):

    return model(val_inputs)
