from pathlib import Path
class Config:
    def __init__(self):
        # WNet
        self.in_channels = 1
        self.out_channels = 1
        self.num_classes = 2

        self.lr = 1e-4

        self.o_i = 10  # initialization as in the paper
        self.o_x = 4  # initialization as in the paper
        self.radius = None  # yields to a radius depending on the data shape

        self.num_epochs = 400
        self.batch_size = 1
        self.num_workers = 4

        # CRF
        self.sa = 50
        self.sb = 10
        self.sg = 1
        self.w1 = 50
        self.w2 = 10
        self.n_iter = 5

        # Data
        self.train_volume_directory = (
            # r"/tmp/pycharm_project_622/dataset/cropped_visual/train/volumes"
            r"../../dataset/cropped_visual/train/volumes"
        )

        self.do_augmentation = True
        self.parallel = False

        self.save_model = True
        self.save_model_path = r"./2_class/REP_wnet_2class.pth"
        self.save_losses_path = r"./2_class/REP_wnet_2class.pkl"
        self.save_every = 100
        self.weights_path = None


c = Config()
###############
# WANDB_CONFIG
###############
# WANDB_MODE = "disabled"
WANDB_MODE = "online"

WANDB_CONFIG = {
    # data setting
    "num_workers": c.num_workers,
    "do_augmentation": c.do_augmentation,
    "model_save_path": c.save_model_path,
    # train setting
    "batch_size": c.batch_size,
    "learning_rate": c.lr,
    "max_epochs": c.num_epochs,
    "save_every": c.save_every,
    # model
    "model_type": "wnet",
    "model_params": {
        "in_channels": c.in_channels,
        "out_channels": c.out_channels,
        "num_classes": c.num_classes,
    },
    # CRF
    "crf_params": {
        "sa": c.sa,
        "sb": c.sb,
        "sg": c.sg,
        "w1": c.w1,
        "w2": c.w2,
        "n_iter": c.n_iter,
    },
}
