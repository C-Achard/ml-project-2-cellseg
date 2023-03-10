from pathlib import Path
class Config:
    def __init__(self):
        # WNet
        self.in_channels = 1
        self.out_channels = 1
        self.num_classes = 3

        self.lr = 1e-4

        self.o_i = 10  # initialization as in the paper
        self.o_x = 4  # initialization as in the paper
        self.radius = None  # yields to a radius depending on the data shape

        self.num_epochs = 1500
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
            r"/tmp/pycharm_project_622/dataset/cropped_visual/train/volumes"
        )
        self.do_augmentation = True
        self.parallel = False

        self.save_model = True
        self.save_model_path = r"./chkpt_res/test_wnet_checkpoint_4500e.pth"
        self.save_losses_path = r"./chkpt_res/checkpoint_loss_record_4500e.txt"
        self.save_every = 250
        self.weights_path = None
