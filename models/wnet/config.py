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
        self.sa = 5  # According to the paper, should be learnt via grid search. However we miss ground truth for that
        self.sb = 5  # According to the paper, should be learnt via grid search. However we miss ground truth for that
        self.sg = 1  # Initialization as in the paper
        self.w1 = 1  # According to the paper, should be learnt via grid search. However we miss ground truth for that
        self.w2 = 1  # Initialization as in the paper
        self.n_iter = 5

        # Data
        self.train_volume_directory = (
            r"/tmp/pycharm_project_622/dataset/cropped_visual/train/volumes"
        )
        self.do_augmentation = True
        self.parralel = False

        self.save_model = True
        self.save_model_path = r"./test_wnet_1500e.pth"
        self.save_losses_path = r"./1500e_loss_record.txt"
        self.save_every = 250
