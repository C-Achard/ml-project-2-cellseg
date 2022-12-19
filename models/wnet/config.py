class Config:
    def __init__(self):
        self.in_channels = 1
        self.out_channels = 1
        self.num_classes = 2

        self.lr = 1e-4

        self.o_i = 10  # initialization as in the paper
        self.o_x = 4  # initialization as in the paper
        self.radius = None  # yields to a radius depending on the data shape

        self.sa = 5  # According to the paper, should be learnt via grid search. However we miss ground truth for that
        self.sb = 5  # According to the paper, should be learnt via grid search. However we miss ground truth for that
        self.sg = 1  # Initialization as in the paper
        self.w1 = 1  # According to the paper, should be learnt via grid search. However we miss ground truth for that
        self.w2 = 1  # Initialization as in the paper
        self.n_iter = 5

        self.num_epochs = 100
        self.batch_size = 1
        self.num_workers = 4

        self.train_volume_directory = ""
        self.do_augmentation = True

        self.save_model = True
        self.save_model_path = "models/wnet/wnet.pt"
