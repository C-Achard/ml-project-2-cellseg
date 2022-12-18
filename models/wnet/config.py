class Config:
    def __init__(self):
        self.in_channels = 1
        self.out_channels = 1
        self.num_classes = 2

        self.lr = 1e-4

        self.o_i = 5        # find how to initialize this (learn it?)
        self.o_x = 5        # find how to initialize this (learn it?)
        self.radius = None
        
        self.sa = 5         # find how to initialize this (learn it?)
        self.sb = 5         # find how to initialize this (learn it?)
        self.sg = 5         # find how to initialize this (learn it?)
        self.n_iter = 5 

        self.num_epochs = 100
        self.batch_size = 1
        self.num_workers = 4

        self.train_volume_directory = ""
        self.do_augmentation = True