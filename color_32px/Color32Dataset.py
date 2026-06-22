from ImageDataset import ImageDataset

class Color32Dataset(ImageDataset):
    def __init__(self, data_dir_path):
        super().__init__(
            data_dir_path=data_dir_path,
            color_mode="RGB",
            target_size=(32,32),
            normalize=True
        )
