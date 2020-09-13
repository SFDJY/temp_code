from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np
import cv2

n_class = 6


class SFDJY_DataSet(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.images = os.listdir(self.images_dir)
        self.labels = os.listdir(self.labels_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        label_index = self.labels[index]

        image_path = os.path.join(self.images_dir, image_index)
        label_path = os.path.join(self.labels_dir, label_index)

        image = np.array(cv2.imread(image_path))
        image = image / 127.5 - 1.0
        label = np.array(cv2.imread(label_path, 0))

        image = torch.from_numpy(image)

        image = image.float()

        return image, label


sfdjy_data = SFDJY_DataSet('./sfdjy_train_data/images/', './sfdjy_train_data/labels/')
sfdjy_data_loader = DataLoader(sfdjy_data, batch_size=64, shuffle=True)

sfdjy_test_data = SFDJY_DataSet('./sfdjy_test_data/images/', './sfdjy_test_data/labels/')
sfdjy_test_data_loader = DataLoader(sfdjy_test_data, batch_size=64, shuffle=True)
