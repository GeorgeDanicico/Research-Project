import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from torchvision.datasets.folder import pil_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else
                      'cpu')


class LeafImageClassifierDataset(Dataset):
    def __init__(self, image_list, image_classes):
        self.images = []
        self.labels = []
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.image_size = 224
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.CenterCrop(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])
        for image, image_class in zip(image_list, image_classes):
            transformed_image = self.transforms(image)
            self.images.append(transformed_image)
            label = self.class_to_label[image_class]
            self.labels.append(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


def get_image_classifier_dataset_test():
    # TODO: import dataset
    return None


def get_image_classifier_dataset_train():
    # TODO: import dataset
    return None
