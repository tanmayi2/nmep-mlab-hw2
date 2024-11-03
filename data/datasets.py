from __future__ import annotations

import h5py
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


# from torchvision.transforms import Compose, Normalize, PILToTensor, Resize


# deprecated in favor of MediumImagenetHDF5Dataset
# class MediumImagenetDataset(Dataset):
#     def __init__(self, img_size, split:str='train', augment=True):
#         assert split in ['train', 'val', 'test']
#         self.split = split
#         self.augment = augment
#         self.input_size = img_size
#         self.transform = self._get_transforms()
#         ds = ImageFolder("/data/medium-imagenet/data")
#         if split == 'train':
#             self.dataset = Subset(ds, range(0, len(ds) * 9 // 10))
#         else:
#             self.dataset = Subset(ds, range(len(ds) * 9 // 10, len(ds)))

#     def __getitem__(self, index):
#         image, label = self.dataset[index]
#         image = self.transform(image)
#         return image, label

#     def __len__(self):
#         return len(self.dataset)

#     def _get_transforms(self):
#         transform = []
#         transform.append(transforms.PILToTensor())
#         transform.append(lambda x: x.to(torch.float))
#         normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
#         transform.append(transforms.Normalize(normalization[0], normalization[1]))
#         if self.train and self.augment:
#             transform.extend(
#                 [
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                 ]
#             )
#         transform.append(transforms.Resize([self.input_size] * 2))
#         return transforms.Compose(transform)


class MediumImagenetHDF5Dataset(Dataset):
    def __init__(
        self,
        img_size,
        split: str = "train",
        filepath: str = "/data/medium-imagenet/medium-imagenet-nmep-96.hdf5",
        augment: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.augment = augment
        self.input_size = img_size
        self.transform = self._get_transforms()
        self.file = h5py.File(filepath, "r")

    def __getitem__(self, index):
        image = self.file[f"images-{self.split}"][index]
        if self.split != "test":
            label = self.file[f"labels-{self.split}"][index]
        else:
            label = -1
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.file[f"images-{self.split}"])

    def _get_transforms(self):
        transform = []
        transform.append(lambda x: torch.tensor(x / 256).to(torch.float))
        normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform.append(transforms.Normalize(normalization[0], normalization[1]))
        transform.append(transforms.Resize([self.input_size] * 2))
        if self.split == "train" and self.augment:
            transform.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ]
            )
        return transforms.Compose(transform)

    def visualize_samples(self, samples=10):
        images, labels = [], []
        for index in range(0, samples):
            raw_image = self.file[f"images-{self.split}"][index]
            #image = self.transform(raw_image),dont want to do transform
            image = raw_image
            if self.split != "test":
                label = self.file[f"labels-{self.split}"][index]
            else:
                label = -1
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        # Denormalize images?????
        #mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        #images = images * std + mean
        grid = make_grid(images, nrow=5, padding=2)
        plt.figure(figsize=(15, 8))
        plt.imshow(grid.permute(1, 2, 0).clip(0, 1))
        # # Add labels
        # label_names = [self.class_names[str(int(label))] if label != -1 else "Unknown"
        #               for label in labels]
        # plt.title("Labels: " + ", ".join(label_names), fontsize=8)
        plt.axis('off')
        plt.show()


class CIFAR10Dataset(Dataset):
    def __init__(self, img_size=32, train=True):
        self.train = train
        self.img_size = img_size

        self.transform = self._get_transforms()
        self.dataset = CIFAR10(root="/data/cifar10", train=self.train, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self):
        if self.train:
            transform = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Resize([self.img_size] * 2),
            ]
        else:
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Resize([self.img_size] * 2),
            ]
        return transforms.Compose(transform)
