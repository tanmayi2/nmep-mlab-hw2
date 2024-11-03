from .datasets import MediumImagenetHDF5Dataset
from torchvision.utils import make_grid


data = MediumImagenetHDF5Dataset(224, split="test", augment=False) 
data.visualize_sample(samples=10)



