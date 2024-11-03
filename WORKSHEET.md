# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`YOUR GITHUB REPO HERE (or notice that you DMed us to share a private repo)`

## To move to your own repo:
First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

Module can be added to a sequential as it is an object, where as a functional will just be a function.

## -1.1 What is the difference between a Dataset and a DataLoader?

A Dataset holds the data while a DataLoader loads batches of data.

## -1.2 What does `@torch.no_grad()` above a function header do?

It doesn't calculate any gradients while doing tensor operations.



# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

build.py files contains the model builder. We have them to build the models and also the data loaders.

## 0.1 Where would you define a new model?

'models/directory'

## 0.2 How would you add support for a new dataset? What files would you need to change?

'data/dataset.py'

## 0.3 Where is the actual training code?

'main.py'

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

```
def evaluate()
def train_one_epoch()
def validate()

def main():
	get config
	get model
	get dataset
	build dataloader
	build optimizer
	for epoch in NUM_EPOCHS:
		train_one_epoch()
			validate()
		evaluate()
```


# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

It uses the config.py file to create the DataLoaders

### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

The three functions are `__init__`, `__len__` , and `__getitem__`.

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

The dataset attribute contains the data. You don't need to download it ahead of time.

### 1.1.1 What is `self.train`? What is `self.transform`?

self.train is a boolean value set to True by default. self.transform is a composition of various image transformations.

### 1.1.2 What does `__getitem__` do? What is `index`?

`__getitem__ ` returns the image and label at a particular index of the dataset. `index` is the position of the image within the dataset.

### 1.1.3 What does `__len__` do?

`__len__` returns the length of the dataset.

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

`self._get_transforms` returns the appropriate image transformation depending on if `self.train` is set to True or False. If it's True, then the transform adds noise to the images.

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

`transforms.Normalize` normalizes the channels of each tensor based on the inputted means and standard deviations.

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

`self.file` contains the data. The data is actually stored in the filepath "/data/medium-imagenet/medium-imagenet-nmep-96.hdf5". 

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

Since we're dealing with HDF5 datatype, we have to convert to torch tensors. Additionally. medium net has different normatmalization values and the order of operations is different.

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

Again, since we're working with the HDF5 datatype. accessing data is slightly different. Also, there are 3 data splits (train, val, and test) while CIFAR10 has 2. We don't assign labels for the test data.

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.

 ```
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
```
```
from .datasets import MediumImagenetHDF5Dataset
from torchvision.utils import make_grid


data = MediumImagenetHDF5Dataset(224, split="test", augment=False) 
data.visualize_sample(samples=10)

```


# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

lenet is implemented.

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

They inherit from `nn.module`. `__init__` and `forward` need to be implemented.

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

11 layers and 17 paramaters.



# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

lenet_base (lenet model, cifar10 dataset), resnet18_base (resnet18 model, cifar10 dataset), and resnet18_medium_imagenet (resnet18 model, medium imagenet dataset) are provided. 

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

* create train, val and train datasets and dataloaders
* set device
* move model to available device
* count parameters and complexity
* create epoch scheduler
* Training: run each epoch, validate model, update learning rate
* logging, checkpoint saving

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

`validate()` runs validation against the model using CrossEntropyLoss and returns the loss during training. `evaluate()` simply just returns the predictions of each input image.


# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)
With a common batch size of 32, we got the following performance.
LeNet Parameters: 99276
AlexNet Parameters: 57044810
LeNet Memory Usage: 5955 MiB
AlexNet Memory Usage: 6183 MiB


## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

Lenet Training: 37.186% validation: 38.72%
Lenet hyperparamters: batchsize = 32, optimizer = AdamW
AlexNet Training:  99.842% validation: 77.51%
AlexNet hyperparamters: batchsize = 32, optimizer = AdamW

# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

`PUSH YOUR CODE TO YOUR OWN GITHUB :)`

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

`YOUR ANSWER HERE`

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

`YOUR ANSWER HERE`

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

`YOUR ANSWER HERE`

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

`YOUR ANSWER HERE`

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

`YOUR ANSWER HERE`

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`YOUR ANSWER HERE`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/*`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

When using the default ResNet config, we ran into memory issues so we halved the batch size from 1024 to 512.
Best validation accuracy: 82.16%
Learning Ratef: 0.000293 
Training Time: 

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`YOUR ANSWER HERE`


# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot). The competition ends on __TBD__.

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉
