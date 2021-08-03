# CIFAR-10

Classify images from CIFAR-10 dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

<p align="center">
<img src="imgs/cifar-10.png" width="75%" height="75%"/>
</p>

## Instructions
1. Download CIFAR-10 python version from the [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html).
2. Extract cifar-10-python.tar.gz. 
3. Run python train.py.
4. Once training is over, run python view_predictions.py.

## Results

<p align="center">
Training Loss vs Epochs.
</p>
<p align="center">
<img src="imgs/train_loss.png" width="75%" height="75%"/>
</p>

<p align="center">
Test Accuracy vs Epochs. Achieved around 76% accuracy after nearly 75 epochs. These are expected results as we used a simple CNN architecture and also did not perform any data augmentation. 
</p>
<p align="center">
<img src="imgs/test_accuracy.png" width="75%" height="75%"/>
</p>

<p align="center">
Sample Prediction #1
</p>
<p align="center">
<img src="imgs/img1.png" width="50%" height="50%"/>
</p>
<p align="center">
<img src="imgs/pred1.png" width="75%" height="75%"/>
</p>

<p align="center">
Sample Prediction #2
</p>
<p align="center">
<img src="imgs/img2.png" width="50%" height="50%"/>
</p>
<p align="center">
<img src="imgs/pred2.png" width="75%" height="75%"/>
</p>

<p align="center">
Sample Prediction #3
</p>
<p align="center">
<img src="imgs/img3.png" width="50%" height="50%"/>
</p>
<p align="center">
<img src="imgs/pred3.png" width="75%" height="75%"/>
</p>