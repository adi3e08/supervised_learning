import os
import numpy as np
import torch
import torchvision
from train import Net
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():

    transform = torchvision.transforms.Compose([
        # you can add other transformations in this list
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_data = torchvision.datasets.CIFAR10("cifar-10-python",train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    use_cuda = False
    # torch.device object used throughout this script
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = Net().to(device)
    path = "./log/expt_1/models/74.ckpt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    dataiter = iter(test_loader)
    X, Y = dataiter.next()

    X, Y = X.to(device), Y.to(device)            
    with torch.set_grad_enabled(False):
        Y_pred = model(X)
    _, predicted = torch.max(Y_pred, 1)

    # print images
    print('GroundTruth: ', [classes[Y[j]] for j in range(4)])
    print('Predicted: ', [classes[predicted[j]] for j in range(4)])
    imshow(torchvision.utils.make_grid([X[j] for j in range(4)]))


if __name__ == '__main__':
    main()
