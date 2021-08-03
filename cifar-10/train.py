import os
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 10)

    def forward(self, x):
        y1 = self.pool(F.relu(self.conv1(x)))
        y2 = self.pool(F.relu(self.conv2(y1)))
        y2 = y2.view(-1, 4096)
        y3 = F.relu(self.fc1(y2))
        y = self.fc2(y3)

        return y

def main():

    expt_name = "expt_1"

    transform = torchvision.transforms.Compose([
        # you can add other transformations in this list
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10("cifar-10-python",train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=2)

    test_data = torchvision.datasets.CIFAR10("cifar-10-python",train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=2)

    use_cuda = False
    # torch.device object used throughout this script
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = Net().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    try:
        os.mkdir("./log")
    except :
        pass
    exp_dir = os.path.join("./log", expt_name)
    model_dir = os.path.join(exp_dir, "models")
    tensorboard_dir = os.path.join(exp_dir, "tensorboard")
    os.mkdir(exp_dir)
    os.mkdir(model_dir)
    os.mkdir(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    max_epochs = 100

    # Loop over epochs
    for epoch in range(max_epochs):

        print("\nEpoch : ", epoch)
        
        epoch_train_loss = 0.0
        total = 0

        # Training
        for X, Y in train_loader:
            
            # Transfer to device
            X, Y = X.to(device), Y.to(device)
            
            Y_pred = model(X)

            # Compute and print loss.
            loss = loss_fn(Y_pred, Y)
            epoch_train_loss += (loss.item()*Y.size(0))
            total += Y.size(0)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        epoch_train_loss /= total
        print("Train Loss : ",epoch_train_loss)
        writer.add_scalar('train_loss', epoch_train_loss, epoch)


        # Testing
        epoch_test_loss = 0.0
        correct = 0
        total = 0

        for X, Y in test_loader:
            # Transfer to device
            X, Y = X.to(device), Y.to(device)            
            with torch.set_grad_enabled(False):
                Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            epoch_test_loss += (loss.item()*Y.size(0))
            total += Y.size(0)
            _, predicted = torch.max(Y_pred, 1)
            correct += (predicted == Y).sum().item()

        epoch_test_loss /= total
        print("Test Loss : ",epoch_test_loss)
        writer.add_scalar('test_loss', epoch_test_loss, epoch)
        accuracy = 100 * correct / total 
        print("Test Accuracy : ",accuracy)
        writer.add_scalar('test_accuracy', accuracy, epoch)

        #Save checkpoint
        torch.save({'model' : model.state_dict()}, os.path.join(model_dir, str(epoch)+".ckpt"))

    writer.close()


if __name__ == '__main__':
    main()    