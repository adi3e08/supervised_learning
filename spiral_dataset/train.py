import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Net(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Net,self).__init__()
        self.fc1=torch.nn.Linear(D_in,128)
        self.fc2=torch.nn.Linear(128,128)
        self.fc3=torch.nn.Linear(128,D_out)
 
    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y3 = self.fc3(y2)

        return y3

class SpiralDataset(torch.utils.data.Dataset):
    """Spiral dataset."""

    def __init__(self, path, train):
        dataset = np.load(path)
        if train:
            self.X, self.Y = dataset['X_train'], dataset['Y_train']
        else:
            self.X, self.Y = dataset['X_test'], dataset['Y_test']

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]

def main():

    expt_name = "expt_1"

    train_data = SpiralDataset("./data/dataset.npz",train=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=2)

    test_data = SpiralDataset("./data/dataset.npz",train=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=90,
                                              shuffle=True,
                                              num_workers=2)


    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    D_in, D_out = 2, 3 # input size, output size 

    model = Net(D_in, D_out).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

    max_epochs = 1000

    # Loop over epochs
    for epoch in range(max_epochs):

        print("\nEpoch : ", epoch)
        
        epoch_train_loss = 0.0
        total = 0

        # Training
        for X, Y in train_loader:
            
            # Transfer to device
            X, Y = X.to(torch.float).to(device), Y.to(torch.long).to(device)
            
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
            X, Y = X.to(torch.float).to(device), Y.to(torch.long).to(device)            
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
        if epoch % 100:
            torch.save({'model' : model.state_dict()}, os.path.join(model_dir, str(epoch)+".ckpt"))

    writer.close()


if __name__ == '__main__':
    main()