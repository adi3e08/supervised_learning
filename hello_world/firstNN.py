import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, H)
        self.fc2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = self.fc2(y1)

        return y2

def main():

    device = torch.device("cpu")

    N, D_in, H, D_out = 32, 4, 32, 2 # batch size, input size, hidden size, output size 

    # create random tensors to hold inputs and outputs
    X = torch.randn(N, D_in).to(device)
    Y = torch.randn(N, D_out).to(device)

    model = Net(D_in, H, D_out).to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    try:
        os.mkdir("./log")
    except :
        pass
    writer = SummaryWriter(log_dir="./log")
    
    for i in range(10000):

        # forward pass
        Y_pred = model(X)

        # compute and print loss
        loss = loss_fn(Y_pred, Y)
        # print(i, loss.item())
        writer.add_scalar('loss', loss.item(), i)

        # zero gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

    writer.close()

if __name__ == '__main__':
    main()