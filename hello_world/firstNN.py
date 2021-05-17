import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, H)
        self.fc2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def main():

    device = torch.device("cpu")

    N, D_in, H, D_out = 32, 4, 32, 2 # batch size, input size, hidden size, output size 

    # create random tensors to hold inputs and outputs
    x = torch.randn(N, D_in).to(device)
    y = torch.randn(N, D_out).to(device)

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
        y_pred = model(x)

        # compute and print loss
        loss = loss_fn(y_pred, y)
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