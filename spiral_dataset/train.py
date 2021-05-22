import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)
 
    def forward(self, x):
        h_relu=F.relu(self.linear1(x))
        y_pred=self.linear2(h_relu)
        return y_pred


def main():
    device = torch.device("cpu")

    N, D_in, H, D_out = 300, 2, 100, 3 # batch size, input size, hidden size, output size 

    # Generate Spiral Dataset
    X=np.zeros((N, D_in))
    Y=np.zeros(N)
    for k in range(3):
        for n in range(100):
            r = (n+1.0)/N
            t= np.pi/3.0+ k*2.0*np.pi/3.0-7*np.pi/(6.0 * 99.0)*n
            Y[(k*100+n)]=k
            X[(k*100+n),0]=r*np.cos(t)
            X[(k*100+n),1]=r*np.sin(t)
            
    # fig = plt.figure()
    # plt.scatter(X[:100,0],X[:100,1],color='red')
    # plt.scatter(X[100:200,0],X[100:200,1],color='green')
    # plt.scatter(X[200:300,0],X[200:300,1],color='blue')
    # plt.show()
    # #fig.savefig('temp.png', dpi=fig.dpi)

    X = torch.tensor(X, dtype=torch.float).to(device)
    Y = torch.tensor(Y, dtype=torch.long).to(device)

    model = Net(D_in, H, D_out).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    try:
        os.mkdir("./log")
    except :
        pass
    writer = SummaryWriter(log_dir="./log")
    
    for i in range(100000):

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