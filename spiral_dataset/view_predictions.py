import numpy as np
import torch
from train import Net, SpiralDataset
import matplotlib.pyplot as plt

def main():
    test_data = SpiralDataset("./data/dataset.npz",train=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=90,
                                              shuffle=True,
                                              num_workers=2)

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    D_in, D_out = 2, 3 # input size, output size 

    model = Net(D_in, D_out).to(device)
    path = "./log/expt_1/models/999.ckpt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    dataiter = iter(test_loader)
    X, Y = dataiter.next()

    X, Y = X.to(torch.float).to(device), Y.to(torch.long).to(device)            
    with torch.set_grad_enabled(False):
        Y_pred = model(X)
    _, predicted = torch.max(Y_pred, 1)


    X_test = X.cpu().numpy()
    Y_test = predicted.cpu().numpy()
    plot_X_test = [[],[],[]]
    for i in range(Y_test.shape[0]):
        plot_X_test[int(Y_test[i])].append(X_test[i])
    plot_X_test = [np.array(plot_X_test[0]), np.array(plot_X_test[1]), np.array(plot_X_test[2])]
    fig = plt.figure()
    plt.scatter(plot_X_test[0][:,0],plot_X_test[0][:,1],color='red')
    plt.scatter(plot_X_test[1][:,0],plot_X_test[1][:,1],color='green')
    plt.scatter(plot_X_test[2][:,0],plot_X_test[2][:,1],color='blue')
    # plt.show()
    fig.savefig('./data/predictions.png', dpi=fig.dpi)

if __name__ == '__main__':
    main()

