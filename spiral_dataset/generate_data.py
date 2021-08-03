import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():

    try:
        os.mkdir("./data")
    except :
        pass

    N = 300 

    # Generate Spiral Dataset
    X=np.zeros((N, 2))
    Y=np.zeros(N)
    for k in range(3):
        for n in range(100):
            r = (n+1.0)/N
            t= np.pi/3.0+ k*2.0*np.pi/3.0-7*np.pi/(6.0 * 99.0)*n
            Y[(k*100+n)]=k
            X[(k*100+n),0]=r*np.cos(t)
            X[(k*100+n),1]=r*np.sin(t)
            
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    np.savez("./data/dataset.npz",X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

    plot_X_train = [[],[],[]]
    for i in range(Y_train.shape[0]):
        plot_X_train[int(Y_train[i])].append(X_train[i])
    plot_X_train = [np.array(plot_X_train[0]), np.array(plot_X_train[1]), np.array(plot_X_train[2])]
    fig = plt.figure()
    plt.scatter(plot_X_train[0][:,0],plot_X_train[0][:,1],color='red')
    plt.scatter(plot_X_train[1][:,0],plot_X_train[1][:,1],color='green')
    plt.scatter(plot_X_train[2][:,0],plot_X_train[2][:,1],color='blue')
    # plt.show()
    fig.savefig('./data/train_data.png', dpi=fig.dpi)

if __name__ == '__main__':
    main()