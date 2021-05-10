import sys
import matplotlib
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt



def load_data(dataset):
    acc_list = []
    acc_list.append(np.loadtxt(dataset+'-acc-sync.txt'))
    acc_list.append(np.loadtxt(dataset+'-acc-s=0.txt'))
    acc_list.append(np.loadtxt(dataset+'-acc-s=1.txt'))

    xys = []
    for i, accs in enumerate(acc_list):
        x = np.arange(len(accs)) + 1
        if len(accs) % 5 == 0:
            x = x[4::5]
            y = accs[4::5]
            xys.append([x, y])
        else:
            x = np.concatenate((x[4::5], x[-1:]))
            y = np.concatenate((accs[4::5], accs[-1:]))
            xys.append([x, y])

    return xys


def plot(dataset):
    tool = "Dorylus"
    plt.clf()
    labels = [tool+'-pipe', tool+'-async (s=0)', tool+'-async (s=1)']
    colors = ['o-', '^-', 'X-']

    acc_list = load_data(dataset)
    for i, xy in enumerate(acc_list):
        x, y = xy
        l = plt.plot(x, y, colors[i], label=labels[i])

    plt.xlabel('Epochs')
    plt.ylabel('Test Acc')
    plt.legend()
    filename = 'fig5-'+dataset+'.pdf'
    print("Plot " + filename)
    plt.savefig(filename)
    plt.clf()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 plot-fig5.py [reddit-small | amazon | reddit-large]")
        exit(-1)
    plot(sys.argv[1])


main()
