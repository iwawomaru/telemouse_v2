import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

from scipy.fftpack import fft, fftfreq

random.seed(0)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def t_sigmoid(y):
    return np.log(y / (1. - y))

class ESN(object):
    def __init__(self, n_in, n_resv, n_out, 
                 r_con_inW=0.03, r_con_W=0.1, r_con_recW=0.8):

        self.n_in = n_in
        self.n_out = n_out
        self.n_resv = n_resv

        self.inW = np.zeros((n_in, n_resv))
        self.W = np.zeros((n_resv, n_resv))
        self.outW = np.zeros((n_resv, n_out))

        self.resv = np.zeros(n_resv)

        for x1 in xrange(len(self.inW)):
            for x2 in xrange(len(self.inW[0])):
                if random.random() < r_con_inW:
                    self.inW[x1][x2] = (random.random() - 0.5)

        for x1 in xrange(len(self.W)):
            for x2 in xrange(len(self.W[0])):
                if random.random() < r_con_W:
                    self.W[x1][x2] = (random.random() - 0.5)

        self.W = np.r_[self.inW, self.W]

    def prop(self, data):
        """ take only one data, not sequence """
        data = np.atleast_1d(data)
        inp = np.r_[data, self.resv]
        self.resv = sigmoid(np.dot(inp, self.W))
        # self.resv = np.tanh(np.dot(inp, self.W))
        out = np.dot(np.atleast_2d(self.resv), self.outW)
        out = sigmoid(out)
        return self.resv, out

    def prop_sequence(self, datas):
        resvs = []
        outs = []
        for data in datas:
            out = self.prop(data)
            resvs.append(out[0])
            outs.append(out[1])

        return np.array(resvs), np.array(outs)

    def train(self, indata, outdata):
        sec_resv = []
        for ind in indata:
            sec_resv.append(self.prop(ind)[0])

        sec_resv = np.array(sec_resv)
        self.outW = np.dot(np.linalg.pinv(sec_resv), t_sigmoid(outdata))

if __name__ == "__main__":

    pos_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    dataset = []
    dataset.append(np.load("npy/osawa_meros.npy"))
    dataset.append(np.load("npy/bayashi_meros.npy"))
    dataset.append(np.load("npy/osawa_meros.npy"))
    dataset.append(np.load("npy/bayashi_meros.npy"))

    dataset.append(np.load("npy/osawa_breathe_left.npy"))
    dataset.append(np.load("npy/bayashi_breathe_left.npy"))
    dataset.append(np.load("npy/osawa_breathe_right.npy"))
    dataset.append(np.load("npy/bayashi_breathe_right.npy"))
    dataset.append(np.load("npy/osawa_pick_left.npy"))
    dataset.append(np.load("npy/bayashi_pick_left.npy"))
    dataset.append(np.load("npy/osawa_pick_right.npy"))
    dataset.append(np.load("npy/bayashi_pick_right.npy"))
    dataset.append(np.load("npy/osawa_loud_left.npy"))
    dataset.append(np.load("npy/bayashi_loud_left.npy"))
    dataset.append(np.load("npy/osawa_loud_right.npy"))
    dataset.append(np.load("npy/bayashi_loud_right.npy"))

    dataset.append(np.load("npy/osawa_meros.npy"))
    dataset.append(np.load("npy/bayashi_meros.npy"))
    dataset.append(np.load("npy/osawa_meros.npy"))
    dataset.append(np.load("npy/bayashi_meros.npy"))

    din = []
    dout = []

    for i, dset in enumerate(dataset):
        tin, tout = np.hsplit(dset, 2)
        din.extend(tin)
        if i not in pos_list:
            dout.extend(np.zeros_like(tout))
        else:
            dout.extend(np.ones_like(tout))

    N = 256
    din = np.array([d[0][:N:1] for d in din], np.float32)/255.
    yf = fft(din)[:,:(N/2):4].real/(N/2)
    print yf.shape

    dout = np.array([d[0] for d in dout], dtype=np.float32) * 0.9 + 0.05

    esn = ESN(yf.shape[1], 500, 1)

    esn.train(yf, dout)
    out = esn.prop_sequence(yf)[1]
    plt.plot(out, linewidth=0, marker="o", markersize=0.5, markeredgewidth=0)
    # plt.plot(out)
    plt.plot(dout)
    plt.show()

    with open("esn_breathe_left.pickle", mode="wb") as f:
        pickle.dump(esn, f)
