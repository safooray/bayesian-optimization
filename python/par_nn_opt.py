import numpy as np
from new_par_bo import optimize
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from six.moves import urllib

def fetch_mnist():
    mnist_path = "./mnist-original.mat"
    mnist_raw = loadmat(mnist_path)
    mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",}
    return mnist


def cost_func(params):
    hls = (int(params[1])) * int(params[0])
    max_iter = 5
    alpha = params[2]
    lr = params[3]
    mom = params[4]

    mlp = MLPClassifier(hidden_layer_sizes=hls, max_iter=max_iter, alpha=alpha,
                            solver='sgd', verbose=True, tol=1e-4, random_state=1,
                                                learning_rate_init=lr, momentum = mom)
    mnist = fetch_mnist()
    # rescale the data, use the traditional train/test split
    X, y = mnist['data'] / 255., mnist['target']
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    mlp.fit(X_train, y_train)
    #print("Training set score: %f" % mlp.score(X_train, y_train))
    #print("Test set score: %f" % mlp.score(X_test, y_test))
    return mlp.score(X_test, y_test)

def hyperpar_tune():
    bounds = np.array([[1,5.5],[10,100],[0,1],[0,1],[0,1]])
    xp, yp = optimize(n_iters=5, 
            sample_loss=cost_func,
            bounds=bounds,
            n_pre_samples=1,
            random_search = 100,
            ei_nsamples=1000,
            ei_force_estimate = True,
            ei_q=2)


if __name__ == "__main__":
    #print cost_func([1,50,1e-4,1e-2,.5])
    hyperpar_tune()
