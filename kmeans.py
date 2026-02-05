#full implementation of kmeans using numpy
#Author Lefki Meidi


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


vers = {
    "numpy" : np.__version__,
    "python" : sys.version_info[:3],
    "matplot" : tuple(map(int,matplotlib.__version__.split(".")[:3]))
    }

def call_vers(d : dict):
    if isinstance(d,dict):
        for name,val in vers.items():
            if name == "python":
                if val > (3,10):
                    print(f"Python sys info correct {val} \n")
            else:
                print(f"Version of {name}:{val}\n")
        return True
    else:
        return False




#generating data

if __name__ == "__main__":
#    call_vers(vers)

    A = np.random.rand(15,33)
    B = np.random.rand(15,33)

    X = np.c_[A.ravel(),B.ravel()]

    print(X.shape)

    assert X[:,0].ndim == 1
    assert X[:,1].ndim == 1
    """
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=3, n_init=10).fit_predict(X)
    plt.figure(figsize=(5,12))
    plt.scatter(X[:,0],X[:,1],s=30,alpha=0.7,c="steelblue",marker="X")
    plt.title("I see")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    """
    from sklearn.preprocessing import StandardScaler
    Stand = StandardScaler()
    X_stand = Stand.fit_transform(X)
    



#Kmeans non-convexe problem




