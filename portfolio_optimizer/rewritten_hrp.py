import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_ = cov[cItems][:, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = [link[-1, 0], link[-1, 1]]
    numItems = link[-1, 3]  # number of original items
    while max(sortIx) >= numItems:
        sortIx = [sortIx[i] for i in range(len(sortIx)) if sortIx[i] >= numItems]  # find clusters
        i = [sortIx[j] for j in range(len(sortIx))]
        j = [sortIx[j] - numItems for j in range(len(sortIx))]
        sortIx = [link[j, 0] for j in i]  # item 1
        df0 = [link[j, 1] for j in i]
        sortIx += df0  # item 2
        sortIx.sort()  # re-sort
        sortIx = [sortIx[i] for i in range(len(sortIx))]  # re-index
    return sortIx


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = np.ones(len(sortIx))
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [cItems[i][j:k] for i in range(len(cItems)) for j, k in ((0, len(cItems[i]) // 2),
                                                                         (len(cItems[i]) // 2, len(cItems[i]))) if
                  len(cItems[i]) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    return dist


def main(x: pd.DataFrame):
    cov, corr = x.cov().to_numpy(), x.corr().to_numpy()
    # 3) cluster
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = [corr.index[i] for i in sortIx]  # recover labels
    # 4) Capital allocation
    hrp = getRecBipart(cov, sortIx)

    # My addition
    final_result = hrp.tolist()
    return dict(sorted(zip(sortIx, final_result)))