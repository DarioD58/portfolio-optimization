# Code Source:
# On 20151227 by MLdP <lopezdeprado@lbl.gov>
# Hierarchical Risk Parity
# This code is based on Lopez de Prado's paper: Building Diversified Portfolios that Outperform Out-of-Sample
# Used for comparison with my own implementation of HRP

import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import timeit



def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp


def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link: np.ndarray):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=pd.concat([sortIx,df0]) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1.0,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)//2), \
        (len(i)//2,len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist

def main(x: pd.DataFrame):
    cov,corr=x.cov(),x.corr()
    #3) cluster
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    #4) Capital allocation
    hrp=getRecBipart(cov,sortIx)


    return hrp

def evaluate_matrix_seriation(x: pd.DataFrame, repeat: int, number: int):
    corr = x.corr()
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    time_new = timeit.repeat(lambda: getQuasiDiag(link), repeat=repeat, number=number)
    return time_new

def evaluate_recursive_bisection(x: pd.DataFrame, repeat: int, number: int):
    cov = x.cov()
    corr = x.corr()
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    time_new = timeit.repeat(lambda: getRecBipart(cov, sortIx), repeat=repeat, number=number)

    return time_new


if __name__ == '__main__':
    # Sample data
    np.random.seed(0)
    x = pd.DataFrame(np.random.randn(10, 5), columns=list('ABCDE'))
    print(main(x))