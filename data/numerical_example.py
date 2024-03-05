import numpy as np
import pandas as pd

def generate_data(nObs, size0, size1, sigma1):
    # Time series of correlated variables
    #1) generating some uncorrelated data
    np.random.seed(seed=12345)
    x = np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
    #2) creating correlation between the variables
    cols = [np.random.randint(0, size0-1) for _ in range(size1)]
    cols = [2,0,4,1,1]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs,len(cols)))
    x = np.append(x,y,axis=1)
    x = pd.DataFrame(x,columns=range(1,x.shape[1]+1))
    return x, cols