from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os

#%% distance method.
def sim_cos(u,v):
    
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    if len(ind) > 0:
        up = sum(u[ind] * v[ind])
        down = norm(u[ind]) * norm(v[ind])
        cos_sim = up/down
        if not math.isnan(cos_sim):
            return cos_sim
        else:
            return 0
    else:
        return 0
    
def sim_pcc(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>1:
        u_m = np.mean(u[ind])
        v_m = np.mean(v[ind])
        pcc = np.sum((u[ind]-u_m)*(v[ind]-v_m)) / (norm(u[ind]-u_m)*norm(v[ind]-v_m)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc
        else:
            return 0
    else:
        return 0

def sim_jac(u,v):
    ind1=np.where((1*(u==0)+1*(v==0))==0)[0]
    ind2=np.where((1*(u==0)+1*(v==0))!=2)[0]
    return len(ind1)/len(ind2)

def sim_jmsd(u,v):
    ind1=np.where((1*(u==0)+1*(v==0))==0)[0] # 교집합
    ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] # 합집합
    if len(ind1)>0:
        return (len(ind1)/len(ind2)) * (1-np.sum((u[ind1]/5-v[ind1]/5)**2)/len(ind1))
    else:
        return 0    