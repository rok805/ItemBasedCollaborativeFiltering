#%%
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
from similarity_measure import *


#%%
# 데이터 불러오기 및 rating, item 데이터 전처리.
data_name = 'MovieLens100K'
cwd=os.getcwd()

if data_name == 'MovieLens100K': # MovieLens100K load and preprocessing
    # MovieLens100K: u.data, item.txt 의 경로
    data = pd.read_table(os.path.join(cwd,'movielens\\order\\u.data'),header=None, names=['uid','iid','r','ts'])
    item = pd.read_table(os.path.join(cwd,'movielens\\order\\item.txt'), sep='|', header=None)
    item_cols=['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    item.columns = item_cols
    item=np.array(item.drop(columns=['movie id','movie title', 'release date', 'video release date', 'IMDb URL', 'unknown']))
    # uid, iid minus one. 평점 행렬의 행, 열 지정을 위해서 -1 을 함.
    data['uid'] = np.array(data.uid) - 1
    data['iid'] = np.array(data.iid) - 1

elif data_name == 'MovieLens1M': # MovieLens1M load and preprocessing
    # MovieLens1M: rating.dat, movies.dat 의 경로
    data = pd.read_csv(os.path.join(cwd,'movielens\\1M\\ratings.dat'), sep='::', header=None, names=['uid','iid','r','ts']).drop(columns=['ts'])
    item = pd.read_table(os.path.join(cwd, 'movielens\\1M\\movies.dat'), sep='::', header=None)
    
    # item indexing
    m_d = {}
    for n, i in enumerate(item[0]):
        m_d[i] = n
    item[0] = sorted(m_d.values())
    
    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n
    
    # uid minus one. 평점 행렬의 행, 열 지정을 위해서 -1 을 함.
    data['uid'] = np.array(data.uid) - 1
    
    # movie genre matrix
    genre_name = set()
    for i in range(item.shape[0]):
        gg = item.loc[i,2].split('|')
        for j in gg:
            genre_name.add(j)
    
    m_to_idx = {}
    for n, i in enumerate(genre_name):
        m_to_idx[i] = n
    i_to_g = {}
    for i in range(item.shape[0]):
        i_to_g[i] = [m_to_idx[g] for g in item.loc[i,2].split("|")]
    i_genre = np.zeros([item.shape[0], len(genre_name)])
    for i in i_to_g:
        for j in i_to_g[i]:
            i_genre[i,j] = 1
    item = i_genre

elif data_name == 'Netflix': # Netflix load and preprocessing
    # Netflix: ratings.csv, movies.csv 의 경로
    data = pd.read_csv(os.path.join(cwd,'netflix\\movie_ratings\\ratings.csv'), header=0, names=['uid','iid','r','ts']).drop(columns=['ts'])
    item = pd.read_table(os.path.join(cwd,'netflix\\movie_ratings\\movies.csv'), sep=',', header=0)
    item.columns=[0,1,2]
    # item indexing
    m_d = {}
    for n, i in enumerate(item[0]):
        m_d[i] = n
    item[0] = sorted(m_d.values())
    
    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n
    
    # uid minus one. 평점 행렬의 행, 열 지정을 위해서 -1 을 함.
    data['uid'] = np.array(data.uid) - 1
    
    # movie genre matrix
    genre_name = set()
    for i in range(item.shape[0]):
        gg = item.loc[i,2].split('|')
        for j in gg:
            genre_name.add(j)
    
    m_to_idx = {}
    for n, i in enumerate(genre_name):
        m_to_idx[i] = n
    i_to_g = {}
    for i in range(item.shape[0]):
        i_to_g[i] = [m_to_idx[g] for g in item.loc[i,2].split("|")]
    i_genre = np.zeros([item.shape[0], len(genre_name)])
    for i in i_to_g:
        for j in i_to_g[i]:
            i_genre[i,j] = 1
    item = i_genre



#%%
# Collaborative Filtering
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

# cv validation, random state, split setting.
cv = 5
rs = 35
sk = StratifiedKFold(n_splits=cv, random_state=rs, shuffle=True)

# 결과저장 데이터프레임
result_mae_rmse = pd.DataFrame(columns=['fold','k','MAE','RMSE'])
result_topN = pd.DataFrame(columns=['fold','k','Precision','Recall','F1_score'])
count = 0
sim_name = 'Asim'
for f, (trn,val) in enumerate(sk.split(data,data['uid'].values)):
    print()
    print(f'cv: {f+1}')
    trn_data = data.iloc[trn]
    val_data = data.iloc[val]


##########################################################################################
##########################################################################################
##########################################################################################

#%%
    # train dataset rating dictionary.
    data_d_trn_data = {}
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        if i not in data_d_trn_data:
            data_d_trn_data[i] = {u:r}
        else:
            data_d_trn_data[i][u] = r
    
    # train dataset rating dictionary. user
    data_d_trn_data_u = {}
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        if u not in data_d_trn_data_u:
            data_d_trn_data_u[u] = {i:r}
        else:
            data_d_trn_data_u[u][i] = r

    # train dataset item rating mean dictionary.
    data_d_trn_data_mean = {}
    for i in data_d_trn_data:
        data_d_trn_data_mean[i] = np.mean(list(data_d_trn_data[i].values()))
    


    n_item = item.shape[0]
    n_user = len(set(data['uid']))

    # train rating matrix
    rating_matrix = np.zeros((n_user, n_item))
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        rating_matrix[u,i] = r

    # test rating matrix
    rating_matrix_test = np.zeros((n_user, n_item))
    for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
        rating_matrix_test[u,i] = r

#%%%    
    # genre matrix.
    # user genre dictionary. 사용자의 장르의 개수와 장르에 대한 평균.
    genre_d = {}
    for u in set(data['uid']):
        genre_d[u]={}
        for g in range(item.shape[1]):
            genre_d[u][g]=[0,0,0]
    
    for i in range(data.shape[0]):
        ui = data.loc[i,'uid']
        ii = data.loc[i,'iid']
        iir = data.loc[i,'r']
        
        ii_genre = np.where(item[ii,:] != 0)[0] # 장르 인덱스.
        for g in ii_genre:
            genre_d[ui][g][0]+=1
            genre_d[ui][g][1]+=iir
    
    genre_mat = np.zeros([n_user, item.shape[1]]) # 장르개수 matrix
    for u in range(n_user):
        for g in genre_d[u]:
            genre_mat[u,g] = genre_d[u][g][0]
    
    genre_mat_mean = np.zeros([n_user, item.shape[1]]) # 장르평균 matrix
    for u in range(n_user):
        for g in genre_d[u]:
            try:
                genre_mat_mean[u,g] = genre_d[u][g][1]/genre_d[u][g][0]
            except ZeroDivisionError:
                genre_mat_mean[u,g] = 0
#%%
#유사도계산#############################################################################################      
    print('\n')
    print(f'similarity calculation: {sim_name}')

    if sim_name=='cos':    
        sim=pdist(rating_matrix.T,metric=sim_cos)
        sim=squareform(sim)
    elif sim_name=='pcc':    
        sim=pdist(rating_matrix.T,metric=sim_pcc)
        sim=squareform(sim)
    elif sim_name=='Asim':
        # 아이템간의 hamming distance matirx 생성.
        Amat=np.zeros((item.shape[0],item.shape[0]))
        for i in range(item.shape[0]):
            for j in range(item.shape[0]):
                if i != j:
                    Amat[i,j] = len(np.where(1*(item[i,:]!=0)+1*(item[j,:])==2)[0])/item.shape[1]
        sim=pdist(rating_matrix.T,metric=sim_cos)
        sim=squareform(sim)
        sim=np.multiply(sim, Amat) # 코사인과 결합
    
    
    np.fill_diagonal(sim,-1)
    nb_ind=np.argsort(sim,axis=1)[:,::-1] # nearest neighbor sort.
    sel_nn=nb_ind[:,:100]
    sel_sim=np.sort(sim,axis=1)[:,::-1][:,:100]
    

    print('\n')
    print('prediction: k=10,20, ..., 100')
    rating_matrix_prediction = rating_matrix.copy()
        
    s=time.time()
    e=0
    for k in tqdm([10,20,30,40,50,60,70,80,90,100]):
        
        for user in range(rating_matrix.shape[0]): # user를 돌고, # user=0
            
            for p_item in list(np.where(rating_matrix_test[user,:]!=0)[0]): # 예측할 item을 돌고, p_item=252
                
                molecule = []
                denominator = []
                
                # call K neighbors 아이템 p_item 이랑 유사한 k개의 아이템들이 item_neihbor이 되고,,
                item_neighbor = sel_nn[p_item,:k]
                item_neighbor_sim = sel_sim[p_item,:k]

                for neighbor, neighbor_sim in zip(item_neighbor, item_neighbor_sim): # neighbor=337
                    if neighbor in data_d_trn_data_u[user].keys():
                        molecule.append(neighbor_sim * (rating_matrix[user, neighbor] - data_d_trn_data_mean[neighbor]))
                        denominator.append(abs(neighbor_sim))
                try:
                    rating_matrix_prediction[user, p_item] = data_d_trn_data_mean[p_item] + (sum(molecule) / sum(denominator))
                except : #ZeroDivisionError: user가 p_item의 이웃item을 평가한 적이 없는 경우, KeyError: test에는 있는데 train에는 없는 item.
                    e+=1
                    rating_matrix_prediction[user, p_item] = math.nan

     #3. performance
            # MAE, RMSE
            
    precision, recall, f1_score = [], [], []
    rec_score = 4 # 추천 기준 점수.
    pp=[]
    rr=[]
    for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
        p = rating_matrix_prediction[u,i]
        if not math.isnan(p):
            pp.append(p)
            rr.append(r)
    pp=[5 if i > 5 else i for i in pp]
    
    d = [abs(a-b) for a,b in zip(pp,rr)]
    mae = sum(d)/len(d)
    rmse = np.sqrt(sum(np.square(np.array(d)))/len(d))
    
    result_mae_rmse.loc[count] = [f, k, mae, rmse]
    
    
    # precision, recall, f1-score
    
    pp = np.array(pp)
    rr = np.array(rr)
    TPP = len(set(np.where(pp >= rec_score)[0]).intersection(set(np.where(rr >= rec_score)[0])))
    FPP = len(set(np.where(pp >= rec_score)[0]).intersection(set(np.where(rr < rec_score)[0])))
    FNP = len(set(np.where(pp < rec_score)[0]).intersection(set(np.where(rr >= rec_score)[0])))
    
    _precision = TPP / (TPP + FPP)
    _recall = TPP / (TPP + FNP)
    _f1_score = 2 * _precision * _recall / (_precision + _recall)

    result_topN.loc[count] = [f, k, _precision, _recall, _f1_score]
    count += 1

result_1 = result_mae_rmse.groupby(['k']).mean().drop(columns=['fold'])
result_2 = result_topN.groupby(['k']).mean().drop(columns=['fold'])
result = pd.merge(result_1, result_2, on=result_1.index).drop(columns=['key_0'])
print(result)
    
    
    
    
    
    
    