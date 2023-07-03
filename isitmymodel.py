# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 09:18:00 2023

@author: jp042
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
df = pd.read_excel(r"C:\Users\jp042\Downloads\filterediridata1.xlsx", sheet_name = '1231r')
df = df.drop(df[df['time diff(days)'] < 100].index)

all_list=[]
final_sec=pd.DataFrame()
for i in df['Section Code'].unique():
    dff= df[df['Section Code']==i]
    
    for j in dff['Lane Number'].unique():
        print(j)
        dfd=dff[dff['Lane Number']==j]
# =============================================================================
#         plt.scatter(dfd.index,dfd['diff iri'])
#         plt.show()
# =============================================================================
        data = np.array(dfd['diff iri']).reshape(-1,1)
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(data)
        distances, indices = nbrs.kneighbors(data)
        distances = distances[:,1]
        for k in np.arange(0.5, 2.01, 0.01):
            if(np.where(distances>k)[0].size<5):
                q=np.delete(data,np.where(distances>k))
                break
            else:
                continue
        plt.scatter(pd.DataFrame(q).index,q)
        plt.show()
        subset_indices = dfd.index[dfd['diff iri'].isin(q)].tolist()
        df_subset = dfd.loc[subset_indices]
        data = np.array(df_subset['diff iri']).reshape(-1, 1)
        dbscan = DBSCAN(eps=1, min_samples=10)
        clusters = dbscan.fit_predict(data)
        df_subset['clusters']=clusters.tolist()
        outliers = df_subset[df_subset['clusters']==-1]['diff iri']
        inliers = df_subset[df_subset['clusters']!=-1]['diff iri']
        if(outliers.size<10):
            df_subset=df_subset.drop(df_subset[df_subset['clusters']==-1].index)
            outliers = df_subset[df_subset['clusters']==-1]['diff iri']
        plt.scatter(df_subset[df_subset['clusters']!=-1].index, inliers, color='blue', label='inliers')
        plt.scatter(df_subset[df_subset['clusters']==-1].index, outliers, color='orange', label='outliers')
        plt.show()
        #print(len(dfd)-len(df_subset))
        final_sec=pd.concat([final_sec,df_subset])
    all_list.append(final_sec)
        
            
            
            
            
            
            
            
            
            
            
            
            
            