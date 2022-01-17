# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:58:51 2021

@author: huguet


"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
sns.set()

##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff   xclara.arff
#  2d-4c-no4    spherical_4_3 
# cluto-t8-8k  cluto-t4-8k cluto-t5-8k cluto-t7-8k diamond9 banana
path = '/home/ghebras/Documents/artificial/'
databrut = arff.loadarff(open(path+"2d-4c.arff", 'r'))
data = [[x[0],x[1]] for x in databrut[0]]
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
#plt.title("Donnees standardisées")
plt.show()

########################################################################
# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
n_neighbors=10
for n_neighbors in range(2,11):
    neigh = NearestNeighbors(n_neighbors)
    nbrs = neigh.fit(data_scaled)
    distances, indices = nbrs.kneighbors(data_scaled)
    
    moyDistances=[]
    for line in distances:
        x=np.mean(line)
        moyDistances.append(x)
    moyDistances = np.sort(moyDistances, axis=0)
    
    plt.title = ("sorted distances")
    plt.plot(moyDistances, label=f"k={n_neighbors}")

plt.legend()
plt.show()


for i in np.arange(0.02,0.1,0.01):
    for min_pts in range(2,8):
        tps3=time.time()
        cl_pred = cluster.DBSCAN(eps=i, min_samples=min_pts).fit_predict(data_scaled)
        tps4=time.time()
        # Plot results
        plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8, label=f"BDscan-Eps={i} Minpt={min_pts}")
#        plt.title("Clustering DBSCAN - Epilson="+str(i)+" - Minpt="+str(min_pts))
        plt.legend()
        plt.show()
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
        n_noise_ = list(cl_pred).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("runtime = ", round((tps4 - tps3)*1000,2),"ms")


########################################################################
# FIND "interesting" values of epsilon and min_samples 
# using distances of the k NearestNeighbors for each point of the dataset
#
# Note : a point x is considered to belong to its own neighborhood  

