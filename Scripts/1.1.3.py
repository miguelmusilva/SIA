# -*- coding: utf-8 -*-

'''
Created on Friday 1 September 2023 based on previous akrashen script
This script takes the segmented nuclei of the LSM coregistered files and classifies them into cell populations
The output is a df that contains the labels and the predicted cell type (also the characteristics of these cells)
@author : miguel.mu.silva
'''

# Import libraries ------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import umap
import sklearn.cluster as cluster

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

# Preprocessing --------------------------------------------------------------------------
path = '/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/assigned_nuclei.tsv'
df = pd.read_csv(path, sep = '\t') # Load data
df = df.fillna(0) # Remove NA values

# Split dataset in train and test to protect it and keep only columns of interest --------
columns_to_keep = ['mean_inside_olig2_norm', 'mean_inside_iba1_norm',
                   'mean_inside_map2_norm', 'mean_inside_dmg_norm',
                   'in-out_olig2_norm', 'mean_inside_map2_ilastik_mask_filt',
                   'mean_inside_iba1_ilastik_mask_filt',
                   'mean_border_iba1_norm', 'mean_border_map2_norm',
                   'mean_border_map2_ilastik_mask_filt','in-out_dmg_norm',
                   'mean_border_iba1_ilastik_mask_filt', 'dapi_smoothness']

df_train, df_test = train_test_split(df, test_size = 0.05, random_state = 42, shuffle = True)

x_train = df_train.loc[:, df_train.columns.isin(columns_to_keep)]
#x_train = x_train.sort.index(axis = 0)
y_train = df_train.loc[:, ~df_train.columns.isin(columns_to_keep)]

x_test = df_test.loc[:, df_test.columns.isin(columns_to_keep)]
y_test = df_test.loc [:, ~df_test.columns.isin(columns_to_keep)]

# Rescale the features -------------------------------------------------------------------
scaler = StandardScaler().fit(x_train.values)
features = scaler.transform(x_train.values)
x_train = pd.DataFrame(data = features, columns = x_train.columns, index = x_train.index)

trans = umap.UMAP(n_neighbors = 20, min_dist = 0.15, random_state = 42).fit(x_train)

# Plot the training data -----------------------------------------------------------------
names = {
    'DAPI_smooth' : 'dapi_smoothness',
    'DMG' : 'dmg_segment', 
    'DMG_inside' : 'mean_inside_dmg_norm',
    'MAP2' : 'map2_segment', 
    'MAP2_inside' : 'mean_inside_map2_norm',
    'MAP2_filt' : 'mean_border_map2_ilastik_mask_filt',
    'Olig2' : 'olig2_segment',
    'Olig2_inside' : 'mean_inside_olig2_norm',
    'Iba1' : 'iba1_segment',
    'Iba1_inside' : 'mean_inside_iba1_norm',
    'Iba1_filt' : 'mean_border_iba1_ilastik_mask_filt' }

# Save all images created into a PDF --------------------------------------------------------------
plt.rcParams['figure.figsize'] = [7.00, 3.50]
plt.rcParams['figure.autolayout'] = True

pp = PdfPages('Output_LSM_heatmap.pdf')
for key, value in names.items():
    if value in columns_to_keep:
        plt.figure()
        sns_plot = sns.scatterplot(x = trans.embedding_[:, 0], 
                                y = trans.embedding_[:, 1], 
                                hue = x_train[value].to_list(), 
                                alpha = 0.5, linewidth = 0.5)
        sns_plot.legend(loc = 'center left', bbox_to_anchor = (1, 0.5)).set_title(key)
        pp.savefig()
    else:    
        plt.figure()
        sns_plot = sns.scatterplot(x = trans.embedding_[:, 0], 
                                y = trans.embedding_[:, 1], 
                                hue = y_train[value].to_list(), 
                                alpha = 0.5, linewidth = 0.5)
        sns_plot.legend(loc = 'center left', bbox_to_anchor = (1, 0.5)).set_title(key)
        pp.savefig()

def clustering_agglo(df, n_clusters):
    connectivity = kneighbors_graph(df, int(df.shape[0]/7), include_self = False)
    agc = cluster.AgglomerativeClustering(linkage = 'ward', connectivity = connectivity, n_clusters = n_clusters)
    agc.fit(df)
    print('Found clusters', len(np.unique(agc.labels_)))
    return agc.labels_

clusters = clustering_agglo(trans.embedding_, 9)
labels = pd.DataFrame(clusters, columns = ['Cluster']) # Cluster the UMAP on dbscan
svc = SVC().fit(trans.embedding_, labels) # Train classifier for assigning new cluster based on embedding
svc.score(trans.transform(x_train), labels) # accuracy scoring

# Predict to test accuracy per class. Plot the confusion matrix on SVC classifier -------------------
predicted_labels = svc.predict(trans.transform(x_train))
C = confusion_matrix(labels, predicted_labels)
C = C * 100 / C.astype(np.float64).sum(axis =1)

sns.set(font_scale = 1)
sns.heatmap(C, annot = True, cmap = 'coolwarm', fmt = '.3g')
plt.show()

df = df.join(labels) # Joint the cluster information by index to the original dataframe

# Save scaler and export the parameters that need to be applied -----------------------------------
joblib.dump(scaler, 'std_scaler.bin', compress = True)
joblib.dump(trans, 'umap.bin', compress = True)
joblib.dump(svc, 'svc.bin', compress = True)
np.savetxt('parameters_for_LSM_classification.csv', columns_to_keep, delimiter = ',', fmt = '%s')

# TODO : plot the heatmap















# Rename the clusters ----------------------------------------------------------------------------
x_train['Cluster'] = x_train['Cluster'].astype('category').values
x_train['Cluster'] = x_train['Cluster'].cat.rename_categories({0 :'Other', 1 : 'MAP2_low', 
                                                               2 : 'Other2', 3 : 'Iba1_low', 
                                                               4 : 'DMG_Olig2neg', 5 : 'DMG_Olig2pos', 
                                                               6 : 'Oligo', 7 : 'Iba1_high', 8 : 'MAP2_high'})

df_cl = x_train
df_cl2 = pd.concat([x_train, y_train], axis = 1)

df_summary = df_cl.groupby(['Cluster']).mean()
df_norm_row = df_summary.apply(lambda x : (x - x.mean())/x.std(), axis = 0)
plt.figure()
sns.clustermap(df_norm_row.drop(['X', 'Y'], axis = 1), 
               cmap = 'coolwarm', xticklabels = True, row_cluster = True, 
               col_cluster = True).fig.suptitle('features used for classification')
plt.show()

df_summary2 = df_cl2.groupby(['Cluster']).mean()
df_norm_row2 = df_summary2.apply(lambda x : (x - x.mean())/x.std(), axis = 0)
plt.figure()
sns.clustermap(df_norm_row2.drop(['X', 'Y'], axis = 1), 
               cmap = 'coolwarm', xticklabels = True, row_cluster = True, 
               col_cluster = True).fig.suptitle('all features')
plt.show()




pp.close()
