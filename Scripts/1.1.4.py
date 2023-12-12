# -*- coding: utf-8 -*-

'''
Created on Friday 1 September 2023 based on previous akrashen script
This script takes the segmented nuclei of the LSM coregistered files and classifies them into cell populations
The output of this script is a df that contains the labels and the predicted cell type (also the characteristics of these cells)
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

# Import Models, scaler and data load and preprocessing ------------------------------------------
scaler = joblib.load('/home/m/Escritorio/std_scaler.bin')
trans = joblib.load('/home/m/Escritorio/umap.bin')
#svc = joblib.load('svc.bin')

path = '/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/assigned_nuclei.tsv'
df = pd.read_csv(path, sep = '\t') # Load data
df = df.fillna(0) # Remove NA values
df = df.sort_index(axis = 0)

# Keep only columns of interest and rescale the features using the predefined scaler and UMAP
columns_to_keep = ['mean_inside_olig2_norm', 'mean_inside_iba1_norm',
                   'mean_inside_map2_norm', 'mean_inside_dmg_norm',
                   'in-out_olig2_norm', 'mean_inside_map2_ilastik_mask_filt',
                   'mean_inside_iba1_ilastik_mask_filt',
                   'mean_border_iba1_norm', 'mean_border_map2_norm',
                   'mean_border_map2_ilastik_mask_filt','in-out_dmg_norm',
                   'mean_border_iba1_ilastik_mask_filt', 'dapi_smoothness']

x_pred = df.loc[:, df.columns.isin(columns_to_keep)]
x_pred = x_pred.sort_index(axis = 0)
y_pred = df.loc[:, ~df.columns.isin(columns_to_keep)]
y_pred = y_pred.sort_index(axis = 0)

features = scaler.transform(x_pred.values)
x_pred = pd.DataFrame(data = features, columns = x_pred.columns, index = x_pred.index)
pred_embedding = trans.transform(x_pred)

# Plot the UMAP plot
sns_plot = sns.scatterplot(x = pred_embedding[:, 0], y = pred_embedding[:, 1], 
                           hue = y_pred['mean_border_dmg'].to_list(), 
                           alpha = 0.5, linewidth = 0.1)
sns_plot.legend(loc = 'center left', bbox_to_anchor = (1, .5)).set_tittle('DMG')

predicted_labels = svc.predict(pred_embedding) # Predict to test accuracy per class


x_pred['X'] = pred_embedding[:, 0]
x_pred['Y'] = pred_embedding[:, 1]
x_pred['Cluster'] = predicted_labels
x_pred['Cluster'] = x_pred['Cluster'].astype('category').values
x_pred['Cluster'] = x_pred['Cluster'].cat.rename_categories({0:'Other', 1:'MAP2_low', 
                                                             2:'Other2', 3: 'Iba1_low', 
                                                             4: 'DMG_Olig2neg',  5:'DMG_Olig2pos', 
                                                             6: 'Oligo',  7:'Iba1_high', 8 :'MAP2_high'})

sns_plot = sns.scatterplot(x = x_pred['X'], y = x_pred['Y'], 
                           hue = x_pred['Cluster'].astype('category'), 
                           alpha = 0.5, linewidth = 0.1)
sns_plot.legend(loc = 'center left', bbox_to_anchor = (1, .5)).set_tittle('Cluster')