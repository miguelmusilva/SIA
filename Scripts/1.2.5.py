# -*- coding: utf-8 -*-
"""
Modified on Tue Sep 26 2023
This script takes preprocessed time series data from coordinates_to_timeseries_final_v2, that convertes timeseries in multivariate summarized data
And it perfromed PCA and unbiased classification of the cell behavior
@author: akrashen
"""

# Import libraries -------------------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.model_selection import train_test_split
import umap
import umap.plot

from sklearn.preprocessing import StandardScaler
import joblib 
import sklearn.cluster as cluster
import os
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.mixture import GaussianMixture

# Import the parameters to keep for classification:
joblib.dump(scaler, '/home/m/Escritorio/Analysis/std_scaler.bin', compress=True)
joblib.dump(pca_model, '/home/m/Escritorio/Analysis/pca_model.bin', compress=True)

parameters = pd.read_csv('/home/m/Escritorio/Analysis/parameters_20220927.csv')
df = pd.read_csv('/home/m/Escritorio/Analysis/tracked_boxes_processed.csv', float_precision='high', parse_dates = False)
df.rename(columns = {'Unnamed: 0' : 'TrackID2'}, inplace = True)
df = df.fillna(0)
dataset = df['TrackID2'].str.split('_').str[:-1].str.join('_')
df["dataset"] = dataset.values


X = df.loc[:, df.columns.isin(parameters.iloc[:,0])]
y = df.loc[:, df.columns.isin(['dataset','TrackID2' ,'Time_norm'])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42, shuffle = True)

# Rescale all the features
scaler = StandardScaler().fit(X_train.values)
features_train = scaler.transform(X_train.values)
X_train = pd.DataFrame(features_train, columns = X_train.columns, index = X_train.index)

features_test = scaler.transform(X_test.values)
X_test = pd.DataFrame(features_test, columns = X_test.columns, index = X_test.index)

# PCA and remove all the variables that are related to Time
pca = PCA(n_components = 0.95)
pca_model = pca.fit(X_train)
principal_components = pca_model.transform(X_train)
principal_df =pd.DataFrame(data = principal_components, columns = ['PC_' + str(i + 1) for i in range(principal_components.shape[1])])

# Identify the components that are correlated to Time
cor_coef = principal_df.corrwith(df['Time_norm'])
PC_subset = cor_coef[cor_coef.abs() < 0.5]
X_PC = principal_df.loc[:, principal_df.columns.isin(PC_subset.index)]
X_PC = X_PC.sort_index(axis = 0)
X_PC.index = y_train.TrackID2

mapper = umap.UMAP(n_neighbors = 50, min_dist = 0.25, random_state = 42).fit(X_PC)


























## Plot the training file
sns_plot = sns.scatterplot(x=trans.embedding_[:, 0], y=trans.embedding_[:, 1],
hue=y_train['dataset'].to_list(),
alpha=.5, linewidth=0.5)
# Adjust legend
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.show()


## Plot the tracks time
sns_plot = sns.scatterplot(x=trans.embedding_[:, 0], y=trans.embedding_[:, 1],
hue=y_train['Time_norm'],
alpha=.5, linewidth=0.5)
# Adjust legend
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.show()


# Cluster the UMAP on dbscan
kmeans_labels = cluster.KMeans(n_clusters=6).fit_predict(trans.embedding_)



#hdbscan_labels = hdbscan.HDBSCAN(min_samples=11, min_cluster_size=10).fit_predict(trans.embedding_)
#clustered = (hdbscan_labels >= 0)
#hdbscan_labels=pd.DataFrame(hdbscan_labels, columns=['Cluster'])
kmeans_labels_df=pd.DataFrame(kmeans_labels, columns=['Cluster'])
sns.set(style="white")
sns_plot = sns.scatterplot(x=trans.embedding_[:, 0], y=trans.embedding_[:, 1],
hue=kmeans_labels_df.Cluster.astype('category') ,
alpha=.9, linewidth=0.5)
# Adjust legend
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.show()

X_train['X']=trans.embedding_[:, 0]
X_train['Y']=trans.embedding_[:, 1]

# CLuster with Gaussina mixture
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X_train[['X','Y']]) for n in n_components]
plt.plot(n_components, [m.bic(X_train[['X','Y']]) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_train[['X','Y']]) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()
# Check the value that minimized X BIC and AIC

gmm = GaussianMixture(n_components=7,random_state=0,covariance_type="tied")
gmm.fit(X_train[['X','Y']])
labels = gmm.predict(X_train[['X','Y']])
plt.scatter(X_train['X'], X_train['Y'], c=labels, cmap='viridis')

# Agregate the data by cluster to plot a heatmap
X_train['Cluster']=labels

fig2 = plt.figure()
sns_plot = sns.scatterplot(x=X_train['X'], y=X_train['Y'],
hue=X_train['Cluster'].astype('category'),
alpha=.9, linewidth=0.5)
# Adjust legend
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.show()



# plot a heatmap

df_cl=X_train.loc[:, ~X_train.columns.isin(['X','Y' ])]

df_summary=df_cl.groupby(
   ['Cluster']
).mean()    


df_norm_row = df_summary.apply(lambda x: (x-x.mean())/x.std(), axis = 0)
df_norm_row=df_norm_row[['distance_tumor__quantile__q_0.1','track_dist__range_count__max_1__min_-1','diff__fft_coefficient__attr_"abs"__coeff_5',
                         'track_dist__autocorrelation__lag_3','cum_dist__absolute_maximum','cum_dist__fft_aggregated__aggtype_"skew"',
                         'distance_tumor__range_count__max_1__min_-1', 'track_dist__skewness', 'distance_tumor__median']]
fig3 = plt.figure()
sns.set(font_scale=1.5)
cg=sns.clustermap(df_norm_row.transpose(),cmap='coolwarm',xticklabels=True,row_cluster=True,col_cluster=True,yticklabels=True)
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
plt.savefig('E:/SURF_2/Shared/Dream3DLab (Groupfolder)/1.Projects/LSD_LandscapeStimulatedDynamics/3.Analysis/Test_data_EXP29/processed_MA/Behavioral_classification_heatmap.pdf')
plt.show()
#plt.show()
   
# Visualize the tracks that are clustered:
df_cl2=y_train[X_train['Cluster']>=0]
df_cl2['Cluster'] =df_cl['Cluster'] 

##import tracks coordinates
path='/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/processed_MA/Output20220927_coordinates_MA1.csv'

df_coord = pd.read_csv(path,float_precision='high',parse_dates=False)
## filter the TrackID2 that were clustered and assign cluster
#df_coord=df_coord[df_coord['TrackID2'].isin(df_cl2['TrackID2'])]
df_coord = pd.merge(left=df_coord, right=df_cl2[['TrackID2', 'Cluster']], how='left', left_on='TrackID2', right_on='TrackID2')
## sample some tracks
tracks = df_coord[df_coord['TrackID2'].isin(df_cl2['TrackID2'])]

tracks = tracks.groupby(['TrackID2']).sample(1)
tracks = tracks.groupby(['Cluster']).sample(5)

df_sample= df_coord[df_coord['TrackID2'].isin(tracks['TrackID2'])]

## Transform all the coordinates to relative values:
df_sample['Position X'] = df_sample['Position X'].sub(df_sample.groupby('TrackID2')['Position X'].transform('first'))
df_sample['Position Y'] = df_sample['Position Y'].sub(df_sample.groupby('TrackID2')['Position Y'].transform('first'))
df_sample['Position Z'] = df_sample['Position Z'].sub(df_sample.groupby('TrackID2')['Position Z'].transform('first'))
#df_sample=df_sample[df_sample['Time']<199]
df_sample=df_sample.sort_values(by=["Cluster", "Time"], axis=0)
for track in df_sample['TrackID2'].unique():
    df_sliced = df_sample[df_sample['TrackID2'] == track]
    plt.figure()
    plt.plot(df_sliced['Position X'], df_sliced['Position Y'])
    plt.title(print(track,str("Cluster") ,df_sliced.iloc[1]['Cluster']))
    plt.ylim(-40, 40)
    plt.xlim(-40, 40)
    plt.show()
    if track == df_sample.iloc[-1]['TrackID2']:
        break
## Print each experiment color-coded by behavior cluster    
df_coord['Cluster'] = df_coord['Cluster'].astype('category')
#colors = {0.0: "#F28E2B", 1.0: "#4E79A7", 2.0: "#79706E", 3.0: '#a9aaab', 4.0: "#79706E", 5.0: "#79706E",6.0: "#79706E",7.0: "#79706E"}
# palette=colors,
plt.figure(figsize=(6, 6), 
           dpi = 600)
sns.set(style="white")
g = sns.FacetGrid(data=df_coord, col='dataset', col_wrap=2,sharex=False,sharey=False)
g.map_dataframe(sns.lineplot, x='Position X', y='Position Y', hue='Cluster',units="TrackID2",estimator=None,lw=1)
g.add_legend()
#plt.show()


## rename the clusters
df_coord['Cluster'] = df_coord['Cluster'].cat.rename_categories({ 0:'slow', 1:'returning_tum', 2:'static', 3: 'leaving_erratic', 4: 'slow_returning',  5:'leaving_tum', 6: 'static_erratic'})
fig4 = plt.figure(figsize=(6, 6), 
           dpi = 600)
sns.set(style="white")
g = sns.FacetGrid(data=df_coord, col='dataset', col_wrap=2,sharex=False,sharey=False)
g.map_dataframe(sns.lineplot, x='Position X', y='Position Y', hue='Cluster',units="TrackID2",estimator=None,lw=1)
g.add_legend()
plt.savefig('/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/processed_MA/Behavioral_classification2.pdf')
plt.show()

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
save_multi_image('/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/processed_MA/Behavioral_classification2.pdf')

pp = PdfPages('/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/processed_MA/Behavioral_classification.pdf')
pp.savefig(fig4)
pp.savefig(fig2)
pp.savefig(fig3)

pp.close()    
df_coord.to_csv('/home/m/Escritorio/LSD/Analysis/Test_data_EXP29/processed_MA/Behavioral_classification.csv')