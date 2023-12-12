# -*- coding: utf-8 -*-
"""
Modified on Tue Sept 12 based on akrashen script
This script selects the features to which time series in the format of coordinates can be converted. 
It is not perfect so I use another script where parameters are further selected byt dtw clustering and SVC clasifier.
@author: miguelmusilva
"""

# Import libraries -----------------------------------------
import os
import glob
import random
import tsfresh
import json
import shapely.geometry

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import tsfresh.feature_extraction.settings as settings

random.seed(123)

# Import all files ------------------------------------------------------------------------
random.seed(123)
path = '/home/m/Escritorio/Analysis/tracked_boxes'
filenames = os.listdir(path)

t_int ={'dataset':['tracked_boxes_LSD_RC1_Exp10_img001.csv', 'tracked_boxes_LSD_RC1_Exp10_img002.csv',
                   'tracked_boxes_LSD_RC1_Exp10_img003.csv', 'tracked_boxes_LSD_RC1_Exp10_img005.csv',
                   'tracked_boxes_LSD_RC1_Exp10_img007.csv', 'tracked_boxes_LSD_RC1_Exp11_Img001.csv',
                   'tracked_boxes_LSD_RC1_Exp11_Img003.csv', 'tracked_boxes_LSD_RC1_Exp11_Img005.csv'],
        'min_int' :[16,16,16,16,16,16,16,16],
        'scale' :[1,1,1,1,1,1,1,1]}
t_int= pd.DataFrame(t_int)
t_int['dataset']=t_int['dataset'].str.split('.').str[0]
dfs = []
for file in filenames:
    newdf = pd.read_csv(path +"/" + file)
    newdf['dataset']= file.split(".")[0]
    dfs.append(newdf)
df = pd.concat(dfs)

# Add new unique column to tracks
df['TrackID2'] = df['dataset'].astype('str') + '_' + df['TrackID'].astype('str')
# Merge both dataframes, scale the data and normalize the timepoints. Then, keep only columns of interest
df = df.merge(t_int, on = 'dataset', how = 'left')
df['Time'] = df['Time'] * df['min_int']
df['Time_norm'] = df['Time'] - df.sort_values('Time').groupby('TrackID2')['Time'].transform('first') + 1
df[['Position X', 'Position Y', 'Position Z']] = df[['Position X', 'Position Y', 'Position Z']].div(df.scale, axis = 0)
df = df[['TrackID2', "dataset", 'Position X', 'Position Y', 'Position Z', "Time", "Time_norm", 'min_int']]

# Filter few counts per dataset and show coordinate plot. 
df_sampled = df.groupby(['dataset']).sample(1000)

# Function to calculate the distance to the defined core/polygon:
def calculate_distance(x, coordinates):
    point = shapely.geometry.Point(x)
    poly1 = shapely.geometry.Polygon(coordinates)
    return poly1.distance(point)

'''
# MANUALLY: Function to calculate the distance to the defined core/polygon: 

def distance_tumor_per_exp_manual(dataset, coordinates):
    test_df = df[df['dataset'].isin([dataset])]
    points1 = test_df[['Position X','Position Y']].to_numpy()
    test_df['distance_tumor'] = np.apply_along_axis(calculate_distance, axis = 1, arr = points1, coordinates = coordinates)
    return test_df

#eg_coordinates = [(400,1020),(500,900),(600,850),(700,900),(900,1020)]
'''

# AUTOMATICALLY : Function to calculate the distance to the defined core/polygon: -------------------------------------
def distance_tumor_per_exp_auto(dataset, max_radius):
    test_df = df[(df['dataset'].isin([dataset]))& (df['Time_norm'] == 1)]
    test_df2 = df[df['dataset'].isin([dataset])]
    test_df3 = df_sampled[df_sampled['dataset'].isin([dataset])]
    arr = test_df[['Position X', 'Position Y']].to_numpy() # Create numpy array from coordinates
    grid_x = np.linspace(min(arr[:,0]), max(arr[:,0]), num = 10)
    grid_y = np.linspace(min(arr[:,1]), max(arr[:,1]), num = 10)
    xc, yc = np.meshgrid(grid_x, grid_y)
    kernel = stats.gaussian_kde(np.vstack([arr[:,0], arr[:,1]]),bw_method = 'silverman')
    positions = np.vstack([xc.ravel(), yc.ravel()])
    z = kernel.pdf(positions).reshape(xc.shape)
    contour_set = plt.contourf(xc, yc, -z, levels = max_radius) 
    polygon = contour_set.collections[0].get_paths()[0] ## Get the countour of the more inner polygon
    poly1 = shapely.geometry.Polygon(polygon.vertices)
    p = gpd.GeoSeries(poly1)
    p.plot()
    sns.lineplot(data = test_df3, x = 'Position X', y = 'Position Y', hue = 'TrackID2',legend = False)
    plt.show()
    points1 = test_df2[['Position X','Position Y']].to_numpy()
    test_df2['distance_tumor'] = np.apply_along_axis(calculate_distance, axis = 1, arr = points1, coordinates = polygon.vertices)
    return test_df2

# Check each dataset to make sure the right center polygon is defined. 
# If needed, adjust the max radius or make coordinates manually.
tracked_boxes_LSD_RC1_Exp10_img001 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp10_img001', max_radius = 5)
tracked_boxes_LSD_RC1_Exp10_img002 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp10_img002', max_radius = 1) 
tracked_boxes_LSD_RC1_Exp10_img003 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp10_img003', max_radius = 7)
tracked_boxes_LSD_RC1_Exp10_img005 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp10_img005', max_radius = 3)
tracked_boxes_LSD_RC1_Exp10_img007 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp10_img007', max_radius = 4) 

tracked_boxes_LSD_RC1_Exp11_Img001 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp11_Img001', max_radius = 5)
tracked_boxes_LSD_RC1_Exp11_Img003 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp11_Img003', max_radius = 5)
tracked_boxes_LSD_RC1_Exp11_Img005 = distance_tumor_per_exp_auto('tracked_boxes_LSD_RC1_Exp11_Img005', max_radius = 2) 

df = pd.concat([
              tracked_boxes_LSD_RC1_Exp11_Img001,
              tracked_boxes_LSD_RC1_Exp11_Img003,
              tracked_boxes_LSD_RC1_Exp11_Img005,
              tracked_boxes_LSD_RC1_Exp10_img001,
              tracked_boxes_LSD_RC1_Exp10_img002,
              tracked_boxes_LSD_RC1_Exp10_img003,
              tracked_boxes_LSD_RC1_Exp10_img005,
              tracked_boxes_LSD_RC1_Exp10_img007,
              tracked_boxes_LSD_RC1_Exp11_Img001,
              tracked_boxes_LSD_RC1_Exp11_Img003,
              tracked_boxes_LSD_RC1_Exp11_Img005])

# Select the TrackID that are outside the polygon only
df_polygon = df.groupby(['TrackID2'], sort = False)['distance_tumor'].min()
df_polygon = df_polygon[df_polygon > 0]
df = df[df['TrackID2'].isin(df_polygon.index)]
df['distance_tumor'] = df['distance_tumor'].sub(df.groupby('TrackID2')['distance_tumor'].transform('first'))

######################################################################################################################

# Select the tracks that have more than 200 minutes:
df_timepoints = df.groupby(['TrackID2', 'dataset'], sort = False)['Time_norm'].max().to_frame().reset_index()  
g = sns.displot(data = df, x = 'Time_norm', col = 'dataset', col_wrap = 2, common_bins = False, height = 4)
df_timepoints = df_timepoints[df_timepoints.Time_norm > 200]
df = df[df['TrackID2'].isin(df_timepoints.TrackID2 )]

# Transform all the coordinates to relative values:
df['Position X'] = df['Position X'].sub(df.groupby('TrackID2')['Position X'].transform('first'))
df['Position Y'] = df['Position Y'].sub(df.groupby('TrackID2')['Position Y'].transform('first'))
df['Position Z'] = df['Position Z'].sub(df.groupby('TrackID2')['Position Z'].transform('first'))

# Get two dataframes and randomly slightly modify the second one
df1 = df

df2 = df
random_array = np.random.uniform(0, 0.3, size = (df2.shape[0], 3))
random_array = np.add(df2[['Position X', 'Position Y', 'Position Z']].values, random_array)
df2[['Position X', 'Position Y', 'Position Z']] = random_array

idx = df2.groupby(['TrackID2'])['Time'].transform(max) == df2['Time']
df_last = df2[idx]
df_last = df_last[['TrackID2', 'Position X', 'Position Y', 'Position Z', 'Time']]
df_last = df_last.rename(columns = {
                                    'Position X' : 'last_X', 
                                    'Position Y' : 'last_Y', 
                                    'Position Z' : 'last_Z', 
                                    'Time' : 'last_t'
                                    })
merged_left = pd.merge(left = df2, right = df_last, how = 'left', left_on = 'TrackID2', right_on = 'TrackID2')
merged_left['Position X'] = merged_left['Position X'] + merged_left['last_X']
merged_left['Position Y'] = merged_left['Position Y'] + merged_left['last_Y']
merged_left['Position Z'] = merged_left['Position Z'] + merged_left['last_Z']
merged_left['Time'] = merged_left['Time'] + merged_left['last_t']
merged_left = merged_left.drop(['last_X', 'last_Y', 'last_Z', 'last_t'], axis = 1)

synthetic = pd.concat([df1.reset_index(drop = True), 
                       merged_left.reset_index(drop = True)], 
                       axis = 0)
synthetic = synthetic.reset_index()

# Visualize the short (real data) and long dataset (synthetic data)
test_Set1 = df1[df1['TrackID2'].isin(['tracked_boxes_LSD_RC1_Exp11_Img001_6'])]
plt.plot(test_Set1['Position X'], test_Set1['Position Y'], 'b')
plt.show()

test_Set2 = synthetic[synthetic['TrackID2'].isin(['tracked_boxes_LSD_RC1_Exp11_Img001_6'])]
plt.plot(test_Set2['Position X'], test_Set2['Position Y'], 'b')
plt.show()


########################################################################################
# Function that calculates the track from previous point, displacement, msd:
def my_func_speed(a):
    """calculate the speed"""
    return (np.linalg.norm(a - [0,0,0]))

def my_func_displacement(a):
    """calculate the displacement to the first timepoint"""
    return (np.linalg.norm(a - a[[0]]))

def compute_MSD(path):
    totalsize=len(path)
    msd=[]
    for i in range(totalsize-1):
        j=i+1
        msd.append(np.sum((path[0:-j]-path[j::])**2)/float(totalsize-j))
    msd=np.array(msd)
    msd= np.insert(msd, 0, 0, axis=0)
    return msd

# Split by unique trackID2 and process
def coord_to_series(data):  
    dfs_2 = []
    for track in data['TrackID2'].unique():
        df_sliced = data[data['TrackID2'] == track ]
        df_sliced2 = df_sliced[['Position X', 'Position Z', 'Position Y']] ## select the data of interest
        # convert to array
        array = df_sliced2.to_numpy()
        array1= np.diff(array,axis=0,prepend=array[[0]]) #normalized to previous raw coordinates
        arr_norm= array - array[[0]] #normalized to first raw coordinates
        # Compute per timepoint time stats
        array_track_dist=np.apply_along_axis(my_func_speed, 1, array1)
        array_cum_dist = np.cumsum(array_track_dist, axis = 0) 
        array_start_dist=np.apply_along_axis(my_func_displacement, 1, arr_norm)
        array_msd=compute_MSD(array)
        # combine
        d = {'track_dist': array_track_dist, 'cum_dist': array_cum_dist, 'displacement': array_start_dist,'msd':array_msd}
        df_computed = pd.DataFrame(data=d)
        df_result= pd.concat([df_sliced.reset_index(drop=True),df_computed.reset_index(drop=True)], axis=1)
        # calculate the diffusion coefficient from the msd based on : msd= 2*n*D*t
        df_result['diff']=df_result['msd']/(6*df_result['Time'])
        dfs_2.append(df_result)
    df_processed = pd.concat(dfs_2)
    df_processed['track_dist']=df_processed['track_dist']/df_processed['min_int']
    df_processed['cum_dist']=df_processed['cum_dist']/df_processed['min_int']
    df_processed['displacement']=df_processed['displacement']/df_processed['min_int']
    df_processed= df_processed[['Time','TrackID2', 'track_dist', 'cum_dist', 'displacement', 'distance_tumor','diff']]
    return df_processed

df_processed_short = coord_to_series(df1)
df_processed_long = coord_to_series(synthetic)

# Show example time series plot
test_df2 = df_processed_long.sample(20)
test_df2 = df_processed_long[df_processed_long['TrackID2'].isin(test_df2['TrackID2'])].reset_index()
sns.lineplot(data = test_df2, x = 'Time', y = 'cum_dist', hue = 'TrackID2', legend = False)
plt.show()
sns.lineplot(data = test_df2, x = 'Time', y = 'displacement', hue = 'TrackID2', legend = False)
plt.show()
sns.lineplot(data = test_df2, x = 'Time', y = 'diff', hue = 'TrackID2', legend = False)
plt.show()

test_Set=df_processed_long[df_processed_long['TrackID2'].isin(['tracked_boxes_LSD_RC1_Exp10_img007_1','tracked_boxes_LSD_RC1_Exp10_img007_200'])]
plt.plot(test_Set['Time'], test_Set['cum_dist'], 'g')
plt.show()
plt.plot(test_Set['Time'], test_Set['displacement'], 'g')
plt.show()
plt.plot(test_Set['Time'], test_Set['track_dist'], 'b')
plt.show()
plt.plot(test_Set['Time'], test_Set['diff'], 'r')
plt.show()

# Select features of interest based on a subset with tsfresh extractor
test = df_processed_short.iloc[:10000,]
test = test.fillna(0)
test = test.replace([np.inf, -np.inf], 0)

# Extract, remove NA or inf and remove features with the same variance
X_tsfresh = tsfresh.extract_features(test, column_id = 'TrackID2', column_sort = 'Time')
X_tsfresh = X_tsfresh.fillna(0)
X_tsfresh = X_tsfresh.replace([np.inf, -np.inf], 0)
selector = VarianceThreshold(0)
selector.fit(X_tsfresh)

# Construct the corresponding settings object and apply the transformation on both tracks (short and long)
selected_columns = X_tsfresh.columns[selector.get_support()]
parameters = tsfresh.feature_extraction.settings.from_columns(X_tsfresh.columns[selector.get_support()])

df_processed_short = df_processed_short.fillna(0)
df_processed_short = df_processed_short.replace([np.inf, -np.inf],0)
X_tsfresh_full_short = tsfresh.extract_features(df_processed_short, column_id = 'TrackID2', column_sort = 'Time', kind_to_fc_parameters = parameters)
X_tsfresh_full_short = X_tsfresh_full_short.fillna(0)
X_tsfresh_full_short = X_tsfresh_full_short.replace([np.inf, -np.inf], 0)

df_processed_long = df_processed_long.fillna(0)
df_processed_long = df_processed_long.replace([np.inf, -np.inf],0)
X_tsfresh_full_long = tsfresh.extract_features(df_processed_long,  column_id = 'TrackID2', column_sort = 'Time',kind_to_fc_parameters = parameters)
X_tsfresh_full_long = X_tsfresh_full_long.fillna(0)
X_tsfresh_full_long =X_tsfresh_full_long.replace([np.inf, -np.inf],0)

X_tsfresh_full_long=X_tsfresh_full_long.reindex(columns=X_tsfresh_full_short.columns)

# Calculate the fold change between long and short series
fc = pd.DataFrame(X_tsfresh_full_long.to_numpy() / X_tsfresh_full_short.to_numpy(), columns = X_tsfresh_full_short.columns, index = X_tsfresh_full_short.index)
cor_df = X_tsfresh_full_long.corrwith(X_tsfresh_full_short, axis = 0)
cor_df = cor_df[cor_df > 0.9]
col_mean = fc.mean()
col_mean = col_mean[col_mean.index.isin(cor_df.index)]

## From these we need to indentify which ones have a factor difference of 2 (since this is the difference in time)
ax = col_mean[col_mean.between(-4,4)].plot.hist(bins = 50, alpha = 0.5)
plt.show()

std_threshold = 2.0
not_norm = col_mean[col_mean.between(-std_threshold, std_threshold)]
not_norm_parameters = tsfresh.feature_extraction.settings.from_columns(not_norm.index)

to_norm = col_mean[col_mean.abs() > std_threshold]
to_norm_parameters = tsfresh.feature_extraction.settings.from_columns(to_norm.index)

# Process and normalize the parameters of interest
def extract_features_dif_length(timeseries):
    X_tsfresh_not_norm = tsfresh.extract_features(timeseries,  column_id = 'TrackID2', column_sort = 'Time',kind_to_fc_parameters = not_norm_parameters)
    X_tsfresh_to_norm = tsfresh.extract_features(timeseries,  column_id = 'TrackID2', column_sort = 'Time',kind_to_fc_parameters = to_norm_parameters)
    df_time = timeseries.groupby(['TrackID2'], sort = False)['Time'].max()
    X_tsfresh_norm = X_tsfresh_to_norm.divide(df_time, axis = 0)
    result = pd.concat([X_tsfresh_norm, X_tsfresh_not_norm], axis = 1, join = 'inner')
    result = result.fillna(0)
    result = result.replace([np.inf, -np.inf],0)
    return(result)

test1 = extract_features_dif_length(df_processed_short)    
test1['target'] ='short'
test2 = extract_features_dif_length(df_processed_long)    
test2['target'] ='long'

combi = pd.concat([test1, test2], axis = 0)
X = combi.drop(['target'], axis = 1).values

x = StandardScaler().fit_transform(X)
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
principal_df.reset_index(inplace = True, drop = True)
combi.reset_index(inplace = True, drop = True)
final_df = pd.concat([principal_df, combi[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['short', 'long']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = final_df['target'] == target
    ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
               , final_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 8
               , alpha = 0.3)
ax.legend(targets)
ax.grid()
plt.show()

os.chdir('/home/m/Escritorio/')
a_file = open("tsfresh_not_norm.json", "w")
json.dump(to_norm_parameters, a_file)
a_file.close()

a_file = open("tsfresh_norm.json", "w")
json.dump(not_norm_parameters, a_file)
a_file.close()