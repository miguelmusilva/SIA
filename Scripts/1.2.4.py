# -*- coding: utf-8 -*-
"""
Modified on Tues Sep 12 based on akrashen script
This script processes the new data that is tracked. It takes as input the coordinates of each cell at each timepoint. 
To assess the direction of the movement it is required input of the researcher to draw a polygon corresponding to the tumor core of the image
As output it gives Tsfresh extracted measures for each parameter for each cell. These are variables that summarize the cell behavior.
In the next script (pca umap projection these are further filtered to keep only the ones that are time independent)
@author: miguel.mu.silva
"""

# Import libraries ------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
import seaborn as sns
import geopandas as gpd
import shapely.geometry
import random
import tsfresh
import json

from scipy import stats

# Import the parameters for features extraction -------------------------------------------
os.chdir('/home/m/Escritorio/Analysis')
with open('tsfresh_not_norm.json') as json_file:
    not_norm_parameters = json.load(json_file)
with open('tsfresh_norm.json') as json_file:
    to_norm_parameters = json.load(json_file)

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
df[['Position X','Position Y', 'Position Z']] = df[['Position X','Position Y', 'Position Z']].div(df.scale, axis = 0)
df = df[['TrackID2',"dataset",'Position X', 'Position Y', 'Position Z', "Time","Time_norm",'min_int']]

# Show coordinates plot. This plot show the ENTIRE data -------------------------------------
pdf_filename = '/home/m/Escritorio/Analysis/Tracked_cells_plot.pdf'
pdf_pages = pdf.PdfPages(pdf_filename)
datasets = df['dataset'].unique()
limit = -50, 1050
for dataset in datasets:
    subset = df[df['dataset'] == dataset]
    sns.lineplot(data = subset, x = 'Position X', y = 'Position Y', hue = 'TrackID2', legend = False)
    plt.title(dataset)
    plt.xlim(limit)
    plt.ylim(limit)
    plt.show()
    pdf_pages.savefig()
    plt.clf()
pdf_pages.close()

# Filter few counts per dataset and show coordinate plot. 
df_sampled = df.groupby(['dataset']).sample(1000)
# g = sns.FacetGrid(data = df_sampled, col = 'dataset', col_wrap = 1,sharex = False,sharey = False)
# g.map_dataframe(sns.lineplot, x = 'Position X', y = 'Position Y', hue = 'TrackID2', legend = False)
# plt.show()

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

# AUTOMATICALLY : Function to calculate the distance to the defined core/polygon:
def distance_tumor_per_exp_auto(dataset, max_radius):
    test_df = df[(df['dataset'].isin([dataset]))& (df['Time_norm'] == 1)]
    test_df2 = df[df['dataset'].isin([dataset])]
    test_df3 = df_sampled[df_sampled['dataset'].isin([dataset])]
    arr = test_df[['Position X', 'Position Y']].to_numpy() #Create numpy array from coordinates
    grid_x = np.linspace(min(arr[:,0]), max(arr[:,0]), num = 10)
    grid_y = np.linspace(min(arr[:,1]), max(arr[:,1]), num = 10)
    xc, yc = np.meshgrid(grid_x, grid_y)
    kernel = stats.gaussian_kde(np.vstack([arr[:,0], arr[:,1]]),bw_method = 'silverman')
    positions = np.vstack([xc.ravel(), yc.ravel()])
    z = kernel.pdf(positions).reshape(xc.shape)
    contour_set = plt.contourf(xc, yc, -z, levels = max_radius) 
    polygon = contour_set.collections[0].get_paths()[0] ## get the countour of the more inner polygon
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

# Create a variable for the cells that we will at the end consider for the behavioral profiling (only the ones that are outside of the core)
df["core"] = np.where(df['TrackID2'].isin(df_polygon.index), 'Outside_tumor_core', 'Inside_tumor_core')
df['distance_tumor'] = df['distance_tumor'].sub(df.groupby('TrackID2')['distance_tumor'].transform('first'))

# Plot the time_norm histograms to set threshold
df_timepoints = df.groupby(['TrackID2', 'dataset'], sort = False)['Time_norm'].max().to_frame().reset_index()  
g = sns.displot(data = df, x = 'Time_norm', col = 'dataset', col_wrap = 2, common_bins = False, height = 4)
df_timepoints = df_timepoints[df_timepoints.Time_norm > 200]
df = df[df['TrackID2'].isin(df_timepoints.TrackID2 )]

# Run it only for cells outside of the polygon --------------------------------------
df = df[df['core'].isin(['Outside_tumor_core'])]

# Functions that calculate the track from previous point, displacement and msd. 
def my_func_speed(a):
    """calculate the speed"""
    return (np.linalg.norm(a - [0,0,0]))

def my_func_displacement(a):
    """calculate the displacement to the first timepoint"""
    return (np.linalg.norm(a - a[[0]]))

def compute_MSD(path):
    totalsize = len(path)
    msd = []
    for i in range(totalsize-1):
        j = i+1
        msd.append(np.sum((path[0:-j]-path[j::])**2)/float(totalsize-j))
    msd = np.array(msd)
    msd = np.insert(msd, 0, 0, axis=0)
    return msd

def coord_to_series(data):  
    dfs_2 = []
    for track in data['TrackID2'].unique():
        df_sliced = data[data['TrackID2'] == track ]
        df_sliced2 = df_sliced[['Position X', 'Position Z', 'Position Y']] # Select the data of interest
        # Convert to array
        array = df_sliced2.to_numpy()
        array1 = np.diff(array,axis = 0,prepend = array[[0]]) # normalized to previous raw coordinates
        arr_norm= array - array[[0]] # normalized to first raw coordinates
        # Compute per timepoint time stats
        array_track_dist = np.apply_along_axis(my_func_speed, 1, array1)
        array_cum_dist = np.cumsum(array_track_dist, axis = 0) 
        array_start_dist = np.apply_along_axis(my_func_displacement, 1, arr_norm)
        array_msd = compute_MSD(array)
        # Combine
        d = {'track_dist': array_track_dist, 'cum_dist': array_cum_dist, 'displacement': array_start_dist,'msd':array_msd}
        df_computed = pd.DataFrame(data = d)
        df_result= pd.concat([df_sliced.reset_index(drop=True),df_computed.reset_index(drop=True)], axis = 1)
        # Calculate the diffusion coefficient from the msd based on : msd= 2*n*D*t
        df_result['diff']=df_result['msd']/(6*df_result['Time_norm'])
        dfs_2.append(df_result)
    df_processed = pd.concat(dfs_2)
    df_processed['track_dist'] = df_processed['track_dist']/df_processed['min_int']
    df_processed['cum_dist'] = df_processed['cum_dist']/df_processed['min_int']
    df_processed['displacement'] = df_processed['displacement']/df_processed['min_int']
    df_processed = df_processed[['Time', 'dataset', 'core', 'Time_norm', 'TrackID2', 'track_dist', 'cum_dist', 'displacement', 'distance_tumor', 'diff']]
    return df_processed

def extract_features_dif_length(timeseries):
    X_tsfresh_not_norm = tsfresh.extract_features(timeseries,  column_id = 'TrackID2', column_sort = 'Time_norm',kind_to_fc_parameters = not_norm_parameters)
    X_tsfresh_to_norm = tsfresh.extract_features(timeseries,  column_id = 'TrackID2', column_sort = 'Time_norm',kind_to_fc_parameters = to_norm_parameters)
    df_time = timeseries.groupby(['TrackID2'], sort = False)['Time_norm'].max()
    X_tsfresh_norm = X_tsfresh_to_norm.divide(df_time, axis = 0)
    result = pd.concat([X_tsfresh_norm, X_tsfresh_not_norm], axis = 1, join = 'inner')
    result = result.fillna(0)  ## remove na
    result = result.replace([np.inf, -np.inf],0) ## remove inf
    return(result)

df_processed = coord_to_series(df)
df_processed = extract_features_dif_length(df_processed)

# Add time to see if there is any difference in clustering
df_time_var= df.groupby(['TrackID2'], sort=False)['Time_norm'].max()
df_processed['Time_norm']=df_time_var   

# Save output
df_processed.to_csv('/home/m/Escritorio/Analysis/tracked_boxes_processed.csv')