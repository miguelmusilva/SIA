# -*- coding: utf-8 -*-
"""
Modified on Wed Sept 13 2023 based on akrashen script. 
This script is used to select the summary time-series variables that represent cell behavior but are time-independent
As input it is give cell corrdinate data that is pre-processed by script (feature selection synthethic dataset tumor core) where also some features are filtered.
A short and a synthethic long dataset (twice in size) is generated and is used to compare behavior. Short dataset is clustered using dtw algorithm. For the long dataset the same clusters are assumed
Finally the input features that represent the data in both datasets are selected by training a SVC classifier on a combination of long and short datasets.
Extracted parameters are stored to use in pca_umap_projection.
@author: miguel.mu.silva
"""

# Import libraries ----------------------------------------------------------------------------------------------------------------------------------
import os
import random
import tsfresh
import json
import umap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from dtaidistance import dtw_ndim

random.seed(123)

# Import the parameters for features extraction:
os.chdir('/home/m/Escritorio/Analysis')
with open('tsfresh_not_norm.json') as json_file:
    not_norm_parameters = json.load(json_file)
with open('tsfresh_norm.json') as json_file:
    to_norm_parameters = json.load(json_file)

df = pd.read_csv('/home/m/Escritorio/df.csv')
df_sampled = df.groupby(['dataset']).sample(1000)
