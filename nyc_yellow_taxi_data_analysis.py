# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:37:08 2018

@author: NishitP
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as s

from sklearn.preprocessing import normalize, scale


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os, json, requests, pickle
from scipy.stats import skew
from shapely.geometry import Point,Polygon,MultiPoint,MultiPolygon
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
#import scipy.stats as st
from sklearn.preprocessing import normalize, scale
from tabulate import tabulate #pretty print of tables. source: http://txt.arboreus.com/2013/03/13/pretty-print-tables-in-python.html
from shapely.geometry import Point,Polygon,MultiPoint


taxi_1 = 'nyc_yellow_taxi_data_subset_1.csv'
taxi_2 = 'nyc_yellow_taxi_data_subset_2.csv'

#taxi_dt_1 = pd.read_csv('C:/Users/NishitP/Desktop/UIUC MCS-DS/CS-498 - Cloud Computing Applications - SPRING 2018/Project/nyc_yellow_taxi_data_subset_1.csv', sep=',')
taxi_dt = pd.read_csv('C:/Users/NishitP/Desktop/UIUC MCS-DS/CS-498 - Cloud Computing Applications - SPRING 2018/Project/nyc_taxi_data.csv', sep=',')

#initial analysis on data quality and check
#print dataset dimensions
print("number of observations:", taxi_dt.shape[0])
print( "number of variables:", taxi_dt.shape[1])
print(taxi_dt.shape)

#print variable datatypes
print(taxi_dt.dtypes)

#print sample data 
print(taxi_dt.head(10))

#print statistics for each variables
taxi_dt.describe()

# DISTRIBUTION OF TRIP DISTANCES

# define the figure with 2 subplots
fig,ax = plt.subplots(1,2,figsize = (15,4)) 

# histogram of the number of trip distance
taxi_dt.trip_distance.hist(bins=20,ax=ax[0], edgecolor='black')
ax[0].set_xlabel('Trip Distance (miles)')
ax[0].set_ylabel('Count')
ax[0].set_yscale('log')
ax[0].set_title('Histogram of Trip Distance')

# create a vector to contain Trip Distance
v = taxi_dt.trip_distance 
# exclude any data point located further than 3 standard deviations of the median point and 
# plot the histogram with 20 bins
v[~((v-v.median()).abs()>3*v.std())].hist(bins=20,ax=ax[1], edgecolor='black') # 
ax[1].set_xlabel('Trip Distance (miles)')
ax[1].set_ylabel('Count')
ax[1].set_title('A. Histogram of Trip Distance (without outliers)')

# apply a lognormal fit. Use the mean of trip distance as the scale parameter
scatter,loc,mean = lognorm.fit(taxi_dt.trip_distance.values,
                               scale=taxi_dt.trip_distance.mean(),
                               loc=0)
pdf_fitted = lognorm.pdf(np.arange(0,12,.1),scatter,loc,mean)
ax[1].plot(np.arange(0,12,.1),600000*pdf_fitted,'r') 
ax[1].legend(['data','lognormal fit'])

plt.show()
 

# DISTRIBUTION OF TRIP DISTANCE BY HOUR
s.pairplot(taxi_dt, vars=["tip_amount","fare_amount"], size=5)

taxi_dt.plot.scatter("fare_amount","tip_amount",alpha=0.5)
plt.title("Fare Amount vs Tip")
plt.show()


# random stuff for own testing
y = taxi_dt.fare_amount
x = taxi_dt[['vendor_id','rate_code','passenger_count','payment_type']]
test_frame = taxi_dt[['trip_distance','rate_code','passenger_count','trip_time','payment_type','fare_amount']]

taxi_df = pd.concat([x,y], axis=1)

scat_mat = pd.plotting.scatter_matrix(x, c=y,marker='o', hist_kwds={'bins':100}, s=100,alpha=0.7)
plt.tight_layout()
#plt.savefig('scat_mat.png')

plot = s.pairplot(test_frame)
plot