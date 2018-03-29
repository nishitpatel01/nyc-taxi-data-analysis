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
import datetime as dt
import os, json, requests, pickle
from scipy.stats import skew
from shapely.geometry import Point,Polygon,MultiPoint,MultiPolygon
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
#import scipy.stats as st
from tabulate import tabulate #pretty print of tables. source: http://txt.arboreus.com/2013/03/13/pretty-print-tables-in-python.html


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


#RIDERSHIP IMPACT OF TIME OF THE DAY ON TRIP DISTANCE
taxi_dt['pickup_hour'] = taxi_dt['pickup_time'].str[:2]

fix, axis = plt.subplots(1,1,figsize=(12,7))
#aggregate trip_distance by hour for plotting
tab = taxi_dt.pivot_table(index='pickup _hour', values='trip_distance', aggfunc=('mean','median')).reset_index()
     
tab.columns = ['Hour','Mean_distance','Median_distance']
tab[['Mean_distance','Median_distance']].plot(ax=axis)
plt.ylabel('Metric (miles)')
plt.xlabel('Hours after midnight')
plt.title('Distribution of trip distance by pickup hour')
plt.xlim([0,23])
plt.show()

print(tabulate(tab.values.tolist(),["Hour","Mean Distance","Median Distance"]))

""" Plot is suggesting that the mean trip distances is longer in morning and evening hours. This could be the population that uses cabs to commute for
work. But if so the the evening commuter are much less than morning commuters. This indicates that the people who takes taxi in the morning to work 
do not use it when they go back home. Hypothetically this makes sense as people do not want to get late to work. Other hypothesis is there are usually
large number of people who take flights during morning and evening. This could also be a contributing factor that pushes the mean higher that other time
of the day. To prove that, we will now take a look at the airport and non-airport taxi rides and its distribution.
"""

# CALCULATE AIRPORT TRIPS
airport_trips = taxi_dt[(taxi_dt.rate_code == 2) | (taxi_dt.rate_code ==3)]  #rate_code 2 and 3 are jfk and Newark respectively 

airport_trips['pickup_hour'] = airport_trips['pickup_time'].str[:2]

fix, axis = plt.subplots(1,1,figsize=(12,7))
#aggregate trip_distance by hour for plotting
tab = airport_trips.pivot_table(index='pickup_hour', values='trip_distance', aggfunc=('mean','median')).reset_index()
     
tab.columns = ['Hour','Mean_distance','Median_distance']
tab[['Mean_distance','Median_distance']].plot(ax=axis)
plt.ylabel('Metric (miles)')
plt.xlabel('Hours after midnight')
plt.title('Distribution of airport trip distance by pickup hour')
plt.xlim([0,23])
plt.show()

print(tabulate(tab.values.tolist(),["Hour","Mean Distance","Median Distance"]))

""" This plot proves that the mean and the median of airport trips are almost similar during the days. These trips are appeared to be at the highest
between 7PM - 10PM. ........MORE TO ADD.
"""

v_at = airport_trips.trip_distance #airport trips
v_nat = taxi_dt.loc[~taxi_dt.index.isin(v_at.index),'trip_distance'] #non-airport trips

#excluding any outliers (points that are more than 3 sd away)
v_at = v_at[~((v_at - v_at.median()).abs()>3*v_at.std())]
v_nat = v_nat[~((v_nat - v_nat.median()).abs()>3*v_nat.std())]

bins = np.histogram(v_at,normed=True)[1]
h_at = np.histogram(v_at,bins=bins,normed=True)
h_nat = np.histogram(v_nat,bins=bins,normed=True)


# plot distributions of trip distance normalized among groups
fig,ax = plt.subplots(1,2,figsize = (15,6))
w = .4*(bins[1]-bins[0])
ax[0].bar(bins[:-1],h_at[0],alpha=1,width=w,color='b')
ax[0].bar(bins[:-1]+w,h_nat[0],alpha=1,width=w,color='g')
ax[0].legend(['Airport trips','Non-airport trips'],loc='best',title='group')
ax[0].set_xlabel('Trip distance (miles)')
ax[0].set_ylabel('Group normalized trips count')
ax[0].set_title('A. Trip distance distribution')

# plot hourly distribution
airport_trips.pickup_hour.value_counts(normalize=True).sort_index().plot(ax=ax[1])
taxi_dt.loc[~taxi_dt.index.isin(v_at.index),'pickup_hour'].value_counts(normalize=True).sort_index().plot(ax=ax[1])
ax[1].set_xlabel('Hours after midnight')
ax[1].set_ylabel('Group normalized trips count')
ax[1].set_title('B. Hourly distribution of trips')
ax[1].legend(['Airport trips','Non-airport trips'],loc='best',title='group')
plt.show()

list(taxi_dt)

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