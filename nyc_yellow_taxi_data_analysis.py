# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:37:08 2018

@author: NishitP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s

from sklearn.preprocessing import normalize, scale
import datetime as dt
import os, json, requests, pickle
from scipy.stats import skew
from shapely.geometry import Point,Polygon,MultiPoint,MultiPolygon
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
#import scipy.stats as st
from tabulate import tabulate 
from sklearn import linear_model
from sklearn.metrics import  mean_squared_error, r2_score

import gmplot
import psycopg2


taxi_dt = pd.read_csv('C:/Users/NishitP/Desktop/UIUC MCS-DS/CS-498 - Cloud Computing Applications - SPRING 2018/Project/nyc_taxi_data.csv', sep=',')

#initial analysis on data quality and check
#print dataset dimensions
print("number of observations:", taxi_dt.shape[0])
print( "number of variables:", taxi_dt.shape[1])
print(taxi_dt.shape)

#print variable datatypes
print(taxi_dt.dtypes)

#check missing values
taxi_dt.isnull().sum()

#print sample data 
print(taxi_dt.head(10))

#print statistics for each variables
taxi_dt.describe()

#standard daviation of each column 
taxi_dt.std()

#pairs plot 
taxi_frame = taxi_dt[['trip_distance','rate_code','passenger_count','trip_time','payment_type','fare_amount']]
taxi_pplot = s.pairplot(taxi_frame)
taxi_pplot


# Initial analysis before fitting the linear model
# Distribution of trip distances
# define the figure with 2 subplots
fig,ax = plt.pyplot.subplots(1,2,figsize = (15,4)) 

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
 


## WHAT IS THIS ???????????
s.pairplot(taxi_dt, vars=["tip_amount","fare_amount"], size=5)

taxi_dt.plot.scatter("fare_amount","tip_amount",alpha=0.5)
plt.title("Fare Amount vs Tip")
plt.show()



# Distribution of trip distance by pickup hour
# Q: does time of the day affect the taxi ridership?
taxi_dt['pickup_hour'] = taxi_dt['pickup_time'].str[:2]

fix, axis = plt.subplots(1,1,figsize=(12,7))
#aggregate trip_distance by hour for plotting
tab = taxi_dt.pivot_table(index='pickup_hour', values='trip_distance', aggfunc=('mean','median')).reset_index()
     
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


# DO PEOPLE LIKE TO TRAVEL IN GROUP/SHARE RIDES OR PREFER TO RIDE ALONE
# Q: Are people more likely to travel in groups during holiday seasons/
taxi_dt["pickup_month"] = pd.DatetimeIndex(taxi_dt['pickup_date']).month
jan_rides  = taxi_dt[(taxi_dt["pickup_month"] == 1)]
feb_rides  = taxi_dt[(taxi_dt["pickup_month"] == 2)]
mar_rides  = taxi_dt[(taxi_dt["pickup_month"] == 3)]
apr_rides  = taxi_dt[(taxi_dt["pickup_month"] == 4)]
may_rides  = taxi_dt[(taxi_dt["pickup_month"] == 5)]
jun_rides  = taxi_dt[(taxi_dt["pickup_month"] == 6)]
jul_rides  = taxi_dt[(taxi_dt["pickup_month"] == 7)]
aug_rides  = taxi_dt[(taxi_dt["pickup_month"] == 8)]
sep_rides  = taxi_dt[(taxi_dt["pickup_month"] == 9)]
oct_rides  = taxi_dt[(taxi_dt["pickup_month"] == 10)]
nov_rides  = taxi_dt[(taxi_dt["pickup_month"] == 11)]
dec_rides  = taxi_dt[(taxi_dt["pickup_month"] == 12)]

s.countplot(x="passenger_count",data=taxi_dt)
""" about 2/3 of the total rides are single passenger rides. This seems to be working individuals. lets see how this trend looks like during
holiday season and if people are taking more group rides than single rides"""


fig, axs = plt.subplots(1,4,figsize=(18,4))
s.countplot(x="passenger_count",data=jan_rides,ax=axs[0])
s.countplot(x="passenger_count",data=feb_rides,ax=axs[1])
s.countplot(x="passenger_count",data=mar_rides,ax=axs[2])
s.countplot(x="passenger_count",data=apr_rides,ax=axs[3])
s.countplot(x="passenger_count",data=may_rides,ax=axs[0])
s.countplot(x="passenger_count",data=jun_rides,ax=axs[1])
s.countplot(x="passenger_count",data=jul_rides,ax=axs[2])
s.countplot(x="passenger_count",data=aug_rides,ax=axs[3])
s.countplot(x="passenger_count",data=sep_rides,ax=axs[0])
s.countplot(x="passenger_count",data=oct_rides,ax=axs[1])
s.countplot(x="passenger_count",data=nov_rides,ax=axs[2])
s.countplot(x="passenger_count",data=dec_rides,ax=axs[3])

""" from the monthly passenger count plots, it looks like that most of the rides throughout the year were from single riders. There are
rides in 30-40k ranges which has more than one passengers. Interesting observation here is from January to August, we see that the most 
number of passengers are between 1-6 but in the last months and specifically in holiday seasons (november and december) there are many
rides with passenger count of upto 9. We do not see the numbers in chart because it is relatively smaller than smaller passenger rides but 
it is appearing in plot because it does exists there. This trend implies that during the holiday seasons, people are indeed traveling more
in groups compare to rest of the time.
"""

# TIP RATE AS FUNCTION OF TIME/LOCATION
# Q: how does tipping rate change according to location and peak timing
fix, axis = plt.subplots(1,1,figsize=(12,7))
#aggregate trip_distance by hour for plotting
taxi_dt["tip_percent"] = taxi_dt["tip_amount"] / taxi_dt["total_amount"] * 100

taxi_dt["tip_percent"].unique()
taxi_dt["tip_percent"] = taxi_dt["tip_percent"].replace(np.inf,0)
taxi_dt["tip_percent"] = taxi_dt["tip_percent"].replace(np.nan,0)
taxi_dt.tip_percent = taxi_dt.tip_percent.astype(np.int64)

tab = taxi_dt.pivot_table(index='pickup_hour', values='tip_percent', aggfunc=('mean','median')).reset_index()
     
tab.columns = ['Hour','Mean_tip','Median_tip']
tab[['Mean_tip','Median_tip']].plot(ax=axis)
plt.ylabel('Tip percent')
plt.xlabel('Hours after midnight')
plt.title('Distribution of trip percent by pickup hour')
plt.xlim([0,23])
plt.show()

""" From the chart, it looks like that riders are giving more tips during morning peak commute hours and in the night. At some point the 
median  tip is close to 0 but mean tip si considerably higher. This is because there are many trips with no tip at all and there is no data
available for certain rides. during the processing, we make those tip values to be 0. 
"""




# TRAVEL IN TIME OF THE YEAR
holiday_dt = taxi_dt[(taxi_dt['pickup_date'] > '2013-01-01') & (taxi_dt['pickup_date'] < '2013-10-31')]
non_holiday_dt = taxi_dt[(taxi_dt['pickup_date'] > '2013-11-01') & (taxi_dt['pickup_date'] < '2013-12-31')]

holiday_rides = holiday_dt.groupby(['passenger_count']).size().reset_index(name='trip_count')
ax = holiday_rides.plot(kind='bar', title ="Passenger Counts during non holiday season", figsize=(12,5), legend=True, fontsize=12)
ax.set_xlabel("Number of passenger in trip", fontsize=12)
ax.set_ylabel("Taxi ride counts", fontsize=12)
plt.show()

non_holiday_ride = non_holiday_dt.groupby(['passenger_count']).size().reset_index(name='trip_count')
ax = non_holiday_ride.plot(kind='bar', title ="Passenger Counts during holidat seasons", figsize=(12,5), legend=True, fontsize=12)
ax.set_xlabel("Number of passenger in trip", fontsize=12)
ax.set_ylabel("Taxi ride counts", fontsize=12)
plt.show()


# FIT LINEAR REGRESSION WITH TIP AMOUNT AS RESPONSE
# data exploration
# check the derived variable 
taxi_dt.tip_percent.describe()

#we need to compare tip percent from manhatten island vs non-manhatten island
# get the coordinates of manhatten island from web to create a bounding box of territory
no_manhatten = [(40.796937, -73.949503),(40.787945, -73.955822),(40.782772, -73.943575),
              (40.794715, -73.929801),(40.811261, -73.934153),(40.835371, -73.934515),
              (40.868910, -73.911145),(40.872719, -73.910765),(40.878252, -73.926350),
              (40.850557, -73.947262),(40.836225, -73.949899),(40.806050, -73.971255)]

boundry = Polygon(no_manhatten)

#custom function to check if a rides is in manhatten or not
def is_manhatten(location,bound = boundry):
    return 1*(Point(location).within(boundry))

#createting new variable in dataset to check trip origin
taxi_dt["is_manhatten"] = taxi_dt[["pickup_lattitude","pickup_longtidue"]].apply(lambda m:is_manhatten((m[0],m[1])),axis=1)

#plot manhatten vs non-manhatten trips
non_manhatten = taxi_dt[(taxi_dt.is_manhatten==0) & (taxi_dt.tip_percent>0)].tip_percent
manhatten = taxi_dt[(taxi_dt.is_manhatten==1) & (taxi_dt.tip_percent>0)].tip_percent

bins = np.histogram(manhatten,bins=20)[1]
hist1 = np.histogram(manhatten,bins)
hist2 = np.histogram(non_manhatten,bins)

fig,axis = plt.subplots(1,1,figsize=(10,5))
w = .4*(bins[1]-bins[0])
axis.bar(bins[:-1],hist1[0],width=w,color='b')
axis.bar(bins[:-1]+w,hist2[0],width=w,color='r')
axis.set_yscale('log')
axis.set_xlabel('Tip percent')
axis.set_ylabel('Count')
axis.set_title('Tip')
axis.legend(['Non-Manhattan','Manhattah'],title='origin')
plt.show()

print('t-test results:', ttest_ind(non_manhatten,manhatten,equal_var=False))

# tip vs non tip ride distribution
# TO BE WORKED ON.................



# feature engineering 
# creating new variables - trip_direction, tip_given
"""
1. trip directions:  Direction_NS (is the cab moving Northt to South?) and Direction_EW (is the cab moving East to West). 
    These are components of the two main directions, horizontal and vertical. The hypothesis is that the traffic may be different 
    in different directions and it may affect the riders enthousiasm to tipping. They were derived from pickup and dropoff coordinates.
"""
#trip direction variable
# value of this variable will be 2 if ride is moving from north to south, 1 if moving from south to north and 0 otherewise
taxi_dt["trip_direction_NS"] = (taxi_dt.pickup_lattitude > taxi_dt.dropoff_lattitude)*1+1
indices = taxi_dt[(taxi_dt.pickup_lattitude == taxi_dt.dropoff_lattitude) & (taxi_dt.pickup_lattitude!=0)].index
taxi_dt.loc[indices,'trip_direction_NS'] = 0

# value of this variable will be 2 if ride is moving from east to west, 1 if moving from west to east and 0 otherewise
taxi_dt['trip_direction_EW'] = (taxi_dt.pickup_longtidue > taxi_dt.dropoff_longitude)*1+1
indices = taxi_dt[(taxi_dt.pickup_longtidue == taxi_dt.dropoff_longitude) & (taxi_dt.pickup_longtidue!=0)].index
taxi_dt.loc[indices,'trip_direction_EW'] = 0


#tip given variable
 taxi_dt["tip_given"] = (taxi_dt.tip_percent > 0) * 1


#after feature engineering, we need to create model
#we will build two models. First model will be classification model which will predict wheather or not passenger will give tip
#Second model will be a regression model which will predict how much tip a passenger is likely to give (if he decises to give tip)

#compare tip vs non-tip data distribution
tip = taxi_dt[taxi_dt.tip_percent > 0]
no_tip = taxi_dt[taxi_dt.tip_percent == 0]

fig,axis = plt.subplots(1,2,figsize=(14,4))
taxi_dt.tip_percent.hist(bins = 20,normed=True,ax=axis[0])
axis[0].set_xlabel('Tip percent')
axis[0].set_title('Distribution of Tip (%) - All rides')

tip.tip_percent.hist(bins = 20,normed=True,ax=axis[1])
axis[1].set_xlabel('Tip (%)')
axis[1].set_title('Distribution of Tip (%) - Rides with tips')
axis[1].set_ylabel('Group normalized count')
plt.show()




#MOST COMMON PICKUP AND DROPOFF LOCATIONS
# Q: what are the frequent pickup and dropoff places and how do they relate to specific time of the year?
# Fetch all the data returned by the database query as a list
lat_long = taxi_dt[['pickup_lattitude','pickup_longtidue']] #misspelled in data prep
len(lat_long)
lat_long = lat_long.values.T.tolist()

# Initialize two empty lists to hold the latitude and longitude values
latitude = []
longitude = [] 

# Transform the the fetched latitude and longitude data into two separate lists
for i in range(len(lat_long)):
	latitude.append(lat_long[i][0])
	longitude.append(lat_long[i][1])

# Initialize the map to the first location in the list
gmap = gmplot.GoogleMapPlotter(latitude[0],longitude[0],1)

# Draw the points on the map. I created my own marker for '#FF66666'. 
# You can use other markers from the available list of markers. 
# Another option is to place your own marker in the folder - 
# /usr/local/lib/python3.5/dist-packages/gmplot/markers/
gmap.scatter(latitude, longitude, '#FF6666', edge_width=10)

# Write the map in an HTML file
gmap.draw('map.html')

#############################################
############################################


gmap = gmplot.GoogleMapPlotter.from_geocode("New York")
gmap.draw('map.html')






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