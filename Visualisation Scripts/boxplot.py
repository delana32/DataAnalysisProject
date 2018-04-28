# This code creates box plots of the data to help us identify outliers in the Delay times.

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
import matplotlib.cm as cm

data = pandas.read_csv('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Data\\Combined Dataset\\122_1.csv', delimiter = ',')

#data['Actual_Arrival_Time'] = pandas.to_datetime(data['Actual_Arrival_Time'],  format="%d/%m/%Y %H:%M")
#data['Scheduled_Arrival_Time'] = pandas.to_datetime(data['Scheduled_Arrival_Time'],  format="%d/%m/%Y %H:%M:%S")
data = data.drop('Scheduled_Arrival_Time',1)
data = data.drop('Date',1)
data = data.drop('Weather_Location',1)
data = data.drop('Index_Key',1)
data = data.drop('Noise_Location',1)
data = data.drop('Delayed_Flag',1)
stop_unique = pandas.unique(data.Stop_ID)

stop_unique = [1363,1365, 1278,5149,16]

max_clusters = 20
for i in range(0,len(stop_unique)):
		
		#print "THIS INFORMATION IS FOR "+str(stop_unique[i])
		data1 = data.loc[data['Stop_ID'] == stop_unique[i] ]
		data1 = data1.reset_index(drop=True)
		data1 = data1.drop('Stop_ID',1)
		
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.plot(data1.Actual_Arrival_Time, data1.Difference, 'o')
		ax1.set_title('Plot of Actual Arrival Time v Delay for Stop '+str(stop_unique[i]))
		plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\Delay_v_arrivaltime_'+str(stop_unique[i])+'.png', bbox_inches = 'tight')
		
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		ax2.boxplot(data1.Difference)
		ax2.set_title('Boxplot for Stop : '+str(stop_unique[i]))
		#plt.show()
		plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\boxplot_'+str(stop_unique[i])+'.png', bbox_inches = 'tight')