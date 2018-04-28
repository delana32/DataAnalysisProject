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

data = pandas.read_csv('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Data\\Combined Dataset\\122_1_test.csv', delimiter = ',')

#data['Actual_Arrival_Time'] = pandas.to_datetime(data['Actual_Arrival_Time'],  format="%d/%m/%Y %H:%M")
#data['Scheduled_Arrival_Time'] = pandas.to_datetime(data['Scheduled_Arrival_Time'],  format="%d/%m/%Y %H:%M:%S")
date_unique = pandas.unique(data.Date)


stop_unique = pandas.unique(data.Stop_ID)

stop_unique = [1363,1365, 1278,5149,16]

n_clusters = [2,2,2,2,4]
for i in range(0,len(stop_unique)):
		
		#print "THIS INFORMATION IS FOR "+str(stop_unique[i])
		data1 = data.loc[data['Stop_ID'] == stop_unique[i] ]
		data1 = data1.loc[data['Type'] == 'Training' ]
		
		for j in range(0,6):
	
			plt.subplot(3, 2, j+1)
			data2 = data1.loc[data['Date'] == date_unique[j] ]
			data2=data2.sort_values('Actual_Arrival_Time',ascending = 'True')
			plt.plot(data2.Actual_Arrival_Time, data2.Difference, 'o-')
			plt.title('Date: '+str(date_unique[j]))
			plt.xlabel('Arrival Time (mins)')
			plt.ylabel('Delay(mins)')
		plt.suptitle('122: Stop: '+str(stop_unique[i]))
		plt.tight_layout()
		plt.show()