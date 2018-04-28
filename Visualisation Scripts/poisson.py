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
#data = data.loc[data['Date'] == '22/03/2018']
data = data.drop('Scheduled_Arrival_Time',1)
data = data.drop('Date',1)
data = data.drop('Weather_Location',1)
data = data.drop('Index_Key',1)
data = data.drop('Noise_Location',1)
data = data.drop('Delayed_Flag',1)
data1 = data.loc[data['Stop_ID'] == 1278 ]



data1 = data1.reset_index(drop=True)
data1 = data1.drop('Stop_ID',1)
data2 = data.loc[data['Stop_ID'] == 1363 ]
data2 = data2.drop('Stop_ID',1)
data2 = data2.reset_index(drop=True)

data3 = data.loc[data['Stop_ID'] == 5149 ]
data3 = data3.drop('Stop_ID',1)
data3 = data3.reset_index(drop=True)

data4 = data.loc[data['Stop_ID'] == 16 ]
data4 = data4.drop('Stop_ID',1)
data4 = data4.reset_index(drop=True)

stop_unique = [1363,1365, 1278,5149,16]
for i in range(0, len(stop_unique)):
	data1 = data.loc[data['Stop_ID'] == stop_unique[i] ]
	fig = plt.figure()
	plt.hist(data1.Difference)
	plt.title("Stop number : "+str(stop_unique[i]))
	plt.text(50,100,"mean is : "+str(data1.Difference.mean()))
	plt.show()	
# plt.plot(data1.index, data1.Actual_Arrival_Time, 'b')
# plt.plot(data2.index, data2.Actual_Arrival_Time, 'y')
# plt.plot(data3.index, data3.Actual_Arrival_Time, 'm')
# plt.plot(data4.index, data4.Actual_Arrival_Time, 'g')

