import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
from datetime import datetime




data = pandas.read_csv('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Data\\Combined Dataset\\122_1.csv', delimiter = ',')


#Convert the time to datetime format rather than string format
#data['Actual_Arrival_Time'] = pandas.to_datetime(data['Actual_Arrival_Time'],  format="%d/%m/%Y %H:%M:%S")
#data = data.assign(ARRIVAL_TIME = data.Actual_Arrival_Time.dt.strftime('%H:%M:%S'))
#data = data.assign(ARRIVAL_DATE = data.Actual_Arrival_Time.dt.date)
date_unique = pandas.unique(data.Date)

data = data.assign(STOP_LABEL = data.Index_Key.str[6:])
data.STOP_LABEL = pandas.to_numeric(data.STOP_LABEL)

# Loop through the data for each day and generate a graph for each day's trips.
for i in range(0,len(date_unique)):
	 data1 = data.loc[data['Date'] == date_unique[i] ]
	 data1 = data1.loc[data['Scheduled_Arrival_Time'] >= 420]
	
	#Generate graph for the whole day
	 fig = plt.figure()
	 ax = fig.add_subplot(111)
	 ax.plot(data1.Scheduled_Arrival_Time,data1.STOP_LABEL,'o')
	 #ax.yticks(data1.STOP_LABEL,data1.Stop_ID)
	 ax.set_title('122 Day\'s Journey for '+str(date_unique[i]) )
	 ax.axes.set_xticks([420,600,840,1080,1320])
	 ax.axes.set_xticklabels(["7:00","10:00","14:00","18:00","22:00"])
	 ax.axes.set_yticks([1,12,19,27, 37, 50])
	 ax.axes.set_yticklabels(["Ashington","Cabra","Dorset St","Camden Street","Dolphin's Barn","Drimnagh"])
	 ax.set_xlabel('Scheduled Arrival Time')
	 ax.set_ylabel('Location')
	 #plt.savefig('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Graphs\\122_1_'+str(date_unique[i])+'.png')
	 plt.show()
	# Generate graph for different periods of the day
	
	# Generate graph for the first third of the day
	# data2 = data.loc[(data['ARRIVAL_TIME'] >= (datetime.strptime(str(date_unique[i])+' 00:00:00', "%Y-%m-%d %H:%M:%S")).strftime('%H:%M:%S')) &  (data['ARRIVAL_TIME'] <=(datetime.strptime(str(date_unique[i])+' 11:00:00', "%Y-%m-%d %H:%M:%S")).strftime('%H:%M:%S'))]
	
	
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# plt.plot(data2.ARRIVAL_TIME,data2.STOP_LABEL,'o')
	# plt.yticks(data2.STOP_LABEL,data2.Stop_ID)
	# plt.title('122 Day\'s Journey for '+str(date_unique[i]) )
	# plt.xlabel('Time')
	# plt.ylabel('Stop Id')
	# plt.savefig('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Graphs\\122_1_'+str(date_unique[i])+'_am.png')
	
	
	# Generate graph for the second third of the day
	# data2 = data.loc[(data['ARRIVAL_TIME'] >= (datetime.strptime(str(date_unique[i])+' 11:00:00', "%Y-%m-%d %H:%M:%S")).strftime('%H:%M:%S')) &  (data['ARRIVAL_TIME'] <=(datetime.strptime(str(date_unique[i])+' 11:00:00', "%Y-%m-%d %H:%M:%S")).strftime('%H:%M:%S'))]
	
	
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# plt.plot(data2.ARRIVAL_TIME,data2.STOP_LABEL,'o')
	# plt.yticks(data2.STOP_LABEL,data2.Stop_ID)
	# plt.title('122 Day\'s Journey for '+str(date_unique[i]) )
	# plt.xlabel('Time')
	# plt.ylabel('Stop Id')
	# plt.savefig('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Graphs\\122_1_'+str(date_unique[i])+'_am.png')
	
	
	
	
	