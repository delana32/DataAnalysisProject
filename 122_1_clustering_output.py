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

#Gets unique stop numbers from data column
#stop_unique = pandas.unique(data.Stop_ID)
stop_unique = [1363,1365, 1278,5149,16]

#initialising the number of clusters that each stop should be run for 
n_clusters = [3,3,3,3,3]

for i in range(0,len(stop_unique)):
		
		#creating the dataset for output
		data1 = data.loc[data['Stop_ID'] == stop_unique[i] ]
		
		data1 = data1.reset_index(drop=True)
	
		#Splitting the training and test data
		
		data1_training = data1.loc[data1['Type'] == 'Training' ]
		data1_training = data1_training.reset_index(drop=True)
		
		data1_test = data1.loc[data1['Type'] == 'Test' ]
		data1_test = data1_test.reset_index(drop=True)
		
		#creating the training dataset for kmeans algorithm
		data_training = data1_training.drop('Stop_ID',1)
		data_training = data_training.drop('Actual_Arrival_Time',1)
		data_training = data_training.drop('Date',1)
		data_training = data_training.drop('Weather_Location',1)
		data_training = data_training.drop('Index_Key',1)
		data_training = data_training.drop('Gust_Speed',1)
		data_training = data_training.drop('Feels_Like',1)
		data_training = data_training.drop('Noise_Location',1)
		data_training = data_training.drop('Delayed_Flag',1)
		data_training = data_training.drop('Total_Rainfall',1)
		data_training = data_training.drop('Type',1)
		
		
		#creating the test dataset for kmeans algorithm
		data_test = data1_test.drop('Stop_ID',1)
		data_test = data_test.drop('Actual_Arrival_Time',1)
		data_test = data_test.drop('Date',1)
		data_test = data_test.drop('Weather_Location',1)
		data_test = data_test.drop('Index_Key',1)
		data_test = data_test.drop('Gust_Speed',1)
		data_test = data_test.drop('Feels_Like',1)
		data_test = data_test.drop('Noise_Location',1)
		data_test = data_test.drop('Delayed_Flag',1)
		data_test = data_test.drop('Total_Rainfall',1)
		data_test = data_test.drop('Type',1)
		
		
		#Array for the sum of square distances for the kmeans output
		distorsions = []
		
		#Fitting the K-means model
		kmeans = KMeans(n_clusters=n_clusters[i], random_state=10)
		
		kmeans_training = kmeans.fit(data_training)
		labels_training = kmeans_training.labels_
		results_training = pandas.DataFrame([data_training.index,labels_training]).T
		
		#Predicting the K-means model
		kmeans_test= kmeans.predict(data_test)
		labels_test= kmeans_test
		results_test = pandas.DataFrame([data_test.index,labels_test]).T
		
		#distorsions.append(kmeans.inertia_)
		

		#Merging the cluster labels and the training dataset
		training_merged = data_training.join(results_training,lsuffix='_data', rsuffix='_results')
		training_final = training_merged.rename(index = str, columns={ "Scheduled_Arrival_Time": "Scheduled_Arrival_Time", "Difference": "Difference","Sound_Level":"Sound_Level","Temperature":"Temperature","Wind_Speed":"Wind_Speed","Visibility":"Visibility","Rain_1_Hr":"Rain_1_Hr", 0:"old_index", 1:"label"})
		training_final = training_final.drop('old_index',1)
		
		#Merging the cluster labels and the training dataset for output
		output_training_data = data1_training.join(results_training,lsuffix='_data', rsuffix = '_results')
		output_training_data = output_training_data.rename(index = str, columns={ 0:"old_index", 1:"label"})
		output_training_data = output_training_data.drop('old_index',1)
		#output_data.to_csv('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\Cluster_data\\122_1_'+str(stop_unique[i])+'_scheduled.csv',sep=',')
			
		#Merging the cluster labels and the test dataset
		test_merged = data_test.join(results_test,lsuffix='_data', rsuffix='_results')
		test_final = test_merged.rename(index = str, columns={ "Scheduled_Arrival_Time": "Scheduled_Arrival_Time", "Difference": "Difference","Sound_Level":"Sound_Level","Temperature":"Temperature","Wind_Speed":"Wind_Speed","Visibility":"Visibility","Rain_1_Hr":"Rain_1_Hr", 0:"old_index", 1:"label"})
		test_final = test_final.drop('old_index',1)
		
		#Merging the cluster labels and the test dataset for output
		output_test_data = data1_test.join(results_test,lsuffix='_data', rsuffix = '_results')
		output_test_data = output_test_data.rename(index = str, columns={ 0:"old_index", 1:"label"})
		output_test_data = output_test_data.drop('old_index',1)
		#output_data.to_csv('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\Cluster_data\\122_1_'+str(stop_unique[i])+'_scheduled.csv',sep=',')
		
		#Merging the training and test datasets for output
		output = output_training_data.append(output_test_data, ignore_index=True)
		output.to_csv('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\Cluster_data\\122_1_'+str(stop_unique[i])+'.csv',sep=',')
		

		fig = plt.figure()
		
		ax = fig.add_subplot(111, projection='3d')
		#Plotting the clustered training data
		for j in range(0,n_clusters[i]):
			ith_cluster_values = training_final.loc[training_final['label'] == j]
			color = cm.spectral(float(j) / n_clusters[i])
			ax.scatter(ith_cluster_values.Difference, ith_cluster_values.Sound_Level, ith_cluster_values.Scheduled_Arrival_Time,facecolors=color, label = 'Training, cluster '+str(j), edgecolors=(color),marker = '+')
			
				
		#Plotting the clustered test data		
		for m in range(0,n_clusters[i]):
			mth_cluster_values = test_final.loc[test_final['label'] == m]
			color = cm.spectral(float(m) / n_clusters[i])
			ax.scatter(mth_cluster_values.Difference, mth_cluster_values.Sound_Level, mth_cluster_values.Scheduled_Arrival_Time,facecolors=color, label = 'Test, cluster '+str(m), edgecolors=(color),marker = 'o')
			

				
		plt.title('122: '+str(n_clusters[i])+' Clusters for Bus Stop '+str(stop_unique[i]))
		ax.set_xlabel('Delay (mins)')
		ax.set_ylabel('Sound Level (dB)')
		ax.set_zlabel('Scheduled Arrival Time')
		ax.axes.set_zticks([360,600,840,1080,1320])
		ax.axes.set_zticklabels(["6:00","10:00","14:00","18:00","22:00"])
		plt.legend(loc = 2)	
		plt.tight_layout()
		plt.show()
		#plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\final_graphs\\cluster_plot_'+str(n_clusters[i])+'_'+str(stop_unique[i])+'_scheduled.png', bbox_inches = 'tight')
			
				 
				#print data_final;

		silhouette_avg = silhouette_score(test_final, labels_test)
		#print silhouette_avg
		sample_silhouette_values = silhouette_samples(test_final, labels_test)
			#print sample_silhouette_values
			

		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.set_xlim([-0.1, 1])
					# The (n_clusters+1)*10 is for inserting blank space between silhouette
					# plots of individual clusters, to demarcate them clearly.
				 
		ax1.set_ylim([0, int(test_final.Difference.count()) + (n_clusters[i] + 1) * 10])
		y_lower = 10
		for k in range(0,n_clusters[i]):
				# Aggregate the silhouette scores for samples belonging to
				# cluster i, and sort them
				kth_cluster_silhouette_values = \
					sample_silhouette_values[labels_test == k]

				kth_cluster_silhouette_values.sort()

				size_cluster_k = kth_cluster_silhouette_values.shape[0]
				y_upper = y_lower + size_cluster_k

				color = cm.spectral(float(k) / n_clusters[i])
				ax1.fill_betweenx(np.arange(y_lower, y_upper),
								  0, kth_cluster_silhouette_values,
								  facecolor=color, edgecolor=color, alpha=0.7)

				# Label the silhouette plots with their cluster numbers at the middle
				ax1.text(-0.05, y_lower + 0.5 * size_cluster_k, str(k))

				# Compute the new y_lower for next plot
				y_lower = y_upper + 10  # 10 for the 0 samples
			# print merged
		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
		ax1.set_title("The Silhouette plot for Bus Stop: "+str(stop_unique[i])+", No. Clusters: "+str(n_clusters[i])+" for Test data")
		ax1.set_xlabel("Silhouette coefficient value")
		ax1.set_ylabel("Cluster label")

		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
		
		#plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\final_graphs\\silhouette_plot_122_1_'+str(stop_unique[i])+'_scheduled.png', bbox_inches = 'tight')
		plt.show()
"""
		for l in range(0,n_clusters[i]):
			lth_cluster = data1_final.loc[data1_final['label'] == l]
			fig2 = plt.figure()
			ax2 = fig2.add_subplot(111)
			ax2.hist(lth_cluster.Difference)
			ax2.set_title('Histogram of Delay Times for Cluster '+str(n_clusters[i])+' at Stop '+str(stop_unique[i]))
			ax2.set_xlabel('Delay')
			ax2.set_ylabel('Frequency')
			ax2.text(40,30,r'$\mu is '+str(round(lth_cluster.Difference.mean(),2)))
			plt.show()
			#plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\final_graphs\\hist_122_1_'+str(stop_unique[i])+'_'+str(n_clusters[i])+'_scheduled.png', bbox_inches = 'tight')
"""
