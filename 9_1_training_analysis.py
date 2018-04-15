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

data = pandas.read_csv('D:\Masters\Data Analysis and Data Mining\Assignment 2\\Data\\Combined Dataset\\9_1_test.csv', delimiter = ',')


#stop_unique = pandas.unique(data.Stop_ID)

stop_unique = [7458,319, 2461,1347,196]

max_clusters = 14 
for i in range(0,len(stop_unique)):
		
		#print "THIS INFORMATION IS FOR "+str(stop_unique[i])
		data1 = data.loc[data['Stop_ID'] == stop_unique[i] ]
		data1_training = data1.loc[data1['Type'] == 'Training' ]
		data1_training = data1_training.reset_index(drop=True)
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

		#print "The number of rows in this new dataset for this stops is "+str(data1['Difference'].count())
		#print data1
		distorsions = []
		for n_clusters in range(2,max_clusters):
			#Fitting the K-means model
			kmeans = KMeans(n_clusters=n_clusters, random_state=10)
			kmeans = kmeans.fit(data_training)
			labels = kmeans.labels_
			results = pandas.DataFrame([data_training.index,labels]).T
			distorsions.append(kmeans.inertia_)
			#print "The amount of cluster names returned: "+str(len(results))

			#Merging the cluster labels and the dataset
			merged = data_training.join(results,lsuffix='_data', rsuffix='_results')
			data1_final = merged.rename(index = str, columns={ 0:"old_index", 1:"label"})
			data1_final = data1_final.drop('old_index',1)
			
# For plot of Elbow method without silhouette and clustering graphs
		#plot of elbow method
		#plt.plot(range(2,max_clusters),distorsions,'o-')
		#plt.title('The Elbow method for Stop '+str(stop_unique[i]))
		#plt.xlabel('No. of Clusters')
		#plt.ylabel('Sum of Square Distances to nearest Cluster Centre')
		#plt.show()


				

			#fig = plt.figure()
			
			#ax = fig.add_subplot(111, projection='3d')
			#for j in range(0,n_clusters):
			#	ith_cluster_values = data1_final.loc[data1_final['label'] == j]
		#		color = cm.spectral(float(j) / n_clusters)
			#	ax.scatter(ith_cluster_values.Difference, ith_cluster_values.Sound_Level, ith_cluster_values.Scheduled_Arrival_Time,facecolors=color, label = color, edgecolors=(color),marker = 'o')
				
				


				
			#plt.title('Clustered Data for '+str(n_clusters)+' clusters for stop '+str(stop_unique[i]))
			#ax.set_xlabel('Difference')
			#ax.set_ylabel('Sound_Level')
			#ax.set_zlabel('Actual_Arrival_Time (s)')
						#plt.legend(['Cluster 0','Cluster 1', 'Cluster 2'], ['bo','oo','ro'])
			#plt.show()
			#plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\cluster_plot_'+str(n_clusters)+'_'+str(stop_unique[i])+'.png', bbox_inches = 'tight')
			
				 
				#print data_final;
			
			if n_clusters < 8:
				silhouette_avg = silhouette_score(data1_final, labels)
			#print silhouette_avg
				sample_silhouette_values = silhouette_samples(data1_final, labels)
				#print sample_silhouette_values
			
			
			#fig1 = plt.figure()
			#ax1 = fig1.add_subplot(111)
				ax1= plt.subplot(3, 2, n_clusters-1)
				ax1.set_xlim([-0.1, 1])
					# The (n_clusters+1)*10 is for inserting blank space between silhouette
					# plots of individual clusters, to demarcate them clearly.
				 
				ax1.set_ylim([0, int(data1_final.Difference.count()) + (n_clusters + 1) * 10])
				y_lower = 10
				for k in range(n_clusters):
					# Aggregate the silhouette scores for samples belonging to
					# cluster i, and sort them
					kth_cluster_silhouette_values = \
						sample_silhouette_values[labels == k]

					kth_cluster_silhouette_values.sort()

					size_cluster_k = kth_cluster_silhouette_values.shape[0]
					y_upper = y_lower + size_cluster_k

					color = cm.spectral(float(k) / n_clusters)
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
			#ax1.set_title("The silhouette plot for "+str(n_clusters)+" clusters for stop "+str(stop_unique[i]))
				ax1.set_title("No. Clusters: "+str(n_clusters))
				ax1.set_xlabel("The silhouette coefficient values")
				ax1.set_ylabel("Cluster label")

				ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
			
			#plt.show()

		
		
		plt.suptitle('Silhouette Plot for Bus: 9, Stop: '+str(stop_unique[i]))
		plt.tight_layout()
		#plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\final_graphs\\9_1_silhouette_plot_'+str(stop_unique[i])+'.png')
		plt.show()
		
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		#plot of elbow method
		ax2.plot(range(2,max_clusters),distorsions,'o-')
		ax2.set_title('9: The Elbow method for Stop '+str(stop_unique[i]))
		ax2.set_xlabel('No. of Clusters')
		ax2.set_ylabel('Distortions')
		plt.show()
		#plt.savefig('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Graphs\\final_graphs\\9_1_elbow_plot_'+str(stop_unique[i])+'.png', bbox_inches = 'tight')
