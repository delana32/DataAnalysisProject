# This script reads in the data for route 122 stop 16 and runs the algorithm for each of the clusters in the data set.
# It then splits the data into new frames depending on the clusters
# The regression model is then run for each cluster and the results are output to CSV files.
# The algorithim is also ran on the dataset as a whole to see if the accuracy is different when not clustered

#install.packages("psych")
#install.packages("car")
#install.packages("Hmisc")
#install.packages("caTools")
#install.packages("biglm")
#install.packages("RcppArmadillo")
library("psych")
library("car")
library("Hmisc")
library("caTools")
library("biglm")
library("RcppArmadillo")


# read-in all bus data for route 122 stop 16
DublinBusData <- read.csv("Raw Data/Clustered_Data/122_1_16.csv")

# brief summary of dataframe
summary(DublinBusData)
# show field headings
ls(DublinBusData)
# field headings and data types
str(DublinBusData)


# filter for cluster 0 for route 122 stop 16
DublinBusData_0 <- select(filter(DublinBusData, label ==0),c(1:20))
# filter for cluster 1 for route 122 stop 16
DublinBusData_1 <- select(filter(DublinBusData, label ==1),c(1:20))
# filter for cluster 2 for route 122 stop 16
DublinBusData_2 <- select(filter(DublinBusData, label ==2),c(1:20))

###########################################################
############### Cluster 0 Route 122 Stop 16 ###############
###########################################################

# splitting data into training and test data sets for cluster 0
BusTrain_0 <- select(filter(DublinBusData_0, Type =="Training"),c(1:20)) #training data
BusTest_0 <- select(filter(DublinBusData_0, Type =="Test"),c(1:20)) #test data/validation

# checking record count to see if split was accurate
dim(DublinBusData_0)
dim(BusTrain_0)
dim(BusTest_0)

#attach data set to be used
attach(BusTrain_0)

# regression model
model <- fastLm(Difference ~ Scheduled_Arrival_Time + Temperature + 
                  Wind_Speed + Total_Rainfall + Sound_Level)

# Use model to predict Number of Journeys in the training set
BusTrain_0$Pred_Difference <- predict(model, newdata = BusTrain_0)

# Use model to predict Number of Journeys in the test set
BusTest_0$Pred_Difference <- predict(model, newdata = BusTest_0)

# See output of the Linear Regression model and the main components of it
summary(model)

# Check how well the model performs on the training set - correlation^2, RMSE and MAE
train.corr <- cor(BusTrain_0$Pred_Difference, BusTrain_0$Difference)
train.RMSE <- sqrt(mean((BusTrain_0$Pred_Difference - BusTrain_0$Difference)^2))
train.MAE <- mean(abs(BusTrain_0$Pred_Difference - BusTrain_0$Difference))
c(train.corr^2, train.RMSE, train.MAE)

# Check how well the model performs on the test set - correlation^2, RMSE and MAE
test.corr <- cor(BusTest_0$Pred_Difference, BusTest_0$Difference)
test.RMSE <- sqrt(mean((BusTest_0$Pred_Difference - BusTest_0$Difference)^2))
test.MAE <- mean(abs(BusTest_0$Pred_Difference - BusTest_0$Difference))
c(test.corr^2, test.RMSE, test.MAE)

# add in column to compare predicted value to actual value for each row
BusTrain_0["Difference_in_Prediction_temp"] <- BusTrain_0$Difference - BusTrain_0$Pred_Difference
BusTest_0["Difference_in_Prediction_temp"] <- BusTest_0$Difference - BusTest_0$Pred_Difference
# round the columns to ensure it is inline with original difference variable
BusTrain_0["Pred_Difference_Final"] <- round(BusTrain_0$Pred_Difference, digits=0)
BusTest_0["Pred_Difference_Final"] <- round(BusTest_0$Pred_Difference, digits=0)
BusTrain_0["Difference_in_Prediction"] <- round(BusTrain_0$Difference_in_Prediction_temp, digits=0)
BusTest_0["Difference_in_Prediction"] <- round(BusTest_0$Difference_in_Prediction_temp, digits=0)

#output data
write.csv(BusTrain_0, "Outputs/Route_122_1_16/01. BusTrain_122_1_16_Cluster0.csv")
write.csv(BusTest_0, "Outputs/Route_122_1_16/02. BusTest_122_1_16_Cluster0.csv")

###################### End Cluster 0 ######################

###########################################################
############### Cluster 1 Route 122 Stop 16 ###############
###########################################################

# splitting data into training and test data sets for cluster 1
BusTrain_1 <- select(filter(DublinBusData_1, Type =="Training"),c(1:20)) #training data
BusTest_1 <- select(filter(DublinBusData_1, Type =="Test"),c(1:20)) #test data/validation

# checking record count to see if split was accurate
dim(DublinBusData_1)
dim(BusTrain_1)
dim(BusTest_1)

#attach data set to be used
attach(BusTrain_1)

# regression model
model <- fastLm(Difference ~ Scheduled_Arrival_Time + Temperature + 
                  Wind_Speed + Total_Rainfall + Sound_Level)

# Use model to predict Number of Journeys in the training set
BusTrain_1$Pred_Difference <- predict(model, newdata = BusTrain_1)

# Use model to predict Number of Journeys in the test set
BusTest_1$Pred_Difference <- predict(model, newdata = BusTest_1)

# See output of the Linear Regression model and the main components of it
summary(model)

# Check how well the model performs on the training set - correlation^2, RMSE and MAE
train.corr <- cor(BusTrain_1$Pred_Difference, BusTrain_1$Difference)
train.RMSE <- sqrt(mean((BusTrain_1$Pred_Difference - BusTrain_1$Difference)^2))
train.MAE <- mean(abs(BusTrain_1$Pred_Difference - BusTrain_1$Difference))
c(train.corr^2, train.RMSE, train.MAE)

# Check how well the model performs on the test set - correlation^2, RMSE and MAE
test.corr <- cor(BusTest_1$Pred_Difference, BusTest_1$Difference)
test.RMSE <- sqrt(mean((BusTest_1$Pred_Difference - BusTest_1$Difference)^2))
test.MAE <- mean(abs(BusTest_1$Pred_Difference - BusTest_1$Difference))
c(test.corr^2, test.RMSE, test.MAE)

# add in column to compare predicted value to actual value for each row
BusTrain_1["Difference_in_Prediction_temp"] <- BusTrain_1$Difference - BusTrain_1$Pred_Difference
BusTest_1["Difference_in_Prediction_temp"] <- BusTest_1$Difference - BusTest_1$Pred_Difference
# round the columns to ensure it is inline with original difference variable
BusTrain_1["Pred_Difference_Final"] <- round(BusTrain_1$Pred_Difference, digits=0)
BusTest_1["Pred_Difference_Final"] <- round(BusTest_1$Pred_Difference, digits=0)
BusTrain_1["Difference_in_Prediction"] <- round(BusTrain_1$Difference_in_Prediction_temp, digits=0)
BusTest_1["Difference_in_Prediction"] <- round(BusTest_1$Difference_in_Prediction_temp, digits=0)

#output data
write.csv(BusTrain_1, "Outputs/Route_122_1_16/03. BusTrain_122_1_16_Cluster1.csv")
write.csv(BusTest_1, "Outputs/Route_122_1_16/04. BusTest_122_1_16_Cluster1.csv")

###################### End Cluster 1 ######################

###########################################################
############### Cluster 2 Route 122 Stop 16 ###############
###########################################################

# splitting data into training and test data sets for cluster 2
BusTrain_2 <- select(filter(DublinBusData_2, Type =="Training"),c(1:20)) #training data
BusTest_2 <- select(filter(DublinBusData_2, Type =="Test"),c(1:20)) #test data/validation

# checking record count to see if split was accurate
dim(DublinBusData_2)
dim(BusTrain_2)
dim(BusTest_2)

#attach data set to be used
attach(BusTrain_2)

# regression model
model <- fastLm(Difference ~ Scheduled_Arrival_Time + Temperature + 
                  Wind_Speed + Total_Rainfall + Sound_Level)

# Use model to predict Number of Journeys in the training set
BusTrain_2$Pred_Difference <- predict(model, newdata = BusTrain_2)

# Use model to predict Number of Journeys in the test set
BusTest_2$Pred_Difference <- predict(model, newdata = BusTest_2)

# See output of the Linear Regression model and the main components of it
summary(model)

# Check how well the model performs on the training set - correlation^2, RMSE and MAE
train.corr <- cor(BusTrain_2$Pred_Difference, BusTrain_2$Difference)
train.RMSE <- sqrt(mean((BusTrain_2$Pred_Difference - BusTrain_2$Difference)^2))
train.MAE <- mean(abs(BusTrain_2$Pred_Difference - BusTrain_2$Difference))
c(train.corr^2, train.RMSE, train.MAE)

# Check how well the model performs on the test set - correlation^2, RMSE and MAE
test.corr <- cor(BusTest_2$Pred_Difference, BusTest_2$Difference)
test.RMSE <- sqrt(mean((BusTest_2$Pred_Difference - BusTest_2$Difference)^2))
test.MAE <- mean(abs(BusTest_2$Pred_Difference - BusTest_2$Difference))
c(test.corr^2, test.RMSE, test.MAE)

# add in column to compare predicted value to actual value for each row
BusTrain_2["Difference_in_Prediction_temp"] <- BusTrain_2$Difference - BusTrain_2$Pred_Difference
BusTest_2["Difference_in_Prediction_temp"] <- BusTest_2$Difference - BusTest_2$Pred_Difference
# round the columns to ensure it is inline with original difference variable
BusTrain_2["Pred_Difference_Final"] <- round(BusTrain_2$Pred_Difference, digits=0)
BusTest_2["Pred_Difference_Final"] <- round(BusTest_2$Pred_Difference, digits=0)
BusTrain_2["Difference_in_Prediction"] <- round(BusTrain_2$Difference_in_Prediction_temp, digits=0)
BusTest_2["Difference_in_Prediction"] <- round(BusTest_2$Difference_in_Prediction_temp, digits=0)

#output data
write.csv(BusTrain_2, "Outputs/Route_122_1_16/05. BusTrain_122_1_16_Cluster2.csv")
write.csv(BusTest_2, "Outputs/Route_122_1_16/06. BusTest_122_1_16_Cluster2.csv")

###################### End Cluster 2 ######################

###########################################################
################ All of Route 122 Stop 16 #################
###########################################################

# splitting data into training and test data sets for all data with no clusters
BusTrain_All <- select(filter(DublinBusData, Type =="Training"),c(1:20)) #training data
BusTest_All <- select(filter(DublinBusData, Type =="Test"),c(1:20)) #test data/validation

# checking record count to see if split was accurate
dim(DublinBusData)
dim(BusTrain_All)
dim(BusTest_All)

#attach data set to be used
attach(BusTrain_All)

# regression model
model <- fastLm(Difference ~ Scheduled_Arrival_Time + Temperature + 
                  Wind_Speed + Total_Rainfall + Sound_Level)

# Use model to predict Number of Journeys in the training set
BusTrain_All$Pred_Difference <- predict(model, newdata = BusTrain_All)

# Use model to predict Number of Journeys in the test set
BusTest_All$Pred_Difference <- predict(model, newdata = BusTest_All)

# See output of the Linear Regression model and the main components of it
summary(model)

# Check how well the model performs on the training set - correlation^2, RMSE and MAE
train.corr <- cor(BusTrain_All$Pred_Difference, BusTrain_All$Difference)
train.RMSE <- sqrt(mean((BusTrain_All$Pred_Difference - BusTrain_All$Difference)^2))
train.MAE <- mean(abs(BusTrain_All$Pred_Difference - BusTrain_All$Difference))
c(train.corr^2, train.RMSE, train.MAE)

# Check how well the model performs on the test set - correlation^2, RMSE and MAE
test.corr <- cor(BusTest_All$Pred_Difference, BusTest_All$Difference)
test.RMSE <- sqrt(mean((BusTest_All$Pred_Difference - BusTest_All$Difference)^2))
test.MAE <- mean(abs(BusTest_All$Pred_Difference - BusTest_All$Difference))
c(test.corr^2, test.RMSE, test.MAE)

# add in column to compare predicted value to actual value for each row
BusTrain_All["Difference_in_Prediction_temp"] <- BusTrain_All$Difference - BusTrain_All$Pred_Difference
BusTest_All["Difference_in_Prediction_temp"] <- BusTest_All$Difference - BusTest_All$Pred_Difference
# round the columns to ensure it is inline with original difference variable
BusTrain_All["Pred_Difference_Final"] <- round(BusTrain_All$Pred_Difference, digits=0)
BusTest_All["Pred_Difference_Final"] <- round(BusTest_All$Pred_Difference, digits=0)
BusTrain_All["Difference_in_Prediction"] <- round(BusTrain_All$Difference_in_Prediction_temp, digits=0)
BusTest_All["Difference_in_Prediction"] <- round(BusTest_All$Difference_in_Prediction_temp, digits=0)

#output data
write.csv(BusTrain_All, "Outputs/Route_122_1_16/07. BusTrain_122_1_16_All_Clusters.csv")
write.csv(BusTest_All, "Outputs/Route_122_1_16/08. BusTest_122_1_16_All_Clusters.csv")

###################### All of Route 122 Stop 16 ######################