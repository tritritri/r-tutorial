###############################
# BEGIN 01_prediction_basic.r #
###############################
#
# Goal: produce a simple prediction (in this case, regression) 
#

# read the data 
data_mtcars_train <- read.table("data_mtcars_train.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
data_mtcars_test  <- read.table("data_mtcars_test.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
# This data is about automobile design and performance, taken from R library datasets
# >> https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html
# Column 1 is the car_id (will be left out in the modeling process). 
# We will use column 2-11 as the (available) features and column 12 is the target variable

# import the library for Random Forest
# install.packages("e1071") # run this command if the package is not yet installed
library(randomForest)

# set the random seed for reproducibility
set.seed(1)

# train a simple Random Forest
# the 1st parameter is the features, i.e., column 2-11
# the 2nd parameter is the target variables, i.e., column 12
# to learn more about randomForest parameters, please run `?randomForest`
model <- randomForest(data_mtcars_train[ , 2:11], data_mtcars_train[ , 12], ntree=100)
# data_mtcars_train[ , 2:11] means that we access all rows, columns 2-11
# data_mtcars_train[ 1:100, c(2:11, 99) ] would mean rows 1-100 and columns 2-11 and column 99 

# predict! 
pred <- predict(model, newdata=data_mtcars_test) 

# write the prediction for each car_id in the test set into a file 
prediction <- data.frame(car_id=data_mtcars_test[ , 1], prediction=pred, stringsAsFactors=FALSE)
write.table(prediction, file="prediction_mtcars_1.csv", quote=TRUE, sep=",", row.names=FALSE)

#############################
# END 01_prediction_basic.r #
#############################






###################################
# BEGIN 02_using_validation_set.r #
###################################
#
# Goal: evaluate how good the model is. 
#   One way to do this is to divide the train set further into the train and validation set, and 
#   then evaluate the model on the validation set
#   This code also serves as the basis for the more robust model/features evaluation: cross validation
#   (see 03_cross_validation)


# first, define some evaluation function. 
# say, RMSE: root-mean-square error. The lower it is, the better.
rmse <- function(trueValues, predictions) {
  return( mean((trueValues-predictions)^2)^0.5 )
}
### end of rmse ###


# read the data
data_mtcars_train <- read.table("data_mtcars_train.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
data_mtcars_test  <- read.table("data_mtcars_test.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)


# divide the train set further into train and validation set
# let's say we take 20% of the train set (randomly) as the validation set
set.seed(1) # set the random seed for reproducibility
idxValidation <- sample(nrow(data_mtcars_train), 0.2*nrow(data_mtcars_train))
data_validation <- data_mtcars_train[idxValidation, ] # take the validation rows
data_train_reduced <- data_mtcars_train[-idxValidation, ] # take other than validation rows


### Model and feature evaluation starts here :) ###
library(randomForest) # import the library for Random Forest

# Step 1: build the model on the newly (reduced) train set
set.seed(1) # set the random seed for reproducibility
model <- randomForest(data_train_reduced[ , 2:11], data_train_reduced[ ,12], ntree=100)
# there are various model/feature combination, for example:
# model <- randomForest(data_train_reduced[ , 2:11], data_train_reduced[ ,12], ntree=50)
# model <- randomForest(data_train_reduced[ , 2:5], data_train_reduced[ ,12], ntree=50)
# model <- randomForest(data_train_reduced[ , c(2, 4:7, 10)], data_train_reduced[ ,12], ntree=100)

# Step 2: predict the validation set 
pred <- predict(model, newdata=data_validation[ , 2:11]) 

# Step 3: evaluate the prediction and print the score
print(rmse(data_validation[ , 12], pred))

### Take note of the current score (and the model/features used to produce this score) ###
### Repeat step 1 to 3 using different model/feature until you are satisfied with the score ###
### Note: some people do this model/feature selection process automatically. But we leave that out for now :)


# after you are happy with the model, let us write the prediction into a file
# say that the best model you found is randomForest(data_train_reduced[ , 2:5], data_train_reduced[ ,12], ntree=100)

# apply the model on the full train set
set.seed(1) # set the random seed for reproducibility
model <- randomForest(data_mtcars_train[ , 2:11], data_mtcars_train[ , 12], ntree=50)

# predict! 
pred <- predict(model, newdata=data_mtcars_test) 

# write the prediction for each car_id in the test set into a file 
prediction <- data.frame(car_id=data_mtcars_test[,1], prediction=pred, stringsAsFactors=FALSE)
write.table(prediction, file="prediction_mtcars_2.csv", quote=TRUE, sep=",", row.names=FALSE)

#################################
# END 02_using_validation_set.r #
#################################












###############################
# BEGIN 03_cross_validation.r #
###############################

# Motivation: 
#   In 02_using_validation_set, when we select the best model/features based on the scores 
#     of the validation set, it could be that we overfit the validation set. That is, we pick the 
#     a model that works best only for that specific set, but perform badly on the other set that 
#     it has not seen before. 
#   However, we would like to have a model that perform well in general. Because in a real system, 
#     most of the times the model will have to predict instances that it has not seen before. And, 
#     we still expect the model to perform better in that situation. Well... if the task of the 
#     model is only to predict the instances that it has seen before, then the best solution is to 
#     simply memorized the historical data. But, we are not gonna do that :)
#   In summary, the model should perform well not only on the instances that it has seen before 
#     (i.e., in the train and validation set), but also on the set that it has never seen before. 
#     (i.e., the *real* test set).
#
# Goal of this code: 
#   Evaluate a model using cross validation (more robust evaluation compared to using one 
#   validation set).
#
# Short description about cross validation:     
#   In an n-fold cross validation, we divide the train set into n chunks (preferrably randomly, 
#   unless the instances are time/sequence-dependent). Use 1 chunk as the validation set and then 
#   train the algorithm with the rest of the data. We repeat this process n times, using every 
#   chunk at most once as the validation set.
#   This will result in n evaluations, which produces n error scores. The final score of the model 
#   is then defined as the average of these n error scores.
#


# create the evaluation function. In this case: RMSE (root-mean-square error). The lower the better.
rmse <- function(trueValues, predictions) {
  return( mean((trueValues-predictions)^2)^0.5 )
}

library(randomForest) # import the library for Random Forest

# read the data
data_mtcars_train <- read.table("data_mtcars_train.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)
data_mtcars_test  <- read.table("data_mtcars_test.csv", header=TRUE, sep=",", stringsAsFactors=FALSE)

# create a random index
set.seed(1) # set the random seed for reproducibility
idxValidation <- sample(nrow(data_mtcars_train), nrow(data_mtcars_train))


### CROSS VALIDATION STARTS HERE ... ###
# let us try 5-fold cross validation 
nFolds <- 5
chunkSize <- nrow(data_mtcars_train)/nFolds
# create a vector to s
errorScore <- vector() 
for (i in 1:nFolds) {
  # build the index for the validation set for this i-th chunks
  idxValEnd <- i*chunkSize
  idxValStart <- idxValEnd-chunkSize

  # divide the full train set into the validation and the reduced train set
  data_validation <- data_mtcars_train[ idxValStart:idxValEnd, ]
  data_train_reduced <- data_mtcars_train [ -(idxValStart:idxValEnd), ]

  # build the model on the newly (reduced) train set
  set.seed(1) # set the random seed for reproducibility
  model <- randomForest(data_train_reduced[ , 2:11], data_train_reduced[ , 12], ntree=100)
  # there are also other various model/feature combination, for example:
  # model <- randomForest(data_train_reduced[ , 2:11], data_train_reduced[ ,12], ntree=50)
  # model <- randomForest(data_train_reduced[ , 2:7], data_train_reduced[ ,12], ntree=50)
  # model <- randomForest(data_train_reduced[ , c(2, 4:7, 10)], data_train_reduced[ ,12], ntree=100)

  # predict the validation set 
  pred <- predict(model, newdata=data_validation[ , 2:11]) 

  # evaluate the prediction and print the score
  errorScore[i] <- rmse(data_validation[ , 12], pred)
}

# print the final score of the model
print(mean(errorScore))

# you can then repeat this cross validation multiple times (starting from the line 
# `CROSS VALIDATION STARTS HERE...` ) using different model/features combination to choose the best
# one.
# It might be better to create a function out of this cross validation process, and pass the 
# model/features as the parameters. That way, you can run the cross validation process in one line 
# (i.e., by calling your function) But, we leave that out for now. As long as you are careful in 
# executing it, everything should be fine :)
#
### END OF CROSS VALIDATION ###


# Let's say after doing some cross validation, we choose: 
#   randomForest(data_train_reduced[ , 2:7], data_train_reduced[ ,12], ntree=50)
# Now, we apply the model to the full train set and predict the test set

set.seed(1) # set the random seed for reproducibility
model <- randomForest(data_mtcars_train[ , 2:7], data_mtcars_train[ , 12], ntree=50)

# predict! 
pred <- predict(model, newdata=data_mtcars_test) 

# write the prediction for each car_id in the test set into a file 
prediction <- data.frame(car_id=data_mtcars_test[,1], prediction=pred, stringsAsFactors=FALSE)
write.table(prediction, file="prediction_mtcars_3.csv", quote=TRUE, sep=",", row.names=FALSE)

#############################
# END 03_cross_validation.r #
#############################






###########################
# BEGIN 04_other_models.r #
###########################

### RANDOM FOREST ###
# just to refresh, here is how we did randomForest:
library(randomForest)
# other than ntree parameter, you might also want to tune other parameters, such as nodesize
set.seed(1)
model <- randomForest(data_mtcars_train[ , 2:11], data_mtcars_train[ , 12], ntree=100, nodesize=10)
pred <- predict(model, newdata=data_mtcars_test) 
prediction <- data.frame(car_id=data_mtcars_test[ , 1], prediction=pred, stringsAsFactors=FALSE)
# end of random forest #


### LINEAR MODEL ###
# Now, let's use linear model. You can replace the lines above with: 
# we spell out the feature one by one
set.seed(1)
model <- lm(mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear + carb , data=data_mtcars_train)
pred <- predict(model, newdata=data_mtcars_test) 
prediction <- data.frame(car_id=data_mtcars_test[ , 1], prediction=pred, stringsAsFactors=FALSE)
# end of linear model #


### SVM ###
# install.packages("e1071") # run this line if the package is not yet installed
library(e1071)
# there are lots of parameters to be tuned here, e.g., kernel, degree, gamma, cost, num, tolerance, epsilon
set.seed(1)
model <- svm(mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear + carb , data=data_mtcars_train)
pred <- predict(model, newdata=data_mtcars_test) 
prediction <- data.frame(car_id=data_mtcars_test[ , 1], prediction=pred, stringsAsFactors=FALSE)
# end of SVM #


### XGBOOST ###
library(xgboost)
set.seed(1)
h<-sample(nrow(data_mtcars_train),0.1*nrow(data_mtcars_train))
dval<-xgb.DMatrix(data=data.matrix(data_mtcars_train[h, 2:11]), label=data_mtcars_train[h, 12])
dtrain<-xgb.DMatrix(data=data.matrix(data_mtcars_train[-h, 2:11]), label=data_mtcars_train[-h, 12])
model <- xgb.train( objective           = "reg:linear", 
                    booster             = "gbtree",
                    data                = dtrain, 
                    watchlist           = list(val=dval, train=dtrain),
                    nrounds             = 1000, # max 1000 rounds
                    early.stop.round    = 100,  # stop after 100 rounds without better performance
                    maximize            = FALSE # we aim to minimize the error
                    )
pred <- predict(model, data.matrix(data_mtcars_test[,2:11]))
prediction <- data.frame(car_id=data_mtcars_test[ , 1], prediction=pred, stringsAsFactors=FALSE)
### end of XGBOOST ###

# These three models: lm, svm, and xgboost are shown here to provide an example on how to adapt the 
# our code above to another models. There are still lots of models out there :)

#########################
# END 04_other_models.r #
######################$##










#####################################
# 5. MISSING VALUES AND OTHER STUFF #
#####################################


### PROBLEM WITH CATEGORICAL FEATURES WHEN USING RANDOM FOREST ###
# There are two types of features: 
# - numeric: such as hotel price, booking amount
# - categorical: where there is no particular meaningful ordering, such as color, country_name, etc
#
# randomForest typically does not have problem with numerical features. However, it might have 
# problem with a categorical feature when it has too many distinct values. Note that, randomForest 
# treats a column as categorical when the class of the column is 'factor' or 'character'
#
# So, if you have problem training a randomForest model because the number of distinct values in 
# one of your categorical feature is too high, then one trick to convert the values into numeric. 
# This will make randomForest treat this feature as a numeric. It is a solution, but might not be a 
# perfect one :)
#
# Assuming 'stringsAsFactors=FALSE' is used when loading CSV, then all text columns are treated as 
# `character`. Here is a sample code to change the categorical feature into a numeric ids, assuming 
# your data is stored in dataTrainWithCategorical
for (i in 1:ncol(dataTrainWithCategorical)) {
  if (class(dataTrainWithCategorical[ , i]) == "character") {
    levels <- unique(dataTrainWithCategorical[ , i])
    dataTrainWithCategorical[ , i] <- as.integer(factor(dataTrainWithCategorical[ , i], levels=levels))
  }
}
#
# Further discussion about converting categorical feature into numeric: 
# From a categorical feature, one could create new numerical features that represent the 
# characteristics of the categorical feature. For example:
#   - for color: we could produce a new feature 'the number of people who like this color' 
#   - for country_name: we could produce new features: 'country_population', 'country_GDP'  
##################################################################


### PROBLEM WITH MISSING VALUES ###
# Some learning functions can deal with missing values automatically (or usually by setting some 
# parameter, such as `na.action`). Some cannot. 
# 
# Then, a solution would be to replace the missing value with some other value, e.g., 0, -1, the 
# average of something, or even a prediction :)
# Here is an example to replace the missing values (NA or NaN) with 0, assuming your data is 
# stored in dataTrainWithMissing
#
for (i in 1:ncol(dataTrainWithMissing)) {
  if (sum(is.na(dataTrainWithMissing[ , i])) > 0 ) {    
    dataTrainWithMissing[ , i][which(is.na(dataTrainWithMissing[ , i]))] <- 0
  }
  if (sum(is.nan(dataTrainWithMissing[ , i])) > 0 ) {    
    dataTrainWithMissing[ , i][which(is.nan(dataTrainWithMissing[ , i]))] <- 0
  }
}
###################################
