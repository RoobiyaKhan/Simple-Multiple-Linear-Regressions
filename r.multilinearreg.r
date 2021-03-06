# multiple linear regression
# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#fitting multiple linear regression to the training set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,data = training_set)
regressor = lm(formula = Profit ~ ., 
               data = training_set) #. means all independent variables
summary(regressor) #since r.d.spend is only significant,we can write the formula as profit~R.D.Spend,there 
                   #wont be any diff in results aswellas y_pred
#predicting the test set results
y_pred = predict(regressor, newdata = test_set)
y_pred

#building the optimal model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset) #we can also use training_Set instead
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset) #we can also use training_Set instead
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset) #we can also use training_Set instead
summary(regressor)
