
# 1. Describe the problem, data, and goal(s). Literature review if appropriate.
# 2. Load all packages at once that will be used (lines 11 - 22)
# 3. Input data (lines 25 - 37)
# 4. Full exploratory data analysis (lines 40 - 138)
# 5. Initialize values to 0
# 6. Set up loops for data resampling
# 5. Data splitting train (60%) and test (40%) (lines 141-144)
# 6. Fit individual and ensemble models using the training data (lines 146 - 202)
# 7. Predict and evaluate on unseen data using the testing and validation data (line 207 - 260)
# 10. Forecasts and data visualizations for current time and 1-month change data for the #1 model results
# 11. Strongest evidence based recommendations and levels of confidence/confidence intervals
# 12. Future work, still open questions, possible reasons for errors, etc.

#### <------------------------ 1. Describe the problem, data, and goals ----------------------->####


#### <------------------- 2. Load all packages at once in alphabetical order ------------------>####

library(adabag)
library(beans)
library(C50)
library(class) # for K-Nearest Neighbors
library(dplyr)
library(e1071) # for Support vector machines
library(gbm) # For boosted models
library(glmnet) # for ridge regression
library(gt)
library(gtExtras)
library(ipred) # for bagging cart
library(ISLR)
library(kernlab) # for 
library(klaR) # for regularized discriminant analysis
library(MASS) # for linear discriminant analysis
library(mda) 
library(mltools) # for one-hot encoding
library(modelr) # for k-fold cross-validation
library(naivebayes) # for Naive Bayes
library(neuralnet) # for Neural Networks
library(pls) # for Principal Components Regression
library(randomForest) # for random forests
library(ranger) # for the Ranger model
library(reactable) # for creating the summary table at the end of the analysis
library(tidyverse)
library(tree) # for decision tree

#### <------------------- 3. Input all the data------------------>####

set.seed(31415) # to make results exactly reproducible

# Get the data

df <- as.data.frame(beans::beans)
df <- df %>% dplyr::mutate('bean_name' = class) %>% 
  relocate(bean_name, .after = last_col()) %>% 
  dplyr::select(-class)
df <- df[sample(nrow(df)),] # To randomize the rows


#### Numerical values against bean_name ####
df %>%
  mutate(across(-bean_name, as.numeric)) %>%
  pivot_longer(-bean_name, names_to = "var", values_to = "value") %>%
  ggplot(aes(x = bean_name, y = value)) +
  geom_col() +
  facet_wrap(~var, scales = "free") +
  labs(title="Numerical values against bean_name")

#### Summary of the dataset ####
summary(df)

#### Data dictionary ####
str(df)

#### Correlation plot of numeric data ####
df1 <- df %>% purrr::keep(is.numeric)
M1 = cor(df1)
title = "Correlation plot of the numerical data"
corrplot::corrplot(M1, method = 'number', title = title, mar=c(0,0,1,0)) # http://stackoverflow.com/a/14754408/54964)
corrplot::corrplot(M1, method = 'circle', title = title, mar=c(0,0,1,0)) # http://stackoverflow.com/a/14754408/54964)

#### Print correlation matrix of numeric data ####
print(M1)

#### Pariwise scatter plot ####
pairs(df)

#### Boxplots of the numeric data ####
df1 %>%
  gather(key = "var", value = "value") %>%
  ggplot(aes(x = '',y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~ var, scales = "free") +
  theme_bw() +
  labs(title = "Boxplots of the numeric data")
#Thanks to https://rstudio-pubs-static.s3.amazonaws.com/388596_e21196f1adf04e0ea7cd68edd9eba966.html

#### Histograms of the numeric data ####
df1 <- df %>% select_if(is.numeric)
ggplot(gather(df1, cols, value), aes(x = value)) +
  geom_histogram(bins = round(nrow(df1)/10)) +
  facet_wrap(.~cols, scales = "free") +
  labs(title = "Histograms of each numeric column. Each bar = 10 rows of data")


#### Set accuracy values to zero ####
adabag_overfitting <- 0
adaboost_train_accuracy <- 0
adaboost_test_accuracy <- 0
adaboost_validation_accuracy <- 0
adaboost_overfitting <- 0
adaboost_duration <- 0
bagging_train_accuracy <- 0
bagging_test_accuracy <- 0
bagging_validation_accuracy <- 0
bagging_overfitting <- 0
bag_cart_train_accuracy <- 0
bag_cart_test_accuracy <- 0
bag_cart_validation_accuracy <- 0
bag_cart_overfitting <- 0
bag_cart_duration <- 0
bag_rf_train_accuracy <- 0
bag_rf_test_accuracy <- 0
bag_rf_validation_accuracy <- 0
bag_rf_overfitting <- 0
bag_rf_duration <- 0
bagging_duration <- 0
bag_rf_train_accuracy <- 0
C50_train_accuracy <- 0
C50_test_accuracy <- 0
C50_validation_accuracy <- 0
C50_overfitting <- 0
C50_duration <- 0
earth_train_accuracy <- 0
earth_test_accuracy <- 0
earth_validation_accuracy <- 0
earth_test_accuracy_mean <- 0
earth_validation_accuracy_mean <- 0
earth_overfitting <- 0
earth_duration <- 0
fda_train_accuracy <- 0
fda_test_accuracy <- 0
fda_test_accuracy_mean <- 0
fda_validation_accuracy <- 0
fda_validation_accuracy_mean <- 0
fda_overfitting <- 0
fda_duration <- 0
knn_train_accuracy_mean <- 0
knn_train_accuracy <- 0
knn_test_accuracy <- 0
knn_test_accuracy_mean <- 0
knn_validation_accuracy <- 0
knn_validation_accuracy_mean <- 0
knn_overfitting <- 0
knn_duration <- 0
lssvm_train_accuracy <- 0
lssvm_train_accuracy_mean <- 0
lssvm_validation_accuracy <- 0
lssvm_test_accuracy <- 0
lssvm_test_accuracy_mean <- 0
lssvm_validation_accuracy_mean <- 0
lssvm_overfitting <- 0
lssvm_duration <- 0
lda_train_accuracy <- 0
lda_train_accuracy_mean <- 0
lda_validation_accuracy <- 0
lda_test_accuracy <- 0
lda_test_accuracy_mean <- 0
lda_validation_accuracy_mean <- 0
lda_overfitting <- 0
lda_duration <- 0
linear_train_accuracy <- 0
linear_validation_accuracy <- 0
linear_test_accuracy <- 0
linear_test_accuracy_mean <- 0
linear_validation_accuracy_mean <- 0
linear_overfitting <- 0
linear_duration <- 0
mda_validation_accuracy <- 0
mda_train_accuracy <- 0
mda_test_accuracy <- 0
mda_test_accuracy_mean <- 0
mda_validation_accuracy_mean <- 0
mda_overfitting <- 0
mda_duration <- 0
n_bayes_train_accuracy <- 0
n_bayes_test_accuracy <- 0
n_bayes_validation_accuracy <- 0
n_bayes_test_accuracy_mean <- 0
n_bayes_validation_accuracy_mean <- 0
n_bayes_overfitting <- 0
n_bayes_duration <- 0
qda_train_accuracy <- 0
qda_test_accuracy <- 0
qda_test_accuracy_mean <- 0
qda_validation_accuracy_mean <- 0
qda_overfitting <- 0
qda_duration <- 0
pls_train_accuracy <- 0
pls_test_accuracy <- 0
pls_test_accuracy_mean <- 0
pls_validation_accuracy <- 0
pls_validation_accuracy_mean <- 0
pls_overfitting <- 0
pls_duration <- 0
pda_train_accuracy <- 0
pda_test_accuracy <- 0
pda_test_accuracy_mean <- 0
pda_validation_accuracy <- 0
pda_validation_accuracy_mean <- 0
pda_overfitting <- 0
pda_duration <- 0
qda_train_accuracy <- 0
qda_validation_accuracy <- 0
qda_test_accuracy <- 0
qda_test_accuracy_mean <- 0
qda_validation_accuracy <- 0
qda_validation_accuracy_mean <- 0
qda_overfitting <- 0
qda_duration <- 0
rf_train_accuracy <- 0
rf_test_accuracy <- 0
rf_test_accuracy_mean <- 0
rf_validation_accuracy <- 0
rf_validation_accuracy_mean <- 0
rf_overfitting <- 0
rf_duration <- 0
ranger_train_accuracy <- 0
ranger_test_accuracy <- 0
ranger_test_accuracy_mean <- 0
ranger_validation_accuracy <- 
ranger_validation_accuracy_mean <- 0
ranger_overfitting <- 0
ranger_duration <- 0
rda_train_accuracy <- 0
rda_test_accuracy <- 0
rda_test_accuracy_mean <- 0
rda_validation_accuracy <- 0
rda_validation_accuracy_mean <- 0
rda_overfitting <- 0
rda_duration <- 0
rpart_train_accuracy <- 0
rpart_test_accuracy <- 0
rpart_test_accuracy_mean <- 0
rpart_validation_accuracy <- 0
rpart_validation_accuracy_mean <- 0
rpart_overfitting <- 0
rpart_duration <- 0
svm_train_accuracy <- 0
svm_test_accuracy <- 0
svm_test_accuracy_mean <- 0
svm_validation_accuracy <- 0
svm_validation_accuracy_mean <- 0
svm_overfitting <- 0
svm_duration <- 0
tree_train_accuracy <- 0
tree_test_accuracy <- 0
tree_test_accuracy_mean <- 0
tree_validation_accuracy <- 0
tree_validation_accuracy_mean <- 0
tree_overfitting <- 0
tree_duration <- 0

ensemble_adabag_train_accuracy <- 0
ensemble_adabag_train_accuracy_mean <- 0
ensemble_adabag_test_accuracy <- 0
ensemble_adabag_test_accuracy_mean <- 0
ensemble_adabag_validation_accuracy <- 0
ensemble_adabag_validation_accuracy_mean <- 0
ensemble_adabag_overfitting <- 0
ensemble_adabag_duration <- 0

ensemble_adaboost_train_accuracy <- 0
ensemble_adaboost_train_accuracy_mean <- 0
ensemble_adaboost_test_accuracy <- 0
ensemble_adaboost_test_accuracy_mean <- 0
ensemble_adaboost_validation_accuracy <- 0
ensemble_adaboost_validation_accuracy_mean <- 0
ensemble_adaboost_overfitting <- 0
ensemble_adaboost_duration <- 0

ensemble_bag_cart_train_accuracy <- 0
ensemble_bag_cart_train_accuracy_mean <- 0
ensemble_bag_cart_test_accuracy <- 0
ensemble_bag_cart_test_accuracy_mean <- 0
ensemble_bag_cart_validation_accuracy <- 0
ensemble_bag_cart_validation_accuracy_mean <- 0
ensemble_bag_cart_overfitting <- 0
ensemble_bag_cart_duration <- 0

ensemble_bag_rf_train_accuracy <- 0
ensemble_bag_rf_train_accuracy_mean <- 0
ensemble_bag_rf_test_accuracy <- 0
ensemble_bag_rf_test_accuracy_mean <- 0
ensemble_bag_rf_validation_accuracy <- 0
ensemble_bag_rf_validation_accuracy_mean <- 0
ensemble_bag_rf_overfitting <- 0
ensemble_bag_rf_duration <- 0

ensemble_C50_train_accuracy <- 0
ensemble_C50_train_accuracy_mean <- 0
ensemble_C50_test_accuracy <- 0
ensemble_C50_test_accuracy_mean <- 0
ensemble_C50_validation_accuracy <- 0
ensemble_C50_validation_accuracy_mean <- 0
ensemble_C50_overfitting <- 0
ensemble_C50_duration <- 0

ensemble_earth_train_accuracy <- 0
ensemble_earth_train_accuracy_mean <- 0
ensemble_earth_test_accuracy <- 0
ensemble_earth_test_accuracy_mean <- 0
ensemble_earth_validation_accuracy <- 0
ensemble_earth_validation_accuracy_mean <- 0
ensemble_earth_overfitting <- 0
ensemble_earth_duration <- 0

ensemble_lssvm_train_accuracy <- 0
ensemble_lssvm_train_accuracy_mean <- 0
ensemble_lssvm_test_accuracy <- 0
ensemble_lssvm_test_accuracy_mean <- 0
ensemble_lssvm_validation_accuracy <- 0
ensemble_lssvm_validation_accuracy_mean <- 0
ensemble_lssvm_overfitting <- 0
ensemble_lssvm_duration <- 0

ensemble_n_bayes_train_accuracy <- 0
ensemble_n_bayes_train_accuracy_mean <- 0
ensemble_n_bayes_test_accuracy <- 0
ensemble_n_bayes_test_accuracy_mean <- 0
ensemble_n_bayes_validation_accuracy <- 0
ensemble_n_bayes_validation_accuracy_mean <- 0
ensemble_n_bayes_overfitting <- 0
ensemble_n_bayes_duration <- 0

ensemble_ranger_train_accuracy <- 0
ensemble_ranger_train_accuracy_mean <- 0
ensemble_ranger_test_accuracy <- 0
ensemble_ranger_test_accuracy_mean <- 0
ensemble_ranger_validation_accuracy <- 0
ensemble_ranger_validation_accuracy_mean <- 0
ensemble_ranger_overfitting <- 0
ensemble_ranger_duration <- 0

ensemble_rf_train_accuracy <- 0
ensemble_rf_train_accuracy_mean <- 0
ensemble_rf_test_accuracy <- 0
ensemble_rf_test_accuracy_mean <- 0
ensemble_rf_validation_accuracy <- 0
ensemble_rf_validation_accuracy_mean <- 0
ensemble_rf_overfitting <- 0
ensemble_rf_duration <- 0

ensemble_rda_train_accuracy <- 0
ensemble_rda_train_accuracy_mean <- 0
ensemble_rda_test_accuracy <- 0
ensemble_rda_test_accuracy_mean <- 0
ensemble_rda_validation_accuracy <- 0
ensemble_rda_validation_accuracy_mean <- 0
ensemble_rda_overfitting <- 0
ensemble_rda_duration <- 0

ensemble_svm_train_accuracy <- 0
ensemble_svm_train_accuracy_mean <- 0
ensemble_svm_test_accuracy <- 0
ensemble_svm_test_accuracy_mean <- 0
ensemble_svm_validation_accuracy <- 0
ensemble_svm_validation_accuracy_mean <- 0
ensemble_svm_overfitting <- 0
ensemble_svm_duration <- 0

ensemble_tree_train_accuracy <- 0
ensemble_tree_train_accuracy_mean <- 0
ensemble_tree_test_accuracy <- 0
ensemble_tree_test_accuracy_mean <- 0
ensemble_tree_validation_accuracy <- 0
ensemble_tree_validation_accuracy_mean <- 0
ensemble_tree_overfitting <- 0
ensemble_tree_duration <- 0


#### Break into train (60%), test (20%) and validation (20%) ####
for (i in 1:5){
  index <- sample(c(1:3), nrow(df), replace=TRUE, prob=c(0.6, 0.2, 0.2))
  
  train  <- df[index == 1, ]
  test   <- df[index == 2, ]
  validation = df[index == 3,]
  
  train01 <- train # needed to run xgboost
  test01 <- test # needed to run xgboost
  validation01 <- validation
  
  y_train <- train$bean_name
  y_test <- test$bean_name
  y_validation <- validation$bean_name
  
  train  <- df[index == 1, ] %>% dplyr::select(-bean_name)
  test   <- df[index == 2, ] %>% dplyr::select(-bean_name)
  validation <- df[index == 3, ] %>% dplyr::select(-bean_name)
  
  #### adaboost ####
  adaboost_start <- Sys.time()
  adaboost_train_fit <- MachineShop::fit(formula = bean_name ~ ., data = train01, model = "AdaBoostModel")
  adaboost_train_pred <- predict(object = adaboost_train_fit, newdata = train)
  adaboost_train_table <- table(adaboost_train_pred, y_train)
  adaboost_train_accuracy[i] <- sum(diag(adaboost_train_table)) / sum(adaboost_train_table)
  adaboost_train_accuracy_mean <- mean(adaboost_train_accuracy)
  adaboost_train_mean <- mean(diag(adaboost_train_table)) / mean(adaboost_train_table)
  adaboost_train_sd <- sd(diag(adaboost_train_table)) / sd(adaboost_train_table)
  adaboost_train_diag <- sum(diag(adaboost_train_table))
  sum_diag_train_adaboost <- sum(diag(adaboost_train_table))
  adaboost_train_prop <- diag(prop.table(adaboost_train_table, margin = 1))
  
  adaboost_test_pred <- predict(object = adaboost_train_fit, newdata = test)
  adaboost_test_table <- table(adaboost_test_pred, y_test)
  adaboost_test_accuracy[i] <- sum(diag(adaboost_test_table)) / sum(adaboost_test_table)
  adaboost_test_accuracy_mean <- sum(adaboost_test_accuracy)/length(adaboost_test_accuracy)
  adaboost_test_mean <- mean(diag(adaboost_test_table)) / mean(adaboost_test_table)
  adaboost_test_sd <- sd(diag(adaboost_test_table)) / sd(adaboost_test_table)
  adaboost_test_diag <- sum(diag(adaboost_test_table))
  sum_diag_test_adaboost <- sum(diag(adaboost_test_table))
  adaboost_test_prop <- diag(prop.table(adaboost_test_table, margin = 1))
  
  adaboost_validation_pred <- predict(object = adaboost_train_fit, newdata = validation)
  adaboost_validation_table <- table(adaboost_validation_pred, y_validation)
  adaboost_validation_accuracy[i] <- sum(diag(adaboost_validation_table)) / sum(adaboost_validation_table)
  adaboost_validation_accuracy_mean <- sum(adaboost_validation_accuracy)/length(adaboost_validation_accuracy)
  adaboost_validation_mean <- mean(diag(adaboost_validation_table)) / mean(adaboost_validation_table)
  adaboost_validation_sd <- sd(diag(adaboost_validation_table)) / sd(adaboost_validation_table)
  adaboost_validation_diag <- sum(diag(adaboost_validation_table))
  sum_diag_validation_adaboost <- sum(diag(adaboost_validation_table))
  adaboost_validation_prop <- diag(prop.table(adaboost_validation_table, margin = 1))
  
  adaboost_holdout_mean <- mean(c(adaboost_test_accuracy_mean, adaboost_validation_accuracy_mean))
  adaboost_overfitting[i] <- adaboost_holdout_mean / adaboost_train_accuracy_mean
  adaboost_overfitting_mean <- mean(adaboost_overfitting)
  adaboost_overfitting_range <- range(adaboost_overfitting)
  
  adaboost_table <- adaboost_test_table +adaboost_validation_table
  adaboost_end <- Sys.time()
  adaboost_duration[i] <- adaboost_end - adaboost_start
  adaboost_duration_mean <- mean(adaboost_duration)

  
  #### Bagging ####
  bagging_start <- Sys.time()
  bagging_train_fit <- ipred::bagging(bean_name ~ ., data = train01, coob = TRUE)
  bagging_train_pred <- predict(object = bagging_train_fit, newdata = train)
  bagging_train_table <- table(bagging_train_pred, y_train)
  bagging_train_accuracy[i] <- sum(diag(bagging_train_table)) / sum(bagging_train_table)
  bagging_train_accuracy_mean <- mean(bagging_train_accuracy)
  bagging_train_mean <- mean(diag(bagging_train_table)) / mean(bagging_train_table)
  bagging_train_sd <- sd(diag(bagging_train_table)) / sd(bagging_train_table)
  bagging_train_diag <- sum(diag(bagging_train_table))
  sum_diag_train_bagging <- sum(diag(bagging_train_table))
  bagging_train_prop <- diag(prop.table(bagging_train_table))
  
  bagging_test_pred <- predict(object = bagging_train_fit, newdata = test)
  bagging_test_table <- table(bagging_test_pred, y_test)
  bagging_test_accuracy[i] <- sum(diag(bagging_test_table)) / sum(bagging_test_table)
  bagging_test_accuracy_mean <- mean(bagging_test_accuracy)
  bagging_test_mean <- mean(diag(bagging_test_table)) / mean(bagging_test_table)
  bagging_test_sd <- sd(diag(bagging_test_table)) / sd(bagging_test_table)
  bagging_test_diag <- sum(diag(bagging_test_table))
  sum_diag_test_bagging <- sum(diag(bagging_test_table))
  bagging_test_prop <- diag(prop.table(bagging_test_table))
  
  bagging_validation_pred <- predict(object = bagging_train_fit, newdata = validation)
  bagging_validation_table <- table(bagging_validation_pred, y_validation)
  bagging_validation_accuracy[i] <- sum(diag(bagging_validation_table)) / sum(bagging_validation_table)
  bagging_validation_accuracy_mean <- mean(bagging_validation_accuracy)
  bagging_validation_mean <- mean(diag(bagging_validation_table)) / mean(bagging_validation_table)
  bagging_validation_sd <- sd(diag(bagging_validation_table)) / sd(bagging_validation_table)
  bagging_validation_diag <- sum(diag(bagging_validation_table))
  sum_diag_validation_bagging <- sum(diag(bagging_validation_table))
  bagging_validation_prop <- diag(prop.table(bagging_validation_table))
  
  bagging_holdout_mean <- mean(c(bagging_test_accuracy_mean, bagging_validation_accuracy_mean))
  bagging_overfitting[i] <- bagging_holdout_mean / bagging_train_accuracy_mean
  bagging_overfitting_mean <- mean(bagging_overfitting)
  bagging_overfitting_range <- range(bagging_overfitting)
  
  bagging_table <- bagging_test_table + bagging_validation_table
  
  bagging_end <- Sys.time()
  bagging_duration[i] <- bagging_end - bagging_start
  bagging_duration_mean <- mean(bagging_duration)
  
  
  #### Bagged CART ####
  bag_cart_start <- Sys.time()
  
  bag_cart_train_fit <- bagging(bean_name ~ ., data = train01)
  bag_cart_train_pred <- predict(bag_cart_train_fit, train)
  bag_cart_train_table <- table(bag_cart_train_pred$class, y_train)
  bag_cart_train_accuracy[i] <- sum(diag(bag_cart_train_table)) / sum(bag_cart_train_table)
  bag_cart_train_accuracy_mean <- mean(bag_cart_train_accuracy)
  bag_cart_train_mean <- mean(diag(bag_cart_train_table)) / mean(bag_cart_train_table)
  bag_cart_train_sd <- sd(diag(bag_cart_train_table)) / sd(bag_cart_train_table)
  sum_diag_bag_train_cart <- sum(diag(bag_cart_train_table))
  bag_cart_train_prop <- diag(prop.table(bag_cart_train_table, margin = 1))
  
  bag_cart_test_pred <- predict(bag_cart_train_fit, test)
  bag_cart_test_table <- table(bag_cart_test_pred$class, y_test)
  bag_cart_test_accuracy[i] <- sum(diag(bag_cart_test_table)) / sum(bag_cart_test_table)
  bag_cart_test_accuracy_mean <- mean(bag_cart_test_accuracy)
  bag_cart_test_mean <- mean(diag(bag_cart_test_table)) / mean(bag_cart_test_table)
  bag_cart_test_sd <- sd(diag(bag_cart_test_table)) / sd(bag_cart_test_table)
  sum_diag_bag_test_cart <- sum(diag(bag_cart_test_table))
  bag_cart_test_prop <- diag(prop.table(bag_cart_test_table, margin = 1))
  
  bag_cart_validation_pred <- predict(bag_cart_train_fit, validation)
  bag_cart_validation_table <- table(bag_cart_validation_pred$class, validation01$bean_name)
  bag_cart_validation_accuracy[i] <- sum(diag(bag_cart_validation_table)) / sum(bag_cart_validation_table)
  bag_cart_validation_accuracy_mean <- mean(bag_cart_validation_accuracy)
  bag_cart_validation_mean <- mean(diag(bag_cart_validation_table)) / mean(bag_cart_validation_table)
  bag_cart_validation_sd <- sd(diag(bag_cart_validation_table)) / sd(bag_cart_validation_table)
  sum_diag_bag_validation_cart <- sum(diag(bag_cart_validation_table))
  bag_cart_validation_prop <- diag(prop.table(bag_cart_validation_table, margin = 1))
  
  bag_cart_holdout_mean <- mean(c(bag_cart_test_accuracy_mean, bag_cart_validation_accuracy_mean))
  bag_cart_overfitting[i] <- bag_cart_holdout_mean / bag_cart_train_accuracy_mean
  bag_cart_overfitting_mean <- mean(bag_cart_overfitting)
  bag_cart_overfitting_range <- range(bag_cart_overfitting)
  
  bag_cart_table <- bag_cart_test_table + bag_cart_validation_table
  
  bag_cart_end <- Sys.time()
  bag_cart_duration[i] <- bag_cart_end - bag_cart_start
  bag_cart_duration_mean <- mean(bag_cart_duration)
  
  
  #### Bagged Random Forest model ####
  bag_rf_start <- Sys.time()
  
  bag_train_rf <- randomForest(y_train ~ ., data = train, mtry = ncol(train)-1)
  bag_rf_train_pred <- predict(bag_train_rf, train, type = "class")
  bag_rf_train_table <- table(bag_rf_train_pred, train01$bean_name)
  bag_rf_train_accuracy[i] <- sum(diag(bag_rf_train_table)) / sum(bag_rf_train_table)
  bag_rf_train_accuracy_mean <- mean(bag_rf_train_accuracy)
  bag_rf_train_diag <- sum(diag(bag_rf_train_table))
  bag_rf_train_mean <- mean(diag(bag_rf_train_table)) / mean(bag_rf_train_table)
  bag_rf_train_sd <- sd(diag(bag_rf_train_table)) / sd(bag_rf_train_table)
  sum_bag_train_rf <- sum(diag(bag_rf_train_table))
  bag_rf_train_prop <- diag(prop.table(bag_rf_train_table, margin = 1))
  
  bag_rf_test_pred <- predict(bag_train_rf, test, type = "class")
  bag_rf_test_table <- table(bag_rf_test_pred, test01$bean_name)
  bag_rf_test_accuracy[i] <- sum(diag(bag_rf_test_table)) / sum(bag_rf_test_table)
  bag_rf_test_accuracy_mean <- mean(bag_rf_test_accuracy)
  sum_diag_test_bag_rf <- sum(diag(bag_rf_test_table))
  bag_rf_test_mean <- mean(diag(bag_rf_test_table)) / mean(bag_rf_test_table)
  bag_rf_test_sd <- sd(diag(bag_rf_test_table)) / sd(bag_rf_test_table)
  sum_bag_test_rf <- sum(diag(bag_rf_test_table))
  bag_rf_test_prop <- diag(prop.table(bag_rf_test_table, margin = 1))
  
  bag_rf_validation_pred <- predict(bag_train_rf, validation, type = "class")
  bag_rf_validation_table <- table(bag_rf_validation_pred, validation01$bean_name)
  bag_rf_validation_accuracy[i] <- sum(diag(bag_rf_validation_table)) / sum(bag_rf_validation_table)
  bag_rf_validation_accuracy_mean <- mean(bag_rf_validation_accuracy)
  sum_diag_validation_bag_rf <- sum(diag(bag_rf_validation_table))
  bag_rf_validation_mean <- mean(diag(bag_rf_validation_table)) / mean(bag_rf_validation_table)
  bag_rf_validation_sd <- sd(diag(bag_rf_validation_table)) / sd(bag_rf_validation_table)
  sum_bag_validation_rf <- sum(diag(bag_rf_validation_table))
  bag_rf_validation_prop <- diag(prop.table(bag_rf_validation_table, margin = 1))
  
  bag_rf_holdout_mean <- mean(c(bag_rf_test_accuracy_mean, bag_rf_validation_accuracy_mean))
  bag_rf_overfitting[i] <- bag_rf_holdout_mean / bag_rf_train_accuracy_mean
  bag_rf_overfitting_mean <- mean(bag_rf_overfitting)
  bag_rf_overfitting_range <- range(bag_rf_overfitting)
  
  bag_rf_table <- bag_rf_test_table + bag_rf_validation_table
  
  bag_rf_end <- Sys.time()
  bag_rf_duration[i] <- bag_rf_end - bag_rf_start
  bag_rf_duration_mean <- mean(bag_rf_duration)
  
  
  #### C50 ####
  C50_start <- Sys.time()
  
  C50_train_fit <- C5.0(as_factor(y_train) ~ ., data = train)
  C50_train_pred <- predict(C50_train_fit, train)
  C50_train_table <- table(C50_train_pred, y_train)
  C50_train_accuracy[i] <- sum(diag(C50_train_table)) / sum(C50_train_table)
  C50_train_accuracy_mean <- mean(C50_train_accuracy)
  C50_train_mean <- mean(diag(C50_train_table)) / mean(C50_train_table)
  C50_train_sd <- sd(diag(C50_train_table)) / sd(C50_train_table)
  sum_diag_train_C50 <- sum(diag(C50_train_table))
  C50_train_prop <- diag(prop.table(C50_train_table, margin = 1))
  
  C50_test_pred <- predict(C50_train_fit, test)
  C50_test_table <- table(C50_test_pred, y_test)
  C50_test_accuracy[i] <- sum(diag(C50_test_table)) / sum(C50_test_table)
  C50_test_accuracy_mean <- mean(C50_test_accuracy)
  C50_test_mean <- mean(diag(C50_test_table)) / mean(C50_test_table)
  C50_test_sd <- sd(diag(C50_test_table)) / sd(C50_test_table)
  sum_diag_test_C50 <- sum(diag(C50_test_table))
  C50_test_prop <- diag(prop.table(C50_test_table, margin = 1))
  
  C50_validation_pred <- predict(C50_train_fit, validation)
  C50_validation_table <- table(C50_validation_pred, y_validation)
  C50_validation_accuracy[i] <- sum(diag(C50_validation_table)) / sum(C50_validation_table)
  C50_validation_accuracy_mean <- mean(C50_validation_accuracy)
  C50_validation_mean <- mean(diag(C50_validation_table)) / mean(C50_validation_table)
  C50_validation_sd <- sd(diag(C50_validation_table)) / sd(C50_validation_table)
  sum_diag_validation_C50 <- sum(diag(C50_validation_table))
  C50_validation_prop <- diag(prop.table(C50_validation_table, margin = 1))
  
  C50_holdout_mean <- mean(c(C50_test_accuracy_mean, C50_validation_accuracy_mean))
  C50_overfitting[i] <- C50_holdout_mean / C50_train_accuracy_mean
  C50_overfitting_mean <- mean(C50_overfitting)
  C50_overfitting_range <- range(C50_overfitting)
  
  C50_table <- C50_test_table + C50_validation_table
  
  C50_end <- Sys.time()
  C50_duration[i] <- C50_end - C50_start
  C50_duration_mean <- mean(C50_duration)

  
  ### Earth model ####
  earth_start <- Sys.time()
  earth_train_fit <- MachineShop::fit(bean_name ~ ., data = train01, model = "EarthModel")
  earth_train_pred <- predict(object = earth_train_fit, newdata = train)
  earth_train_table <- table(earth_train_pred, y_train)
  earth_train_accuracy[i] <- sum(diag(earth_train_table)) / sum(earth_train_table)
  earth_train_accuracy_mean <- mean(earth_train_accuracy)
  earth_train_mean <- mean(diag(earth_train_table)) / mean(earth_train_table)
  earth_train_sd <- sd(diag(earth_train_table)) / sd(earth_train_table)
  sum_diag_train_earth <- sum(diag(earth_train_table))
  earth_train_prop <- diag(prop.table(earth_train_table, margin = 1))
  
  earth_test_pred <- predict(object = earth_train_fit, newdata = test)
  earth_test_table <- table(earth_test_pred, y_test)
  earth_test_accuracy[i] <- sum(diag(earth_test_table)) / sum(earth_test_table)
  earth_test_accuracy_mean <- mean(earth_test_accuracy)
  earth_test_mean <- mean(diag(earth_test_table)) / mean(earth_test_table)
  earth_test_sd <- sd(diag(earth_test_table)) / sd(earth_test_table)
  sum_diag_test_earth <- sum(diag(earth_test_table))
  earth_test_prop <- diag(prop.table(earth_test_table, margin = 1))
  
  earth_validation_pred <- predict(object = earth_train_fit, newdata = validation)
  earth_validation_table <- table(earth_validation_pred, y_validation)
  earth_validation_accuracy[i] <- sum(diag(earth_validation_table)) / sum(earth_validation_table)
  earth_validation_accuracy_mean <- mean(earth_validation_accuracy)
  earth_validation_mean <- mean(diag(earth_validation_table)) / mean(earth_validation_table)
  earth_validation_sd <- sd(diag(earth_validation_table)) / sd(earth_validation_table)
  sum_diag_validation_earth <- sum(diag(earth_validation_table))
  earth_validation_prop <- diag(prop.table(earth_validation_table, margin = 1))
  
  earth_holdout_mean <- mean(c(earth_test_accuracy_mean, earth_validation_accuracy_mean))
  earth_overfitting[i] <- earth_holdout_mean / earth_train_accuracy_mean
  earth_overfitting_mean <- mean(earth_overfitting)
  earth_overfitting_range <- range(earth_overfitting)
  
  earth_table <- earth_test_table + earth_validation_table
  
  earth_end <- Sys.time()
  
  earth_duration[i] <- earth_end - earth_start
  earth_duration_mean <- mean(earth_duration)

  
  #### Flexible discriminant analysis ####
  fda_start <- Sys.time()
  
  fda_train_fit <- mda::fda(y_train ~ ., data = train)
  fda_train_pred <- predict(fda_train_fit, train)
  fda_train_table <- table(fda_train_pred, y_train)
  fda_train_accuracy[i] <- sum(diag(fda_train_table)) / sum(fda_train_table)
  fda_train_accuracy_mean <- mean(fda_train_accuracy)
  fda_train_mean <- mean(diag(fda_train_table)) / mean(fda_train_table)
  fda_train_sd <- sd(diag(fda_train_table)) / sd(fda_train_table)
  sum_diag_train_fda <- sum(diag(fda_train_table))
  fda_train_prop <- diag(prop.table(fda_train_table, margin = 1))
  
  fda_test_pred <- predict(fda_train_fit, test)
  fda_test_table <- table(fda_test_pred, y_test)
  fda_test_accuracy[i] <- sum(diag(fda_test_table)) / sum(fda_test_table)
  fda_test_accuracy_mean <- mean(fda_test_accuracy)
  fda_test_mean <- mean(diag(fda_test_table)) / mean(fda_test_table)
  fda_test_sd <- sd(diag(fda_test_table)) / sd(fda_test_table)
  sum_diag_test_fda <- sum(diag(fda_test_table))
  fda_test_prop <- diag(prop.table(fda_test_table, margin = 1))
  
  fda_validation_pred <- predict(fda_train_fit, validation)
  fda_validation_table <- table(fda_validation_pred, y_validation)
  fda_validation_accuracy[i] <- sum(diag(fda_validation_table)) / sum(fda_validation_table)
  fda_validation_accuracy_mean <- mean(fda_validation_accuracy)
  fda_validation_mean <- mean(diag(fda_validation_table)) / mean(fda_validation_table)
  fda_validation_sd <- sd(diag(fda_validation_table)) / sd(fda_validation_table)
  sum_diag_validation_fda <- sum(diag(fda_validation_table))
  fda_validation_prop <- diag(prop.table(fda_validation_table, margin = 1))
  
  fda_holdout_mean <- mean(c(fda_test_accuracy_mean, fda_validation_accuracy_mean))
  fda_overfitting[i] <- fda_holdout_mean / fda_train_accuracy_mean
  fda_overfitting_mean <- mean(fda_overfitting)
  fda_overfitting_range <- range(fda_overfitting)
  
  fda_table <- fda_test_table + fda_validation_table
  
  fda_end <- Sys.time()
  fda_duration[i] <- fda_end - fda_start
  fda_duration_mean <- mean(fda_duration)

  
  #### K-Nearest Neighbors ####
  knn_start <- Sys.time()
  
  train_train_scale <- scale(select_if(train, is.numeric))
  train_test_scale <- scale(select_if(train, is.numeric))
  knn_train_pred <- class::knn(train = train_train_scale, test = train_test_scale, cl = y_train, k = 1)
  knn_train_table <- table(knn_train_pred, y_train)
  knn_train_accuracy[i] <- sum(diag(knn_train_table)) / sum(knn_train_table)
  knn_train_accuracy_mean <- mean(knn_train_accuracy)
  knn_train_diag <- sum(diag(knn_train_table))
  knn_train_mean <- mean(diag(knn_train_table)) / mean(knn_train_table)
  knn_train_sd <- sd(diag(knn_train_table)) / sd(knn_train_table)
  sum_diag_train_knn <- sum(diag(knn_train_table))
  knn_train_prop <- diag(prop.table(knn_train_table, margin = 1))
  
  train_test_scale <- scale(select_if(train, is.numeric))
  test_test_scale <- scale(select_if(test, is.numeric))
  knn_test_pred <- class::knn(train = train_test_scale, test = test_test_scale, cl = y_train, k = 1)
  knn_test_table <- table(knn_test_pred, y_test)
  knn_test_accuracy[i] <- sum(diag(knn_test_table)) / sum(knn_test_table)
  knn_test_accuracy_mean <- mean(knn_test_accuracy)
  knn_test_diag <- sum(diag(knn_test_table))
  knn_test_mean <- mean(diag(knn_test_table)) / mean(knn_test_table)
  knn_test_sd <- sd(diag(knn_test_table)) / sd(knn_test_table)
  sum_diag_test_knn <- sum(diag(knn_test_table))
  knn_test_prop <- diag(prop.table(knn_test_table, margin = 1))
  
  train_validation_scale <- scale(select_if(train, is.numeric))
  validation_validation_scale <- scale(select_if(validation, is.numeric))
  knn_validation_pred <- class::knn(train = train_validation_scale, test = validation_validation_scale, cl = y_train, k = 1)
  knn_validation_table <- table(knn_validation_pred, y_validation)
  knn_validation_accuracy[i] <- sum(diag(knn_validation_table)) / sum(knn_validation_table)
  knn_validation_accuracy_mean <- mean(knn_validation_accuracy)
  knn_validation_diag <- sum(diag(knn_validation_table))
  knn_validation_mean <- mean(diag(knn_validation_table)) / mean(knn_validation_table)
  knn_validation_sd <- sd(diag(knn_validation_table)) / sd(knn_validation_table)
  sum_diag_validation_knn <- sum(diag(knn_validation_table))
  knn_validation_prop <- diag(prop.table(knn_validation_table, margin = 1))
  
  knn_holdout_mean <- mean(c(knn_test_accuracy_mean, knn_validation_accuracy_mean))
  knn_overfitting[i] <- knn_holdout_mean / knn_train_accuracy_mean
  knn_overfitting_mean <- mean(knn_overfitting)
  knn_overfitting_range <- range(knn_overfitting)
  
  knn_table <- knn_test_table + knn_validation_table
  
  knn_end <- Sys.time()
  knn_duration[i] <- knn_end - knn_start
  knn_duration_mean <- mean(knn_duration)

  
  #### Least Squares Support Vector Machine ####
  lssvm_start <- Sys.time()
  
  lssvm_train_fit <- kernlab::lssvm(bean_name ~ ., data = train01)
  lssvm_train_pred <- predict(lssvm_train_fit, newdata = train)
  lssvm_train_table <- table(lssvm_train_pred, y_train)
  lssvm_train_accuracy[i] <- sum(diag(lssvm_train_table)) / sum(lssvm_train_table)
  lssvm_train_accuracy_mean <- mean(lssvm_train_accuracy)
  lssvm_train_diag <- sum(diag(lssvm_train_table))
  lssvm_train_mean <- mean(diag(lssvm_train_table)) / mean(lssvm_train_table)
  lssvm_train_sd <- sd(diag(lssvm_train_table)) / sd(lssvm_train_table)
  sum_diag_train_lssvm <- sum(diag(lssvm_train_table))
  lssvm_train_prop <- diag(prop.table(lssvm_train_table, margin = 1))
  
  lssvm_test_pred <- predict(lssvm_train_fit, newdata = test)
  lssvm_test_table <- table(lssvm_test_pred, y_test)
  lssvm_test_accuracy[i] <- sum(diag(lssvm_test_table)) / sum(lssvm_test_table)
  lssvm_test_accuracy_mean <- mean(lssvm_test_accuracy)
  lssvm_test_diag <- sum(diag(lssvm_test_table))
  lssvm_test_mean <- mean(diag(lssvm_test_table)) / mean(lssvm_test_table)
  lssvm_test_sd <- sd(diag(lssvm_test_table)) / sd(lssvm_test_table)
  sum_diag_test_lssvm <- sum(diag(lssvm_test_table))
  lssvm_test_prop <- diag(prop.table(lssvm_test_table, margin = 1))
  
  lssvm_validation_pred <- predict(lssvm_train_fit, newdata = validation)
  lssvm_validation_table <- table(lssvm_validation_pred, y_validation)
  lssvm_validation_accuracy[i] <- sum(diag(lssvm_validation_table)) / sum(lssvm_validation_table)
  lssvm_validation_accuracy_mean <- mean(lssvm_validation_accuracy)
  lssvm_validation_diag <- sum(diag(lssvm_validation_table))
  lssvm_validation_mean <- mean(diag(lssvm_validation_table)) / mean(lssvm_validation_table)
  lssvm_validation_sd <- sd(diag(lssvm_validation_table)) / sd(lssvm_validation_table)
  sum_diag_validation_lssvm <- sum(diag(lssvm_validation_table))
  lssvm_validation_prop <- diag(prop.table(lssvm_validation_table, margin = 1))
  
  lssvm_holdout_mean <- mean(c(lssvm_test_accuracy_mean, lssvm_validation_accuracy_mean))
  lssvm_overfitting[i] <- lssvm_holdout_mean / lssvm_train_accuracy_mean
  lssvm_overfitting_mean <- mean(lssvm_overfitting)
  lssvm_overfitting_range <- range(lssvm_overfitting)
  
  lssvm_table <- lssvm_test_table + lssvm_validation_table
  
  lssvm_end <- Sys.time()
  lssvm_duration[i] <- lssvm_end - lssvm_start
  lssvm_duration_mean <- mean(lssvm_duration)
 
  
  #### Linear Discriminant Analysis ####
  lda_start <- Sys.time()
  
  lda_train_fit <- MASS::lda(y_train ~ ., data = train)
  lda_train_pred <- predict(lda_train_fit, train)
  lda_train_table <- table(lda_train_pred$class, y_train)
  lda_train_accuracy[i] <- sum(diag(lda_train_table)) / sum(lda_train_table)
  lda_train_accuracy_mean <- sum(lda_train_accuracy) / length(lda_train_accuracy)
  lda_train_mean <- mean(diag(lda_train_table)) / mean(lda_train_table)
  lda_train_sd <- sd(diag(lda_train_table)) / sd(lda_train_table)
  sum_diag_train_lda <- sum(diag(lda_train_table))
  lda_train_prop <- diag(prop.table(lda_train_table, margin = 1))
  
  lda_test_pred <- predict(lda_train_fit, test)
  lda_test_table <- table(lda_test_pred$class, y_test)
  lda_test_accuracy[i] <- sum(diag(lda_test_table)) / sum(lda_test_table)
  lda_test_accuracy_mean <- sum(lda_test_accuracy) / length(lda_test_accuracy)
  lda_test_mean <- mean(diag(lda_test_table)) / mean(lda_test_table)
  lda_test_sd <- sd(diag(lda_test_table)) / sd(lda_test_table)
  sum_diag_test_lda <- sum(diag(lda_test_table))
  lda_test_prop <- diag(prop.table(lda_test_table, margin = 1))
  
  lda_validation_pred <- predict(lda_train_fit, validation)
  lda_validation_table <- table(lda_validation_pred$class, y_validation)
  lda_validation_accuracy[i] <- sum(diag(lda_validation_table)) / sum(lda_validation_table)
  lda_validation_accuracy_mean <- mean(lda_validation_accuracy)
  lda_validation_mean <- mean(diag(lda_validation_table)) / mean(lda_validation_table)
  lda_validation_sd <- sd(diag(lda_validation_table)) / sd(lda_validation_table)
  sum_diag_validation_lda <- sum(diag(lda_validation_table))
  lda_validation_prop <- diag(prop.table(lda_validation_table, margin = 1))
  
  lda_holdout_mean <- mean(c(lda_test_accuracy_mean, lda_validation_accuracy_mean))
  lda_overfitting[i] <- lda_holdout_mean / lda_train_accuracy_mean
  lda_overfitting_mean <- mean(lda_overfitting)
  lda_overfitting_range <- range(lda_overfitting)
  
  lda_table <- lda_test_table + lda_validation_table
  
  lda_end <- Sys.time()
  lda_duration[i] <- lda_end - lda_start
  lda_duration_mean <- mean(lda_duration)

  
  #### Linear Model ####
  linear_start <- Sys.time()
  
  linear_train_fit <- MachineShop::fit(bean_name ~ ., data = train01, model = "LMModel")
  linear_train_pred <- predict(object = linear_train_fit, newdata = train01)
  linear_train_table <- table(linear_train_pred, y_train)
  linear_train_accuracy[i] <- sum(diag(linear_train_table)) / sum(linear_train_table)
  linear_train_accuracy_mean <- mean(linear_train_accuracy)
  linear_train_mean <- mean(diag(linear_train_table)) / mean(linear_train_table)
  linear_train_sd <- sd(diag(linear_train_table)) / sd(linear_train_table)
  sum_diag_train_linear <- sum(diag(linear_train_table))
  linear_train_prop <- diag(prop.table(linear_train_table, margin = 1))
  
  linear_test_pred <- predict(object = linear_train_fit, newdata = test01)
  linear_test_table <- table(linear_test_pred, y_test)
  linear_test_accuracy[i] <- sum(diag(linear_test_table)) / sum(linear_test_table)
  linear_test_accuracy_mean <- mean(linear_test_accuracy)
  linear_test_mean <- mean(diag(linear_test_table)) / mean(linear_test_table)
  linear_test_sd <- sd(diag(linear_test_table)) / sd(linear_test_table)
  sum_diag_test_linear <- sum(diag(linear_test_table))
  linear_test_prop <- diag(prop.table(linear_test_table, margin = 1))
  
  linear_validation_pred <- predict(object = linear_train_fit, newdata = validation01)
  linear_validation_table <- table(linear_validation_pred, y_validation)
  linear_validation_accuracy[i] <- sum(diag(linear_validation_table)) / sum(linear_validation_table)
  linear_validation_accuracy_mean <- mean(linear_validation_accuracy)
  linear_validation_mean <- mean(diag(linear_validation_table)) / mean(linear_validation_table)
  linear_validation_sd <- sd(diag(linear_validation_table)) / sd(linear_validation_table)
  sum_diag_validation_linear <- sum(diag(linear_validation_table))
  linear_validation_prop <- diag(prop.table(linear_validation_table, margin = 1))
  
  linear_holdout_mean <- mean(c(linear_test_accuracy_mean, linear_validation_accuracy_mean))
  linear_overfitting[i] <- linear_holdout_mean / linear_train_accuracy_mean
  linear_overfitting_mean <- mean(linear_overfitting)
  linear_overfitting_range <- range(linear_overfitting)
  
  linear_table <- linear_test_table + linear_validation_table
  
  linear_end <- Sys.time()
  linear_duration[i] <- linear_end - linear_start
  linear_duration_mean <- mean(linear_duration)

  
  #### Mixed discriminant analysis ####
  mda_start <- Sys.time()
  
  mda_train_fit <- mda::mda(formula = bean_name ~ ., data = test01)
  mda_train_pred <- predict(mda_train_fit, train)
  mda_train_table <- table(mda_train_pred, y_train)
  mda_train_accuracy[i] <- sum(diag(mda_train_table)) / sum(mda_train_table)
  mda_train_accuracy_mean <- mean(mda_train_accuracy)
  mda_train_mean <- mean(diag(mda_train_table)) / mean(mda_train_table)
  mda_train_sd <- sd(diag(mda_train_table)) / sd(mda_train_table)
  sum_diag_train_mda <- sum(diag(mda_train_table))
  mda_train_prop <- diag(prop.table(mda_train_table, margin = 1))
  
  mda_test_pred <- predict(mda_train_fit, test)
  mda_test_table <- table(mda_test_pred, y_test)
  mda_test_accuracy[i] <- sum(diag(mda_test_table)) / sum(mda_test_table)
  mda_test_accuracy_mean <- mean(mda_test_accuracy)
  mda_test_mean <- mean(diag(mda_test_table)) / mean(mda_test_table)
  mda_test_sd <- sd(diag(mda_test_table)) / sd(mda_test_table)
  sum_diag_test_mda <- sum(diag(mda_test_table))
  mda_test_prop <- diag(prop.table(mda_test_table, margin = 1))
  
  mda_validation_pred <- predict(mda_train_fit, validation)
  mda_validation_table <- table(mda_validation_pred, y_validation)
  mda_validation_accuracy[i] <- sum(diag(mda_validation_table)) / sum(mda_validation_table)
  mda_validation_accuracy_mean <- mean(mda_validation_accuracy)
  mda_validation_mean <- mean(diag(mda_validation_table)) / mean(mda_validation_table)
  mda_validation_sd <- sd(diag(mda_validation_table)) / sd(mda_validation_table)
  sum_diag_validation_mda <- sum(diag(mda_validation_table))
  mda_validation_prop <- diag(prop.table(mda_validation_table, margin = 1))
  
  mda_holdout_mean <- mean(c(mda_test_accuracy_mean, mda_validation_accuracy_mean))
  mda_overfitting[i] <- mda_holdout_mean / mda_train_accuracy_mean
  mda_overfitting_mean <- mean(mda_overfitting)
  mda_overfitting_range <- range(mda_overfitting)
  
  mda_table <- mda_test_table + mda_validation_table
  
  mda_end <- Sys.time()
  mda_duration[i] <- mda_end - mda_start
  mda_duration_mean <- mean(mda_duration)

  
  #### Naive Bayes ####
  n_bayes_start <- Sys.time()
  
  n_bayes_train_fit <- naiveBayes(y_train ~ ., data = train)
  n_bayes_train_pred <- predict(n_bayes_train_fit, train)
  n_bayes_train_table <- table(n_bayes_train_pred, y_train)
  n_bayes_train_accuracy[i] <- sum(diag(n_bayes_train_table)) / sum(n_bayes_train_table)
  n_bayes_train_accuracy_mean <- mean(n_bayes_train_accuracy)
  n_bayes_train_diag <- sum(diag(n_bayes_train_table))
  n_bayes_train_mean <- mean(diag(n_bayes_train_table)) / mean(n_bayes_train_table)
  n_bayes_train_sd <- sd(diag(n_bayes_train_table)) / sd(n_bayes_train_table)
  sum_diag_n_train_bayes <- sum(diag(n_bayes_train_table))
  n_bayes_train_prop <- diag(prop.table(n_bayes_train_table, margin = 1))
  
  n_bayes_test_pred <- predict(n_bayes_train_fit, test)
  n_bayes_test_table <- table(n_bayes_test_pred, y_test)
  n_bayes_test_accuracy[i] <- sum(diag(n_bayes_test_table)) / sum(n_bayes_test_table)
  n_bayes_test_accuracy_mean <- mean(n_bayes_test_accuracy)
  n_bayes_test_diag <- sum(diag(n_bayes_test_table))
  n_bayes_test_mean <- mean(diag(n_bayes_test_table)) / mean(n_bayes_test_table)
  n_bayes_test_sd <- sd(diag(n_bayes_test_table)) / sd(n_bayes_test_table)
  sum_diag_n_test_bayes <- sum(diag(n_bayes_test_table))
  n_bayes_test_prop <- diag(prop.table(n_bayes_test_table, margin = 1))
  
  n_bayes_validation_pred <- predict(n_bayes_train_fit, validation)
  n_bayes_validation_table <- table(n_bayes_validation_pred, y_validation)
  n_bayes_validation_accuracy[i] <- sum(diag(n_bayes_validation_table)) / sum(n_bayes_validation_table)
  n_bayes_validation_accuracy_mean <- mean(n_bayes_validation_accuracy)
  n_bayes_validation_diag <- sum(diag(n_bayes_validation_table))
  n_bayes_validation_mean <- mean(diag(n_bayes_validation_table)) / mean(n_bayes_validation_table)
  n_bayes_validation_sd <- sd(diag(n_bayes_validation_table)) / sd(n_bayes_validation_table)
  sum_diag_n_validation_bayes <- sum(diag(n_bayes_validation_table))
  n_bayes_validation_prop <- diag(prop.table(n_bayes_validation_table, margin = 1))
  
  n_bayes_holdout_mean <- mean(c(n_bayes_test_accuracy_mean, n_bayes_validation_accuracy_mean))
  n_bayes_overfitting[i] <- n_bayes_holdout_mean / n_bayes_train_accuracy_mean
  n_bayes_overfitting_mean <- mean(n_bayes_overfitting)
  n_bayes_overfitting_range <- range(n_bayes_overfitting)
  
  n_bayes_table <- n_bayes_test_table + n_bayes_validation_table
  
  n_bayes_end <- Sys.time()
  n_bayes_duration[i] <- n_bayes_end - n_bayes_start
  n_bayes_duration_mean <- mean(n_bayes_duration)

  
  #### Quadratic Discriminant Analysis ####
  qda_start <- Sys.time()
  
  qda_train_fit <- MASS::qda(bean_name ~ ., data = train01)
  qda_train_pred <- predict(object = qda_train_fit, newdata = train01)
  qda_train_table <- table(qda_train_pred$class, y_train)
  qda_train_accuracy[i] <- sum(diag(qda_train_table)) / sum(qda_train_table)
  qda_train_accuracy_mean <- mean(qda_train_accuracy)
  qda_train_mean <- mean(diag(qda_train_table)) / mean(qda_train_table)
  qda_train_sd <- sd(diag(qda_train_table)) / sd(qda_train_table)
  sum_diag_train_qda <- sum(diag(qda_train_table))
  qda_train_prop <- diag(prop.table(qda_train_table, margin = 1))
  
  qda_test_pred <- predict(object = qda_train_fit, newdata = test01)
  qda_test_pred <- qda_test_pred$class
  qda_test_table <- table(qda_test_pred, y_test)
  qda_test_accuracy[i] <- sum(diag(qda_test_table)) / sum(qda_test_table)
  qda_test_accuracy_mean <- mean(qda_test_accuracy)
  qda_test_mean <- mean(diag(qda_test_table)) / mean(qda_test_table)
  qda_test_sd <- sd(diag(qda_test_table)) / sd(qda_test_table)
  sum_diag_test_qda <- sum(diag(qda_test_table))
  qda_test_prop <- diag(prop.table(qda_test_table, margin = 1))
  
  qda_validation_pred <- predict(object = qda_train_fit, newdata = validation01)
  qda_validation_pred <- qda_validation_pred$class
  qda_validation_table <- table(qda_validation_pred, y_validation)
  qda_validation_accuracy[i] <- sum(diag(qda_validation_table)) / sum(qda_validation_table)
  qda_validation_accuracy_mean <- mean(qda_validation_accuracy)
  qda_validation_mean <- mean(diag(qda_validation_table)) / mean(qda_validation_table)
  qda_validation_sd <- sd(diag(qda_validation_table)) / sd(qda_validation_table)
  sum_diag_validation_qda <- sum(diag(qda_validation_table))
  qda_validation_prop <- diag(prop.table(qda_validation_table, margin = 1))
  
  qda_holdout_mean <- mean(c(qda_test_accuracy_mean, qda_validation_accuracy_mean))
  qda_overfitting[i] <- qda_holdout_mean / qda_train_accuracy_mean
  qda_overfitting_mean <- mean(qda_overfitting)
  qda_overfitting_range <- range(qda_overfitting)
  
  qda_table <- qda_test_table  + qda_validation_table
  
  qda_end <- Sys.time()
  qda_duration[i] <- qda_end - qda_start
  qda_duration_mean <- mean(qda_duration)

  
  #### Partial Least Squares ####
  pls_start <- Sys.time()
  
  pls_train_fit <- MachineShop::fit(bean_name ~ ., data = train01, model = "PLSModel")
  pls_train_predict <- predict(object = pls_train_fit, newdata = train01)
  pls_train_table <- table(pls_train_predict, y_train)
  pls_train_accuracy[i] <- sum(diag(pls_train_table)) / sum(pls_train_table)
  pls_train_accuracy_mean <- mean(pls_train_accuracy)
  pls_train_pred <- pls_train_predict
  pls_train_mean <- mean(diag(pls_train_table)) / sum(pls_train_table)
  pls_train_sd <- sd(diag(pls_train_table)) / sd(pls_train_table)
  sum_diag_train_pls <- sum(diag(pls_train_table))
  pls_train_prop <- diag(prop.table(pls_train_table, margin = 1))
  
  pls_test_predict <- predict(object = pls_train_fit, newdata = test01)
  pls_test_table <- table(pls_test_predict, y_test)
  pls_test_accuracy[i] <- sum(diag(pls_test_table)) / sum(pls_test_table)
  pls_test_accuracy_mean <- mean(pls_test_accuracy)
  pls_test_pred <- pls_test_predict
  pls_test_mean <- mean(diag(pls_test_table)) / sum(pls_test_table)
  pls_test_sd <- sd(diag(pls_test_table)) / sd(pls_test_table)
  sum_diag_test_pls <- sum(diag(pls_test_table))
  pls_test_prop <- diag(prop.table(pls_test_table, margin = 1))
  
  pls_validation_predict <- predict(object = pls_train_fit, newdata = validation01)
  pls_validation_table <- table(pls_validation_predict, y_validation)
  pls_validation_accuracy[i] <- sum(diag(pls_validation_table)) / sum(pls_validation_table)
  pls_validation_accuracy_mean <- mean(pls_validation_accuracy)
  pls_validation_pred <- pls_validation_predict
  pls_validation_mean <- mean(diag(pls_validation_table)) / sum(pls_validation_table)
  pls_validation_sd <- sd(diag(pls_validation_table)) / sd(pls_validation_table)
  sum_diag_validation_pls <- sum(diag(pls_validation_table))
  pls_validation_prop <- diag(prop.table(pls_validation_table, margin = 1))
  
  pls_holdout_mean <- mean(c(pls_test_accuracy_mean, pls_validation_accuracy_mean))
  pls_overfitting[i] <- pls_holdout_mean / pls_train_accuracy_mean
  pls_overfitting_mean <- mean(pls_overfitting)
  pls_overfitting_range <- range(pls_overfitting)
  
  pls_table <- pls_test_table + pls_validation_table
  
  pls_end <- Sys.time()
  pls_duration[i] <- pls_end - pls_start
  pls_duration_mean <- mean(pls_duration)

  
  #### Penalized Discriminant Analysis Model ####
  pda_start <- Sys.time()
  
  pda_train_fit <- MachineShop::fit(bean_name ~ ., data = train01, model = "PDAModel")
  pda_train_predict <- predict(object = pda_train_fit, newdata = train01)
  pda_train_table <- table(pda_train_predict, y_train)
  pda_train_accuracy[i] <- sum(diag(pda_train_table)) / sum(pda_train_table)
  pda_train_accuracy_mean <- mean(pda_train_accuracy)
  pda_train_pred <- pda_train_predict
  pda_train_mean <- mean(diag(pda_train_table)) / sum(pda_train_table)
  pda_train_sd <- sd(diag(pda_train_table)) / sd(pda_train_table)
  sum_diag_train_pda <- sum(diag(pda_train_table))
  pda_train_prop <- diag(prop.table(pda_train_table, margin = 1))
  
  pda_test_predict <- predict(object = pda_train_fit, newdata = test01)
  pda_test_table <- table(pda_test_predict, y_test)
  pda_test_accuracy[i] <- sum(diag(pda_test_table)) / sum(pda_test_table)
  pda_test_accuracy_mean <- mean(pda_test_accuracy)
  pda_test_pred <- pda_test_predict
  pda_test_mean <- mean(diag(pda_test_table)) / sum(pda_test_table)
  pda_test_sd <- sd(diag(pda_test_table)) / sd(pda_test_table)
  sum_diag_test_pda <- sum(diag(pda_test_table))
  pda_test_prop <- diag(prop.table(pda_test_table, margin = 1))
  
  pda_validation_predict <- predict(object = pda_train_fit, newdata = validation01)
  pda_validation_table <- table(pda_validation_predict, y_validation)
  pda_validation_accuracy[i] <- sum(diag(pda_validation_table)) / sum(pda_validation_table)
  pda_validation_accuracy_mean <- mean(pda_validation_accuracy)
  pda_validation_pred <- pda_validation_predict
  pda_validation_mean <- mean(diag(pda_validation_table)) / sum(pda_validation_table)
  pda_validation_sd <- sd(diag(pda_validation_table)) / sd(pda_validation_table)
  sum_diag_validation_pda <- sum(diag(pda_validation_table))
  pda_validation_prop <- diag(prop.table(pda_validation_table, margin = 1))
  
  pda_holdout_mean <- mean(c(pda_test_accuracy_mean, pda_validation_accuracy_mean))
  pda_overfitting[i] <- pda_holdout_mean / pda_train_accuracy_mean
  pda_overfitting_mean <- mean(pda_overfitting)
  pda_overfitting_range <- range(pda_overfitting)
  
  pda_table <- pda_test_table + pda_validation_table
  
  pda_end <- Sys.time()
  pda_duration[i] <- pda_end - pda_start
  pda_duration_mean <- mean(pda_duration)
  
  
  #### Random Forest ####
  rf_start <- Sys.time()
  
  tune_train_rf <- tune.randomForest(x = train, y = y_train)
  rf_train_pred <- predict(tune_train_rf$best.model, train, type = "class")
  rf_train_table <- table(rf_train_pred, y_train)
  rf_train_accuracy[i] <- sum(diag(rf_train_table)) / sum(rf_train_table)
  rf_train_accuracy_mean <- mean(rf_train_accuracy)
  rf_train_diag <- sum(diag(rf_train_table))
  rf_train_mean <- mean(diag(rf_train_table)) / mean(rf_train_table)
  rf_train_sd <- sd(diag(rf_train_table)) / sd(rf_train_table)
  sum_train_rf <- sum(diag(rf_train_table))
  rf_train_prop <- diag(prop.table(rf_train_table, margin = 1))
  
  rf_test_pred <- predict(tune_train_rf$best.model, test, type = "class")
  rf_test_table <- table(rf_test_pred, y_test)
  rf_test_accuracy[i] <- sum(diag(rf_test_table)) / sum(rf_test_table)
  rf_test_accuracy_mean <- mean(rf_test_accuracy)
  sum_diag_test_rf <- sum(diag(rf_test_table))
  rf_test_mean <- mean(diag(rf_test_table)) / mean(rf_test_table)
  rf_test_sd <- sd(diag(rf_test_table)) / sd(rf_test_table)
  sum_test_rf <- sum(diag(rf_test_table))
  rf_test_prop <- diag(prop.table(rf_test_table, margin = 1))
  
  rf_validation_pred <- predict(tune_train_rf$best.model, validation, type = "class")
  rf_validation_table <- table(rf_validation_pred, y_validation)
  rf_validation_accuracy[i] <- sum(diag(rf_validation_table)) / sum(rf_validation_table)
  rf_validation_accuracy_mean <- mean(rf_validation_accuracy)
  sum_diag_validation_rf <- sum(diag(rf_validation_table))
  rf_validation_mean <- mean(diag(rf_validation_table)) / mean(rf_validation_table)
  rf_validation_sd <- sd(diag(rf_validation_table)) / sd(rf_validation_table)
  sum_validation_rf <- sum(diag(rf_validation_table))
  rf_validation_prop <- diag(prop.table(rf_validation_table, margin = 1))
  
  rf_holdout_mean <- mean(c(rf_test_accuracy_mean, rf_validation_accuracy_mean))
  rf_overfitting[i] <- rf_holdout_mean / rf_train_accuracy_mean
  rf_overfitting_mean <- mean(rf_overfitting)
  rf_overfitting_range <- range(rf_overfitting)
  
  rf_table <- rf_test_table + rf_validation_table
  
  rf_end <- Sys.time()
  rf_duration[i] <- rf_end - rf_start
  rf_duration_mean <- mean(rf_duration)

  
  #### Ranger Model ####
  ranger_start <- Sys.time()
  
  ranger_train_fit <- MachineShop::fit(bean_name ~ ., data = train01, model = "RangerModel")
  ranger_train_predict <- predict(object = ranger_train_fit, newdata = train01)
  ranger_train_table <- table(ranger_train_predict, y_train)
  ranger_train_accuracy[i] <- sum(diag(ranger_train_table)) / sum(ranger_train_table)
  ranger_train_accuracy_mean <- mean(ranger_train_accuracy)
  ranger_train_pred <- ranger_train_predict
  ranger_train_mean <- mean(diag(ranger_train_table)) / sum(ranger_train_table)
  ranger_train_sd <- sd(diag(ranger_train_table)) / sd(ranger_train_table)
  sum_diag_train_ranger <- sum(diag(ranger_train_table))
  ranger_train_prop <- diag(prop.table(ranger_train_table, margin = 1))
  
  ranger_test_predict <- predict(object = ranger_train_fit, newdata = test01)
  ranger_test_table <- table(ranger_test_predict, y_test)
  ranger_test_accuracy[i] <- sum(diag(ranger_test_table)) / sum(ranger_test_table)
  ranger_test_accuracy_mean <- mean(ranger_test_accuracy)
  ranger_test_pred <- ranger_test_predict
  ranger_test_mean <- mean(diag(ranger_test_table)) / sum(ranger_test_table)
  ranger_test_sd <- sd(diag(ranger_test_table)) / sd(ranger_test_table)
  sum_diag_test_ranger <- sum(diag(ranger_test_table))
  ranger_test_prop <- diag(prop.table(ranger_test_table, margin = 1))
  
  ranger_validation_predict <- predict(object = ranger_train_fit, newdata = validation01)
  ranger_validation_table <- table(ranger_validation_predict, y_validation)
  ranger_validation_accuracy[i] <- sum(diag(ranger_validation_table)) / sum(ranger_validation_table)
  ranger_validation_accuracy_mean <- mean(ranger_validation_accuracy)
  ranger_validation_pred <- ranger_validation_predict
  ranger_validation_mean <- mean(diag(ranger_validation_table)) / sum(ranger_validation_table)
  ranger_validation_sd <- sd(diag(ranger_validation_table)) / sd(ranger_validation_table)
  sum_diag_validation_ranger <- sum(diag(ranger_validation_table))
  ranger_validation_prop <- diag(prop.table(ranger_validation_table, margin = 1))
  
  ranger_holdout_mean <- mean(c(ranger_test_accuracy_mean, ranger_validation_accuracy_mean))
  ranger_overfitting[i] <- ranger_holdout_mean / ranger_train_accuracy_mean
  ranger_overfitting_mean <- mean(ranger_overfitting)
  ranger_overfitting_range <- range(ranger_overfitting)
  
  ranger_table <- ranger_test_table + ranger_validation_table
  
  ranger_end <- Sys.time()
  ranger_duration[i] <- ranger_end - ranger_start
  ranger_duration_mean <- mean(ranger_duration)

  
  #### Regularized discriminant analysis ####
  rda_start <- Sys.time()
  
  rda_train_fit <- klaR::rda(y_train ~ ., data = train)
  rda_train_pred <- predict(object = rda_train_fit, newdata = train)
  rda_train_pred <- rda_train_pred$class
  rda_train_table <- table(rda_train_pred, y_train)
  rda_train_accuracy[i] <- sum(diag(rda_train_table)) / sum(rda_train_table)
  rda_train_accuracy_mean <- mean(rda_train_accuracy)
  rda_train_mean <- mean(diag(rda_train_table)) / mean(rda_train_table)
  rda_train_sd <- sd(diag(rda_train_table)) / sd(rda_train_table)
  sum_diag_train_rda <- sum(diag(rda_train_table))
  rda_train_prop <- diag(prop.table(rda_train_table, margin = 1))
  
  rda_test_pred <- predict(object = rda_train_fit, newdata = test)
  rda_test_pred <- rda_test_pred$class
  rda_test_table <- table(rda_test_pred, y_test)
  rda_test_accuracy[i] <- sum(diag(rda_test_table)) / sum(rda_test_table)
  rda_test_accuracy_mean <- mean(rda_test_accuracy)
  rda_test_mean <- mean(diag(rda_test_table)) / mean(rda_test_table)
  rda_test_sd <- sd(diag(rda_test_table)) / sd(rda_test_table)
  sum_diag_test_rda <- sum(diag(rda_test_table))
  rda_test_prop <- diag(prop.table(rda_test_table, margin = 1))
  
  rda_validation_pred <- predict(object = rda_train_fit, newdata = validation)
  rda_validation_pred <- rda_validation_pred$class
  rda_validation_table <- table(rda_validation_pred, y_validation)
  rda_validation_accuracy[i] <- sum(diag(rda_validation_table)) / sum(rda_validation_table)
  rda_validation_accuracy_mean <- mean(rda_validation_accuracy)
  rda_validation_mean <- mean(diag(rda_validation_table)) / mean(rda_validation_table)
  rda_validation_sd <- sd(diag(rda_validation_table)) / sd(rda_validation_table)
  sum_diag_validation_rda <- sum(diag(rda_validation_table))
  rda_validation_prop <- diag(prop.table(rda_validation_table, margin = 1))
  
  rda_holdout_mean <- mean(c(rda_test_accuracy_mean, rda_validation_accuracy_mean))
  rda_overfitting[i] <- rda_holdout_mean / rda_train_accuracy_mean
  rda_overfitting_mean <- mean(rda_overfitting)
  rda_overfitting_range <- range(rda_overfitting)
  
  rda_table <- rda_test_table + rda_validation_table
  
  rda_end <- Sys.time()
  rda_duration[i] <- rda_end - rda_start
  rda_duration_mean <- mean(rda_duration)

  
  #### RPart Model ####
  rpart_start <- Sys.time()
  
  rpart_train_fit <- MachineShop::fit(bean_name ~ ., data = train01, model = "RPartModel")
  rpart_train_predict <- predict(object = rpart_train_fit, newdata = train01)
  rpart_train_table <- table(rpart_train_predict, y_train)
  rpart_train_accuracy[i] <- sum(diag(rpart_train_table)) / sum(rpart_train_table)
  rpart_train_accuracy_mean <- mean(rpart_train_accuracy)
  rpart_train_pred <- rpart_train_predict
  rpart_train_mean <- mean(diag(rpart_train_table)) / sum(rpart_train_table)
  rpart_train_sd <- sd(diag(rpart_train_table)) / sd(rpart_train_table)
  sum_diag_train_rpart <- sum(diag(rpart_train_table))
  rpart_train_prop <- diag(prop.table(rpart_train_table, margin = 1))
  
  rpart_test_predict <- predict(object = rpart_train_fit, newdata = test01)
  rpart_test_table <- table(rpart_test_predict, y_test)
  rpart_test_accuracy[i] <- sum(diag(rpart_test_table)) / sum(rpart_test_table)
  rpart_test_accuracy_mean <- mean(rpart_test_accuracy)
  rpart_test_pred <- rpart_test_predict
  rpart_test_mean <- mean(diag(rpart_test_table)) / sum(rpart_test_table)
  rpart_test_sd <- sd(diag(rpart_test_table)) / sd(rpart_test_table)
  sum_diag_test_rpart <- sum(diag(rpart_test_table))
  rpart_test_prop <- diag(prop.table(rpart_test_table, margin = 1))
  
  rpart_validation_predict <- predict(object = rpart_train_fit, newdata = validation01)
  rpart_validation_table <- table(rpart_validation_predict, y_validation)
  rpart_validation_accuracy[i] <- sum(diag(rpart_validation_table)) / sum(rpart_validation_table)
  rpart_validation_accuracy_mean <- mean(rpart_validation_accuracy)
  rpart_validation_pred <- rpart_validation_predict
  rpart_validation_mean <- mean(diag(rpart_validation_table)) / sum(rpart_validation_table)
  rpart_validation_sd <- sd(diag(rpart_validation_table)) / sd(rpart_validation_table)
  sum_diag_validation_rpart <- sum(diag(rpart_validation_table))
  rpart_validation_prop <- diag(prop.table(rpart_validation_table, margin = 1))
  
  rpart_holdout_mean <- mean(c(rpart_test_accuracy_mean, rpart_validation_accuracy_mean))
  rpart_overfitting[i] <- rpart_holdout_mean / rpart_train_accuracy_mean
  rpart_overfitting_mean <- mean(rpart_overfitting)
  rpart_overfitting_range <- range(rpart_overfitting)
  
  rpart_table <- rpart_test_table + rpart_validation_table
  
  rpart_end <- Sys.time()
  rpart_duration[i] <- rpart_end - rpart_start
  rpart_duration_mean <- mean(rpart_duration)

  
  #### Support Vector Machines ####
  svm_start <- Sys.time()
  
  svm_train_fit <- svm(y_train ~ ., data = train, kernel = "radial", gamma = 1, cost = 1)
  svm_train_pred <- predict(svm_train_fit, train, type = "class")
  svm_train_table <- table(svm_train_pred, y_train)
  svm_train_accuracy[i] <- sum(diag(svm_train_table)) / sum(svm_train_table)
  svm_train_accuracy_mean <- mean(svm_train_accuracy)
  svm_train_diag <- sum(diag(svm_train_table))
  svm_train_mean <- mean(diag(svm_train_table)) / mean(svm_train_table)
  svm_train_sd <- sd(diag(svm_train_table)) / sd(svm_train_table)
  sum_diag_train_svm <- sum(diag(svm_train_table))
  svm_train_prop <- diag(prop.table(svm_train_table, margin = 1))
  
  svm_test_pred <- predict(svm_train_fit, test, type = "class")
  svm_test_table <- table(svm_test_pred, y_test)
  svm_test_accuracy[i] <- sum(diag(svm_test_table)) / sum(svm_test_table)
  svm_test_accuracy_mean <- mean(svm_test_accuracy)
  svm_test_diag <- sum(diag(svm_test_table))
  svm_test_mean <- mean(diag(svm_test_table)) / mean(svm_test_table)
  svm_test_sd <- sd(diag(svm_test_table)) / sd(svm_test_table)
  sum_diag_test_svm <- sum(diag(svm_test_table))
  svm_test_prop <- diag(prop.table(svm_test_table, margin = 1))
  
  svm_validation_pred <- predict(svm_train_fit, validation, type = "class")
  svm_validation_table <- table(svm_validation_pred, y_validation)
  svm_validation_accuracy[i] <- sum(diag(svm_validation_table)) / sum(svm_validation_table)
  svm_validation_accuracy_mean <- mean(svm_validation_accuracy)
  svm_validation_diag <- sum(diag(svm_validation_table))
  svm_validation_mean <- mean(diag(svm_validation_table)) / mean(svm_validation_table)
  svm_validation_sd <- sd(diag(svm_validation_table)) / sd(svm_validation_table)
  sum_diag_validation_svm <- sum(diag(svm_validation_table))
  svm_validation_prop <- diag(prop.table(svm_validation_table, margin = 1))
  svm_holdout_mean <- mean(c(svm_test_accuracy_mean, svm_validation_accuracy_mean))
  svm_overfitting[i] <- svm_holdout_mean / svm_train_accuracy_mean
  svm_overfitting_mean <- mean(svm_overfitting)
  svm_overfitting_range <- range(svm_overfitting)
  
  svm_table <- svm_test_table + svm_validation_table
  
  svm_end <- Sys.time()
  svm_duration[i] <- svm_end - svm_start
  svm_duration_mean <- mean(svm_duration)

  
  #### Trees ####
  tree_start <- Sys.time()
  
  tree_train_fit <- tree(y_train ~ ., data = train)
  cv_train_tree <- cv.tree(object = tree_train_fit, FUN = prune.misclass)
  prune_train_tree <- prune.misclass(tree_train_fit, best = 5)
  tree_train_pred <- predict(prune_train_tree, train, type = "class")
  tree_train_table <- table(tree_train_pred, y_train)
  tree_train_accuracy[i] <- sum(diag(tree_train_table)) / sum(tree_train_table)
  tree_train_accuracy_mean <- mean(tree_train_accuracy)
  tree_train_diag <- sum(diag(tree_train_table))
  tree_train_mean <- mean(diag(tree_train_table)) / mean(tree_train_table)
  tree_train_sd <- sd(diag(tree_train_table)) / sd(tree_train_table)
  sum_diag_train_tree <- sum(diag(tree_train_table))
  tree_train_prop <- diag(prop.table(tree_train_table, margin = 1))
  
  tree_test_pred <- predict(prune_train_tree, test, type = "class")
  tree_test_table <- table(tree_test_pred, y_test)
  tree_test_accuracy[i] <- sum(diag(tree_test_table)) / sum(tree_test_table)
  tree_test_accuracy_mean <- mean(tree_test_accuracy)
  tree_test_diag <- sum(diag(tree_test_table))
  tree_test_mean <- mean(diag(tree_test_table)) / mean(tree_test_table)
  tree_test_sd <- sd(diag(tree_test_table)) / sd(tree_test_table)
  sum_diag_test_tree <- sum(diag(tree_test_table))
  tree_test_prop <- diag(prop.table(tree_test_table, margin = 1))
  
  tree_validation_pred <- predict(prune_train_tree, validation, type = "class")
  tree_validation_table <- table(tree_validation_pred, y_validation)
  tree_validation_accuracy[i] <- sum(diag(tree_validation_table)) / sum(tree_validation_table)
  tree_validation_accuracy_mean <- mean(tree_validation_accuracy)
  tree_validation_diag <- sum(diag(tree_validation_table))
  tree_validation_mean <- mean(diag(tree_validation_table)) / mean(tree_validation_table)
  tree_validation_sd <- sd(diag(tree_validation_table)) / sd(tree_validation_table)
  sum_diag_validation_tree <- sum(diag(tree_validation_table))
  tree_validation_prop <- diag(prop.table(tree_validation_table, margin = 1))
  
  tree_holdout_mean <- mean(c(tree_test_accuracy_mean, tree_validation_accuracy_mean))
  tree_overfitting[i] <- tree_holdout_mean / tree_train_accuracy_mean
  tree_overfitting_mean <- mean(tree_overfitting)
  tree_overfitting_range <- range(tree_overfitting)
  
  tree_table <- tree_test_table + tree_validation_table
  
  tree_end <- Sys.time()
  tree_duration[i] <- tree_end - tree_start
  tree_duration_mean <- mean(tree_duration)


###################################################################################################################
  
########################################     Start ENSEMBLES Here      ############################################
  
###################################################################################################################

  
  #### Build Ensembles ####
  ensemble1 <- data.frame('ADA boost' = c(adaboost_test_pred, adaboost_validation_pred),
                          'Bagged Random Forest' = c(bag_rf_test_pred, bag_rf_validation_pred),
                          'C50' = c(C50_test_pred, C50_validation_pred),
                          'Earth' = c(earth_test_pred, earth_validation_pred),
                          'Flexible Discriminant Analysis' = c(fda_test_pred, fda_validation_pred),
                          'K-Nearest Neighbors' = c(knn_test_pred, knn_validation_pred),
                          'Least Squares Support Vector Machines' = c(lssvm_test_pred, lssvm_validation_pred),
                          'Linear Discriminant Analysis' = c(lda_test_pred$class, lda_validation_pred$class),
                          'Linear' = c(linear_test_pred, linear_validation_pred),
                          'Mixed Discriminant Analysis' = c(mda_test_pred, mda_validation_pred),
                          'Naive Bayes' = c(n_bayes_test_pred, n_bayes_validation_pred),
                          'Quadratic Discriminant Analysis' = c(qda_test_pred, qda_validation_pred),
                          'Partial Least Squares' = c(pls_test_pred, pls_validation_pred),
                          'Penalized Discriminant Analysis' = c(pda_test_pred, pda_validation_pred),
                          'Random Forest' = c(rf_test_pred, rf_validation_pred),
                          'Ranger' = c(ranger_test_pred, ranger_validation_pred),
                          'Regularized Discriminant Analysis' = c(rda_test_pred, rda_validation_pred),
                          'RPart' = c(rpart_test_pred, rpart_validation_pred),
                          'Support Vector Machines' = c(svm_test_pred, svm_validation_pred),
                          'Trees' = c(tree_test_pred, tree_validation_pred)
  )
  
  ensemble_row_numbers <- as.numeric(row.names(ensemble1))
  ensemble1$y <- df[ensemble_row_numbers, ncol(df)]
  ensemble_index <- sample(c(1:3), nrow(ensemble1), replace=TRUE, prob=c(0.6,0.2, 0.2))
  ensemble_train  <- ensemble1[ensemble_index == 1, ]
  ensemble_test   <- ensemble1[ensemble_index ==2, ]
  ensemble_validation <- ensemble1[ensemble_index == 3, ]
  ensemble_y_train <- ensemble_train$y
  ensemble_y_test <- ensemble_test$y
  ensemble_y_validation <- ensemble_validation$y

  ### Ensemble with adabag ####
  ensemble_adabag_start <- Sys.time()
  
  ensemble_adabag_train_fit <- adabag::bagging(y ~ ., data = ensemble_train)
  ensemble_adabag_train_pred <- predict(object = ensemble_adabag_train_fit, newdata = ensemble_train)
  ensemble_adabag_train_pred <- as_factor(ensemble_adabag_train_pred$class)
  ensemble_adabag_train_table <- table(ensemble_adabag_train_pred, ensemble_train$y)
  ensemble_adabag_train_accuracy[i] <- sum(diag(ensemble_adabag_train_table)) / sum(ensemble_adabag_train_table)
  ensemble_adabag_train_accuracy_mean <- mean(ensemble_adabag_train_accuracy)
  ensemble_adabag_train_mean <- mean(diag(ensemble_adabag_train_table)) / mean(ensemble_adabag_train_table)
  ensemble_adabag_train_sd <- sd(diag(ensemble_adabag_train_table)) / sd(ensemble_adabag_train_table)
  ensemble_adabag_train_diag <- sum(diag(ensemble_adabag_train_table))
  ensemble_sum_diag_train_adabag <- sum(diag(ensemble_adabag_train_table))
  ensemble_adabag_train_prop <- diag(prop.table(ensemble_adabag_train_table, margin = 1))
  
  ensemble_adabag_test_pred <- predict(object = ensemble_adabag_train_fit, newdata = ensemble_test)
  ensemble_adabag_test_pred <- as_factor(ensemble_adabag_test_pred$class)
  ensemble_adabag_test_table <- table(ensemble_adabag_test_pred, ensemble_test$y)
  ensemble_adabag_test_accuracy[i] <- sum(diag(ensemble_adabag_test_table)) / sum(ensemble_adabag_test_table)
  ensemble_adabag_test_accuracy_mean <- mean(ensemble_adabag_test_accuracy)
  ensemble_adabag_test_mean <- mean(diag(ensemble_adabag_test_table)) / mean(ensemble_adabag_test_table)
  ensemble_adabag_test_sd <- sd(diag(ensemble_adabag_test_table)) / sd(ensemble_adabag_test_table)
  ensemble_adabag_test_diag <- sum(diag(ensemble_adabag_test_table))
  ensemble_sum_diag_test_adabag <- sum(diag(ensemble_adabag_test_table))
  ensemble_adabag_test_prop <- diag(prop.table(ensemble_adabag_test_table, margin = 1))
  
  ensemble_adabag_validation_pred <- predict(object = ensemble_adabag_train_fit, newdata = ensemble_validation)
  ensemble_adabag_validation_pred <- as_factor(ensemble_adabag_validation_pred$class)
  ensemble_adabag_validation_table <- table(ensemble_adabag_validation_pred, ensemble_validation$y)
  ensemble_adabag_validation_accuracy[i] <- sum(diag(ensemble_adabag_validation_table)) / sum(ensemble_adabag_validation_table)
  ensemble_adabag_validation_accuracy_mean <- mean(ensemble_adabag_validation_accuracy)
  ensemble_adabag_validationmean <- mean(diag(ensemble_adabag_validation_table)) / mean(ensemble_adabag_validation_table)
  ensemble_adabag_validationsd <- sd(diag(ensemble_adabag_validation_table)) / sd(ensemble_adabag_validation_table)
  ensemble_adabag_validation_diag <- sum(diag(ensemble_adabag_validation_table))
  ensemble_sum_diag_validation_adabag <- sum(diag(ensemble_adabag_validation_table))
  ensemble_adabag_validation_prop <- diag(prop.table(ensemble_adabag_validation_table, margin = 1))
  
  ensemble_adabag_holdout_mean <- mean(c(ensemble_adabag_test_accuracy_mean, ensemble_adabag_validation_accuracy_mean))
  ensemble_adabag_overfitting[i] <- ensemble_adabag_holdout_mean / ensemble_adabag_train_accuracy_mean
  ensemble_adabag_overfitting_mean <- mean(ensemble_adabag_overfitting)
  ensemble_adabag_overfitting_range <- range(ensemble_adabag_overfitting)
  
  ensemble_adabag_table <- ensemble_adabag_test_table + ensemble_adabag_validation_table
  
  ensemble_adabag_end <- Sys.time()
  ensemble_adabag_duration[i] <- ensemble_adabag_end - ensemble_adabag_start
  ensemble_adabag_duration_mean <- mean(ensemble_adabag_duration)
  
  
  #### Boosting with ADA Boost ####
  ensemble_adaboost_start <- Sys.time()
  
  ensemble_adaboost_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "AdaBoostModel")
  ensemble_adaboost_train_pred <- predict(object = ensemble_adaboost_train_fit, newdata = ensemble_train)
  ensemble_adaboost_train_table <- table(ensemble_adaboost_train_pred, ensemble_y_train)
  ensemble_adaboost_train_accuracy[i] <- sum(diag(ensemble_adaboost_train_table)) / sum(ensemble_adaboost_train_table)
  ensemble_adaboost_train_accuracy_mean <- mean(ensemble_adaboost_train_accuracy)
  ensemble_adaboost_train_mean <- mean(diag(ensemble_adaboost_train_table)) / mean(ensemble_adaboost_train_table)
  ensemble_adaboost_train_sd <- sd(diag(ensemble_adaboost_train_table)) / sd(ensemble_adaboost_train_table)
  ensemble_adaboost_train_diag <- sum(diag(ensemble_adaboost_train_table))
  ensemble_sum_diag_train_adaboost <- sum(diag(ensemble_adaboost_train_table))
  ensemble_adaboost_train_prop <- diag(prop.table(ensemble_adaboost_train_table, margin = 1))
  
  ensemble_adaboost_test_pred <- predict(object = ensemble_adaboost_train_fit, newdata = ensemble_test)
  ensemble_adaboost_test_table <- table(ensemble_adaboost_test_pred, ensemble_y_test)
  ensemble_adaboost_test_accuracy[i] <- sum(diag(ensemble_adaboost_test_table)) / sum(ensemble_adaboost_test_table)
  ensemble_adaboost_test_accuracy_mean <- mean(ensemble_adaboost_test_accuracy)
  ensemble_adaboost_test_mean <- mean(diag(ensemble_adaboost_test_table)) / mean(ensemble_adaboost_test_table)
  ensemble_adaboost_test_sd <- sd(diag(ensemble_adaboost_test_table)) / sd(ensemble_adaboost_test_table)
  ensemble_adaboost_test_diag <- sum(diag(ensemble_adaboost_test_table))
  ensemble_sum_diag_test_adaboost <- sum(diag(ensemble_adaboost_test_table))
  ensemble_adaboost_test_prop <- diag(prop.table(ensemble_adaboost_test_table, margin = 1))
  
  ensemble_adaboost_validation_pred <- predict(object = ensemble_adaboost_train_fit, newdata = ensemble_validation)
  ensemble_adaboost_validation_table <- table(ensemble_adaboost_validation_pred, ensemble_y_validation)
  ensemble_adaboost_validation_accuracy[i] <- sum(diag(ensemble_adaboost_validation_table)) / sum(ensemble_adaboost_validation_table)
  ensemble_adaboost_validation_accuracy_mean <- mean(ensemble_adaboost_validation_accuracy)
  ensemble_adaboost_validation_mean <- mean(diag(ensemble_adaboost_validation_table)) / mean(ensemble_adaboost_validation_table)
  ensemble_adaboost_validation_sd <- sd(diag(ensemble_adaboost_validation_table)) / sd(ensemble_adaboost_validation_table)
  ensemble_adaboost_validation_diag <- sum(diag(ensemble_adaboost_validation_table))
  ensemble_sum_diag_validation_adaboost <- sum(diag(ensemble_adaboost_validation_table))
  ensemble_adaboost_validation_prop <- diag(prop.table(ensemble_adaboost_validation_table, margin = 1))
  
  ensemble_adaboost_holdout_mean <- mean(c(ensemble_adaboost_test_accuracy_mean, ensemble_adaboost_validation_accuracy_mean))
  ensemble_adaboost_overfitting[i] <- ensemble_adaboost_holdout_mean / ensemble_adaboost_train_accuracy_mean
  ensemble_adaboost_overfitting_mean <- mean(ensemble_adaboost_overfitting)
  ensemble_adaboost_overfitting_range <- range(ensemble_adaboost_overfitting)
  
  ensemble_adaboost_table <- ensemble_adaboost_test_table + ensemble_adaboost_validation_table
  
  ensemble_adaboost_end <- Sys.time()
  ensemble_adaboost_duration[i] <- ensemble_adaboost_end - ensemble_adaboost_start
  ensemble_adaboost_duration_mean <- mean(ensemble_adaboost_duration)
  
  
  #### Ensemble Bagged CART ####
  ensemble_bag_cart_start <- Sys.time()
  
  ensemble_bag_cart_train_fit <- bagging(y ~ ., data = ensemble_train)
  ensemble_bag_cart_train_pred <- predict(ensemble_bag_cart_train_fit, ensemble_train)
  ensemble_bag_cart_train_table <- table(ensemble_bag_cart_train_pred$class, ensemble_train$y)
  ensemble_bag_cart_train_accuracy[i] <- sum(diag(ensemble_bag_cart_train_table)) / sum(ensemble_bag_cart_train_table)
  ensemble_bag_cart_train_accuracy_mean <- mean(ensemble_bag_cart_train_accuracy)
  ensemble_bag_cart_train_mean <- mean(diag(ensemble_bag_cart_train_table)) / mean(ensemble_bag_cart_train_table)
  ensemble_bag_cart_train_sd <- sd(diag(ensemble_bag_cart_train_table)) / sd(ensemble_bag_cart_train_table)
  ensemble_sum_diag_bag_train_cart <- sum(diag(ensemble_bag_cart_train_table))
  ensemble_bag_cart_train_prop <- diag(prop.table(ensemble_bag_cart_train_table, margin = 1))
  
  ensemble_bag_cart_test_pred <- predict(ensemble_bag_cart_train_fit, ensemble_test)
  ensemble_bag_cart_test_table <- table(ensemble_bag_cart_test_pred$class, ensemble_test$y)
  ensemble_bag_cart_test_accuracy[i] <- sum(diag(ensemble_bag_cart_test_table)) / sum(ensemble_bag_cart_test_table)
  ensemble_bag_cart_test_accuracy_mean <- mean(ensemble_bag_cart_test_accuracy)
  ensemble_bag_cart_test_mean <- mean(diag(ensemble_bag_cart_test_table)) / mean(ensemble_bag_cart_test_table)
  ensemble_bag_cart_test_sd <- sd(diag(ensemble_bag_cart_test_table)) / sd(ensemble_bag_cart_test_table)
  ensemble_sum_diag_bag_test_cart <- sum(diag(ensemble_bag_cart_test_table))
  ensemble_bag_cart_test_prop <- diag(prop.table(ensemble_bag_cart_test_table, margin = 1))
  
  ensemble_bag_cart_validation_pred <- predict(ensemble_bag_cart_train_fit, ensemble_validation)
  ensemble_bag_cart_validation_table <- table(ensemble_bag_cart_validation_pred$class, ensemble_validation$y)
  ensemble_bag_cart_validation_accuracy[i] <- sum(diag(ensemble_bag_cart_validation_table)) / sum(ensemble_bag_cart_validation_table)
  ensemble_bag_cart_validation_accuracy_mean <- mean(ensemble_bag_cart_validation_accuracy)
  ensemble_bag_cart_validation_mean <- mean(diag(ensemble_bag_cart_validation_table)) / mean(ensemble_bag_cart_validation_table)
  ensemble_bag_cart_validation_sd <- sd(diag(ensemble_bag_cart_validation_table)) / sd(ensemble_bag_cart_validation_table)
  ensemble_sum_diag_bag_validation_cart <- sum(diag(ensemble_bag_cart_validation_table))
  ensemble_bag_cart_validation_prop <- diag(prop.table(ensemble_bag_cart_validation_table, margin = 1))
  
  ensemble_bag_cart_holdout_mean <- mean(c(ensemble_bag_cart_test_accuracy_mean, ensemble_bag_cart_validation_accuracy_mean))
  ensemble_bag_cart_overfitting[i] <- ensemble_bag_cart_holdout_mean / ensemble_bag_cart_train_accuracy_mean
  ensemble_bag_cart_overfitting_mean <- mean(ensemble_bag_cart_overfitting)
  ensemble_bag_cart_overfitting_range <- range(ensemble_bag_cart_overfitting)
  
  ensemble_bag_cart_table <- ensemble_bag_cart_test_table + ensemble_bag_cart_validation_table
  
  ensemble_bag_cart_end <- Sys.time()
  ensemble_bag_cart_duration[i] <- ensemble_bag_cart_end - ensemble_bag_cart_start
  ensemble_bag_cart_duration_mean <- mean(ensemble_bag_cart_duration)
  
  
  #### Ensemble Bagged Random Forest ####
  ensemble_bag_rf_start <- Sys.time()
  
  ensemble_bag_train_rf <- randomForest(ensemble_y_train ~ ., data = ensemble_train, mtry = ncol(ensemble_train)-1)
  ensemble_bag_rf_train_pred <- predict(ensemble_bag_train_rf, ensemble_train, type = "class")
  ensemble_bag_rf_train_table <- table(ensemble_bag_rf_train_pred, ensemble_train$y)
  ensemble_bag_rf_train_accuracy[i] <- sum(diag(ensemble_bag_rf_train_table)) / sum(ensemble_bag_rf_train_table)
  ensemble_bag_rf_train_accuracy_mean <- mean(ensemble_bag_rf_train_accuracy)
  ensemble_bag_rf_train_diag <- sum(diag(ensemble_bag_rf_train_table))
  ensemble_bag_rf_train_mean <- mean(diag(ensemble_bag_rf_train_table)) / mean(ensemble_bag_rf_train_table)
  ensemble_bag_rf_train_sd <- sd(diag(ensemble_bag_rf_train_table)) / sd(ensemble_bag_rf_train_table)
  sum_ensemble_bag_train_rf <- sum(diag(ensemble_bag_rf_train_table))
  ensemble_bag_rf_train_prop <- diag(prop.table(ensemble_bag_rf_train_table, margin = 1))
  
  ensemble_bag_rf_test_pred <- predict(ensemble_bag_train_rf, ensemble_test, type = "class")
  ensemble_bag_rf_test_table <- table(ensemble_bag_rf_test_pred, ensemble_test$y)
  ensemble_bag_rf_test_accuracy[i] <- sum(diag(ensemble_bag_rf_test_table)) / sum(ensemble_bag_rf_test_table)
  ensemble_bag_rf_test_accuracy_mean <- mean(ensemble_bag_rf_test_accuracy)
  ensemble_bag_rf_test_diag <- sum(diag(ensemble_bag_rf_test_table))
  ensemble_bag_rf_test_mean <- mean(diag(ensemble_bag_rf_test_table)) / mean(ensemble_bag_rf_test_table)
  ensemble_bag_rf_test_sd <- sd(diag(ensemble_bag_rf_test_table)) / sd(ensemble_bag_rf_test_table)
  sum_ensemble_bag_test_rf <- sum(diag(ensemble_bag_rf_test_table))
  ensemble_bag_rf_test_prop <- diag(prop.table(ensemble_bag_rf_test_table, margin = 1))
  
  ensemble_bag_rf_validation_pred <- predict(ensemble_bag_train_rf, ensemble_validation, type = "class")
  ensemble_bag_rf_validation_table <- table(ensemble_bag_rf_validation_pred, ensemble_validation$y)
  ensemble_bag_rf_validation_accuracy[i] <- sum(diag(ensemble_bag_rf_validation_table)) / sum(ensemble_bag_rf_validation_table)
  ensemble_bag_rf_validation_accuracy_mean <- mean(ensemble_bag_rf_validation_accuracy)
  ensemble_bag_rf_validation_diag <- sum(diag(ensemble_bag_rf_validation_table))
  ensemble_bag_rf_validation_mean <- mean(diag(ensemble_bag_rf_validation_table)) / mean(ensemble_bag_rf_validation_table)
  ensemble_bag_rf_validation_sd <- sd(diag(ensemble_bag_rf_validation_table)) / sd(ensemble_bag_rf_validation_table)
  sum_ensemble_bag_validation_rf <- sum(diag(ensemble_bag_rf_validation_table))
  ensemble_bag_rf_validation_prop <- diag(prop.table(ensemble_bag_rf_validation_table, margin = 1))
  
  ensemble_bag_rf_holdout_mean <- mean(c(ensemble_bag_rf_test_accuracy_mean, ensemble_bag_rf_validation_accuracy_mean))
  ensemble_bag_rf_overfitting[i] <- ensemble_bag_rf_holdout_mean / ensemble_bag_rf_train_accuracy_mean
  ensemble_bag_rf_overfitting_mean <- mean(ensemble_bag_rf_overfitting)
  ensemble_bag_rf_overfitting_range <- range(ensemble_bag_rf_overfitting)
  
  ensemble_bag_rf_table <- ensemble_bag_rf_test_table + ensemble_bag_rf_validation_table
  
  ensemble_bag_rf_end <- Sys.time()
  ensemble_bag_rf_duration[i] <- ensemble_bag_rf_end - ensemble_bag_rf_start
  ensemble_bag_rf_duration_mean <- mean(ensemble_bag_rf_duration)
  
  
  #### Ensemble C50 ####
  ensemble_C50_start <- Sys.time()
  
  ensemble_C50_train_fit <- C50::C5.0(ensemble_y_train ~ ., data = ensemble_train)
  ensemble_C50_train_pred <- predict(ensemble_C50_train_fit, ensemble_train)
  ensemble_C50_train_table <- table(ensemble_C50_train_pred, ensemble_y_train)
  ensemble_C50_train_accuracy[i] <- sum(diag(ensemble_C50_train_table)) / sum(ensemble_C50_train_table)
  ensemble_C50_train_accuracy_mean <- mean(ensemble_C50_train_accuracy)
  ensemble_C50_train_mean <- mean(diag(ensemble_C50_train_table)) / mean(ensemble_C50_train_table)
  ensemble_C50_train_sd <- sd(diag(ensemble_C50_train_table)) / sd(ensemble_C50_train_table)
  sum_diag_ensemble_train_C50 <- sum(diag(ensemble_C50_train_table))
  ensemble_C50_train_prop <- diag(prop.table(ensemble_C50_train_table, margin = 1))
  
  ensemble_C50_test_pred <- predict(ensemble_C50_train_fit, ensemble_test)
  ensemble_C50_test_table <- table(ensemble_C50_test_pred, ensemble_y_test)
  ensemble_C50_test_accuracy[i] <- sum(diag(ensemble_C50_test_table)) / sum(ensemble_C50_test_table)
  ensemble_C50_test_accuracy_mean <- mean(ensemble_C50_test_accuracy)
  ensemble_C50_test_mean <- mean(diag(ensemble_C50_test_table)) / mean(ensemble_C50_test_table)
  ensemble_C50_test_sd <- sd(diag(ensemble_C50_test_table)) / sd(ensemble_C50_test_table)
  sum_diag_ensemble_test_C50 <- sum(diag(ensemble_C50_test_table))
  ensemble_C50_test_prop <- diag(prop.table(ensemble_C50_test_table, margin = 1))
  
  ensemble_C50_validation_pred <- predict(ensemble_C50_train_fit, ensemble_validation)
  ensemble_C50_validation_table <- table(ensemble_C50_validation_pred, ensemble_y_validation)
  ensemble_C50_validation_accuracy[i] <- sum(diag(ensemble_C50_validation_table)) / sum(ensemble_C50_validation_table)
  ensemble_C50_validation_accuracy_mean <- mean(ensemble_C50_validation_accuracy)
  ensemble_C50_validation_mean <- mean(diag(ensemble_C50_validation_table)) / mean(ensemble_C50_validation_table)
  ensemble_C50_validation_sd <- sd(diag(ensemble_C50_validation_table)) / sd(ensemble_C50_validation_table)
  sum_diag_ensemble_validation_C50 <- sum(diag(ensemble_C50_validation_table))
  ensemble_C50_validation_prop <- diag(prop.table(ensemble_C50_validation_table, margin = 1))
  
  ensemble_C50_holdout_mean <- mean(c(ensemble_C50_test_accuracy_mean, ensemble_C50_validation_accuracy_mean))
  ensemble_C50_overfitting[i] <- ensemble_C50_holdout_mean / ensemble_C50_train_accuracy_mean
  ensemble_C50_overfitting_mean <- mean(ensemble_C50_overfitting)
  ensemble_C50_overfitting_range <- range(ensemble_C50_overfitting)
  
  ensemble_C50_table <- ensemble_C50_test_table +  ensemble_C50_validation_table
  
  ensemble_C50_end <- Sys.time()
  ensemble_C50_duration[i] <- ensemble_C50_end - ensemble_C50_start
  ensemble_C50_duration_mean <- mean(ensemble_C50_duration)
  
  
  #### Ensemble Earth
  ensemble_earth_start <- Sys.time()
  
  ensemble_earth_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "EarthModel")
  ensemble_earth_train_pred <- predict(ensemble_earth_train_fit, newdata = ensemble_train)
  ensemble_earth_train_table <- table(ensemble_earth_train_pred, ensemble_y_train)
  ensemble_earth_train_accuracy[i] <- sum(diag(ensemble_earth_train_table)) / sum(ensemble_earth_train_table)
  ensemble_earth_train_accuracy_mean <- mean(ensemble_earth_train_accuracy)
  ensemble_earth_train_mean <- mean(diag(ensemble_earth_train_table)) / mean(ensemble_earth_train_table)
  ensemble_earth_train_sd <- sd(diag(ensemble_earth_train_table)) / sd(ensemble_earth_train_table)
  sum_diag_ensemble_train_earth <- sum(diag(ensemble_earth_train_table))
  ensemble_earth_train_prop <- diag(prop.table(ensemble_earth_train_table, margin = 1))
  
  ensemble_earth_test_pred <- predict(ensemble_earth_train_fit, newdata = ensemble_test)
  ensemble_earth_test_table <- table(ensemble_earth_test_pred, ensemble_y_test)
  ensemble_earth_test_accuracy[i] <- sum(diag(ensemble_earth_test_table)) / sum(ensemble_earth_test_table)
  ensemble_earth_test_accuracy_mean <- mean(ensemble_earth_test_accuracy)
  ensemble_earth_test_mean <- mean(diag(ensemble_earth_test_table)) / mean(ensemble_earth_test_table)
  ensemble_earth_test_sd <- sd(diag(ensemble_earth_test_table)) / sd(ensemble_earth_test_table)
  sum_diag_ensemble_test_earth <- sum(diag(ensemble_earth_test_table))
  ensemble_earth_test_prop <- diag(prop.table(ensemble_earth_test_table, margin = 1))
  
  ensemble_earth_validation_pred <- predict(ensemble_earth_train_fit, newdata = ensemble_validation)
  ensemble_earth_validation_table <- table(ensemble_earth_validation_pred, ensemble_y_validation)
  ensemble_earth_validation_accuracy[i] <- sum(diag(ensemble_earth_validation_table)) / sum(ensemble_earth_validation_table)
  ensemble_earth_validation_accuracy_mean <- mean(ensemble_earth_validation_accuracy)
  ensemble_earth_validation_mean <- mean(diag(ensemble_earth_validation_table)) / mean(ensemble_earth_validation_table)
  ensemble_earth_validation_sd <- sd(diag(ensemble_earth_validation_table)) / sd(ensemble_earth_validation_table)
  sum_diag_ensemble_validation_earth <- sum(diag(ensemble_earth_validation_table))
  ensemble_earth_validation_prop <- diag(prop.table(ensemble_earth_validation_table, margin = 1))
  
  ensemble_earth_holdout_mean <- mean(c(ensemble_earth_test_accuracy_mean, ensemble_earth_validation_accuracy_mean))
  ensemble_earth_overfitting[i] <- ensemble_earth_holdout_mean / ensemble_earth_train_accuracy_mean
  ensemble_earth_overfitting_mean <- mean(ensemble_earth_overfitting)
  ensemble_earth_overfitting_range <- range(ensemble_earth_overfitting)
  
  ensemble_earth_table <- ensemble_earth_test_table + ensemble_earth_validation_table
  
  ensemble_earth_end <- Sys.time()
  ensemble_earth_duration[i] <- ensemble_earth_end - ensemble_earth_start
  ensemble_earth_duration_mean <- mean(ensemble_earth_duration)
  
  
  #### Ensembes Using Least Squares Support Vector Machine ####
  ensemble_lssvm_start <- Sys.time()
  
  ensemble_lssvm_train_fit <- kernlab::lssvm(y ~ ., data = ensemble_train)
  ensemble_lssvm_train_pred <- predict(lssvm_train_fit, newdata = train)
  ensemble_lssvm_train_table <- table(lssvm_train_pred, y_train)
  ensemble_lssvm_train_accuracy[i] <- sum(diag(lssvm_train_table)) / sum(lssvm_train_table)
  ensemble_lssvm_train_accuracy_mean <- mean(lssvm_train_accuracy)
  ensemble_lssvm_train_diag <- sum(diag(lssvm_train_table))
  ensemble_lssvm_train_mean <- mean(diag(lssvm_train_table)) / mean(lssvm_train_table)
  ensemble_lssvm_train_sd <- sd(diag(lssvm_train_table)) / sd(lssvm_train_table)
  ensemble_sum_diag_train_lssvm <- sum(diag(lssvm_train_table))
  ensemble_lssvm_train_prop <- diag(prop.table(lssvm_train_table, margin = 1))
  
  ensemble_lssvm_test_pred <- predict(lssvm_train_fit, newdata = test)
  ensemble_lssvm_test_table <- table(lssvm_test_pred, y_test)
  ensemble_lssvm_test_accuracy[i] <- sum(diag(lssvm_test_table)) / sum(lssvm_test_table)
  ensemble_lssvm_test_accuracy_mean <- mean(lssvm_test_accuracy)
  ensemble_lssvm_test_diag <- sum(diag(lssvm_test_table))
  ensemble_lssvm_test_mean <- mean(diag(lssvm_test_table)) / mean(lssvm_test_table)
  ensemble_lssvm_test_sd <- sd(diag(lssvm_test_table)) / sd(lssvm_test_table)
  ensemble_sum_diag_test_lssvm <- sum(diag(lssvm_test_table))
  ensemble_lssvm_test_prop <- diag(prop.table(lssvm_test_table, margin = 1))
  
  ensemble_lssvm_validation_pred <- predict(lssvm_train_fit, newdata = validation)
  ensemble_lssvm_validation_table <- table(lssvm_validation_pred, y_validation)
  ensemble_lssvm_validation_accuracy[i] <- sum(diag(lssvm_validation_table)) / sum(lssvm_validation_table)
  ensemble_lssvm_validation_accuracy_mean <- mean(lssvm_validation_accuracy)
  ensemble_lssvm_validation_diag <- sum(diag(lssvm_validation_table))
  ensemble_lssvm_validation_mean <- mean(diag(lssvm_validation_table)) / mean(lssvm_validation_table)
  ensemble_lssvm_validation_sd <- sd(diag(lssvm_validation_table)) / sd(lssvm_validation_table)
  ensemble_sum_diag_validation_lssvm <- sum(diag(lssvm_validation_table))
  ensemble_lssvm_validation_prop <- diag(prop.table(lssvm_validation_table, margin = 1))
  
  ensemble_lssvm_holdout_mean <- mean(c(lssvm_test_accuracy_mean, lssvm_validation_accuracy_mean))
  ensemble_lssvm_overfitting[i] <- lssvm_holdout_mean / lssvm_train_accuracy_mean
  ensemble_lssvm_overfitting_mean <- mean(lssvm_overfitting)
  ensemble_lssvm_overfitting_range <- range(lssvm_overfitting)
  
  ensemble_lssvm_table <- ensemble_lssvm_test_table + ensemble_lssvm_validation_table
  
  ensemble_lssvm_end <- Sys.time()
  ensemble_lssvm_duration[i] <- ensemble_lssvm_end - ensemble_lssvm_start
  ensemble_lssvm_duration_mean <- mean(ensemble_lssvm_duration)
  
  
  #### Ensemble Naive Bayes ####
  ensemble_n_bayes_start <- Sys.time()
  
  ensemble_n_bayes_train_fit <- naiveBayes(ensemble_y_train ~ ., data = ensemble_train)
  ensemble_n_bayes_train_pred <- predict(ensemble_n_bayes_train_fit, ensemble_train)
  ensemble_n_bayes_train_table <- table(ensemble_n_bayes_train_pred, ensemble_y_train)
  ensemble_n_bayes_train_accuracy[i] <- sum(diag(ensemble_n_bayes_train_table)) / sum(ensemble_n_bayes_train_table)
  ensemble_n_bayes_train_accuracy_mean <- mean(ensemble_n_bayes_train_accuracy)
  ensemble_n_bayes_train_diag <- sum(diag(ensemble_n_bayes_train_table))
  ensemble_n_bayes_train_mean <- mean(diag(ensemble_n_bayes_train_table)) / mean(ensemble_n_bayes_train_table)
  ensemble_n_bayes_train_sd <- sd(diag(ensemble_n_bayes_train_table)) / sd(ensemble_n_bayes_train_table)
  sum_ensemble_n_train_bayes <- sum(diag(ensemble_n_bayes_train_table))
  ensemble_n_bayes_train_prop <- diag(prop.table(ensemble_n_bayes_train_table, margin = 1))
  
  ensemble_n_bayes_test_model <- naiveBayes(ensemble_y_train ~ ., data = ensemble_train)
  ensemble_n_bayes_test_pred <- predict(ensemble_n_bayes_test_model, ensemble_test)
  ensemble_n_bayes_test_table <- table(ensemble_n_bayes_test_pred, ensemble_y_test)
  ensemble_n_bayes_test_accuracy[i] <- sum(diag(ensemble_n_bayes_test_table)) / sum(ensemble_n_bayes_test_table)
  ensemble_n_bayes_test_accuracy_mean <- mean(ensemble_n_bayes_test_accuracy)
  ensemble_n_bayes_test_diag <- sum(diag(ensemble_n_bayes_test_table))
  ensemble_n_bayes_test_mean <- mean(diag(ensemble_n_bayes_test_table)) / mean(ensemble_n_bayes_test_table)
  ensemble_n_bayes_test_sd <- sd(diag(ensemble_n_bayes_test_table)) / sd(ensemble_n_bayes_test_table)
  sum_ensemble_n_test_bayes <- sum(diag(ensemble_n_bayes_test_table))
  ensemble_n_bayes_test_prop <- diag(prop.table(ensemble_n_bayes_test_table, margin = 1))
  
  ensemble_n_bayes_validation_model <- naiveBayes(ensemble_y_train ~ ., data = ensemble_train)
  ensemble_n_bayes_validation_pred <- predict(ensemble_n_bayes_validation_model, ensemble_validation)
  ensemble_n_bayes_validation_table <- table(ensemble_n_bayes_validation_pred, ensemble_y_validation)
  ensemble_n_bayes_validation_accuracy[i] <- sum(diag(ensemble_n_bayes_validation_table)) / sum(ensemble_n_bayes_validation_table)
  ensemble_n_bayes_validation_accuracy_mean <- mean(ensemble_n_bayes_validation_accuracy)
  ensemble_n_bayes_validation_diag <- sum(diag(ensemble_n_bayes_validation_table))
  ensemble_n_bayes_validation_mean <- mean(diag(ensemble_n_bayes_validation_table)) / mean(ensemble_n_bayes_validation_table)
  ensemble_n_bayes_validation_sd <- sd(diag(ensemble_n_bayes_validation_table)) / sd(ensemble_n_bayes_validation_table)
  sum_ensemble_n_validation_bayes <- sum(diag(ensemble_n_bayes_validation_table))
  ensemble_n_bayes_validation_prop <- diag(prop.table(ensemble_n_bayes_validation_table, margin = 1))
  
  ensemble_n_bayes_holdout_mean <- mean(c(ensemble_n_bayes_test_accuracy_mean, ensemble_n_bayes_validation_accuracy_mean))
  ensemble_n_bayes_overfitting[i] <- ensemble_n_bayes_holdout_mean / ensemble_n_bayes_train_accuracy_mean
  ensemble_n_bayes_overfitting_mean <- mean(ensemble_n_bayes_overfitting)
  ensemble_n_bayes_overfitting_range <- range(ensemble_n_bayes_overfitting)
  
  ensemble_n_bayes_table <- ensemble_n_bayes_test_table + ensemble_n_bayes_validation_table
  
  ensemble_n_bayes_end <- Sys.time()
  ensemble_n_bayes_duration[i] <- ensemble_n_bayes_end - ensemble_n_bayes_start
  ensemble_n_bayes_duration_mean <- mean(ensemble_n_bayes_duration)
  
  
  #### Ensemble Ranger Model #####
  ensemble_ranger_start <- Sys.time()
  
  ensemble_ranger_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RangerModel")
  ensemble_ranger_train_pred <- predict(ensemble_ranger_train_fit, newdata = ensemble_train)
  ensemble_ranger_train_table <- table(ensemble_ranger_train_pred, ensemble_y_train)
  ensemble_ranger_train_accuracy[i] <- sum(diag(ensemble_ranger_train_table)) / sum(ensemble_ranger_train_table)
  ensemble_ranger_train_accuracy_mean <- mean(ensemble_ranger_train_accuracy)
  ensemble_ranger_train_diag <- sum(diag(ensemble_ranger_train_table))
  ensemble_ranger_train_mean <- mean(diag(ensemble_ranger_train_table)) / mean(ensemble_ranger_train_table)
  ensemble_ranger_train_sd <- sd(diag(ensemble_ranger_train_table)) / sd(diag(ensemble_ranger_train_table))
  sum_ensemble_train_ranger <- sum(diag(ensemble_ranger_train_table))
  ensemble_ranger_train_prop <- diag(prop.table(ensemble_ranger_train_table, margin = 1))
  
  ensemble_ranger_test_model <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RangerModel")
  ensemble_ranger_test_pred <- predict(ensemble_ranger_test_model, newdata = ensemble_test)
  ensemble_ranger_test_table <- table(ensemble_ranger_test_pred, ensemble_y_test)
  ensemble_ranger_test_accuracy[i] <- sum(diag(ensemble_ranger_test_table)) / sum(ensemble_ranger_test_table)
  ensemble_ranger_test_accuracy_mean <- mean(ensemble_ranger_test_accuracy)
  ensemble_ranger_test_diag <- sum(diag(ensemble_ranger_test_table))
  ensemble_ranger_test_mean <- mean(diag(ensemble_ranger_test_table)) / mean(ensemble_ranger_test_table)
  ensemble_ranger_test_sd <- sd(diag(ensemble_ranger_test_table)) / sd(diag(ensemble_ranger_test_table))
  sum_ensemble_test_ranger <- sum(diag(ensemble_ranger_test_table))
  ensemble_ranger_test_prop <- diag(prop.table(ensemble_ranger_test_table, margin = 1))
  
  ensemble_ranger_validation_model <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RangerModel")
  ensemble_ranger_validation_pred <- predict(ensemble_ranger_validation_model, newdata = ensemble_validation)
  ensemble_ranger_validation_table <- table(ensemble_ranger_validation_pred, ensemble_y_validation)
  ensemble_ranger_validation_accuracy[i] <- sum(diag(ensemble_ranger_validation_table)) / sum(ensemble_ranger_validation_table)
  ensemble_ranger_validation_accuracy_mean <- mean(ensemble_ranger_validation_accuracy)
  ensemble_ranger_validation_diag <- sum(diag(ensemble_ranger_validation_table))
  ensemble_ranger_validation_mean <- mean(diag(ensemble_ranger_validation_table)) / mean(ensemble_ranger_validation_table)
  ensemble_ranger_validation_sd <- sd(diag(ensemble_ranger_validation_table)) / sd(diag(ensemble_ranger_validation_table))
  sum_ensemble_validation_ranger <- sum(diag(ensemble_ranger_validation_table))
  ensemble_ranger_validation_prop <- diag(prop.table(ensemble_ranger_validation_table, margin = 1))
  
  ensemble_ranger_holdout_mean <- mean(c(ensemble_ranger_test_accuracy_mean, ensemble_ranger_validation_accuracy_mean))
  ensemble_ranger_overfitting[i] <- ensemble_ranger_holdout_mean / ensemble_ranger_train_accuracy_mean
  ensemble_ranger_overfitting_mean <- mean(ensemble_ranger_overfitting)
  ensemble_ranger_overfitting_range <- range(ensemble_ranger_overfitting)
  
  ensemble_ranger_table <- ensemble_ranger_test_table + ensemble_ranger_validation_table
  
  ensemble_ranger_end <- Sys.time()
  ensemble_ranger_duration[i] <- ensemble_ranger_end - ensemble_ranger_start
  ensemble_ranger_duration_mean <- mean(ensemble_ranger_duration)
  
  
  #### Ensemble Random Forest ####
  ensemble_rf_start <- Sys.time()
  
  tune_ensemble_train_rf <- tune.randomForest(x = ensemble_train, y = ensemble_y_train)
  ensemble_rf_train_pred <- predict(tune_ensemble_train_rf$best.model, ensemble_train, type = "class")
  ensemble_rf_train_table <- table(ensemble_rf_train_pred, ensemble_y_train)
  ensemble_rf_train_accuracy[i] <- sum(diag(ensemble_rf_train_table)) / sum(ensemble_rf_train_table)
  ensemble_rf_train_accuracy_mean <- mean(ensemble_rf_train_accuracy)
  ensemble_rf_train_diag <- sum(diag(ensemble_rf_train_table))
  ensemble_rf_train_mean <- mean(diag(ensemble_rf_train_table)) / mean(ensemble_rf_train_table)
  ensemble_rf_train_sd <- sd(diag(ensemble_rf_train_table)) / sd(ensemble_rf_train_table)
  sum_ensemble_train_rf <- sum(diag(ensemble_rf_train_table))
  ensemble_rf_train_prop <- diag(prop.table(ensemble_rf_train_table, margin = 1))
  
  ensemble_rf_test_pred <- predict(tune_ensemble_train_rf$best.model, ensemble_test, type = "class")
  ensemble_rf_test_table <- table(ensemble_rf_test_pred, ensemble_y_test)
  ensemble_rf_test_accuracy[i] <- sum(diag(ensemble_rf_test_table)) / sum(ensemble_rf_test_table)
  ensemble_rf_test_accuracy_mean <- mean(ensemble_rf_test_accuracy)
  ensemble_rf_test_diag <- sum(diag(ensemble_rf_test_table))
  ensemble_rf_test_mean <- mean(diag(ensemble_rf_test_table)) / mean(ensemble_rf_test_table)
  ensemble_rf_test_sd <- sd(diag(ensemble_rf_test_table)) / sd(ensemble_rf_test_table)
  sum_ensemble_test_rf <- sum(diag(ensemble_rf_test_table))
  ensemble_rf_test_prop <- diag(prop.table(ensemble_rf_test_table, margin = 1))
  
  ensemble_rf_validation_pred <- predict(tune_ensemble_train_rf$best.model, ensemble_validation, type = "class")
  ensemble_rf_validation_table <- table(ensemble_rf_validation_pred, ensemble_y_validation)
  ensemble_rf_validation_accuracy[i] <- sum(diag(ensemble_rf_validation_table)) / sum(ensemble_rf_validation_table)
  ensemble_rf_validation_accuracy_mean <- mean(ensemble_rf_validation_accuracy)
  ensemble_rf_validation_diag <- sum(diag(ensemble_rf_validation_table))
  ensemble_rf_validation_mean <- mean(diag(ensemble_rf_validation_table)) / mean(ensemble_rf_validation_table)
  ensemble_rf_validation_sd <- sd(diag(ensemble_rf_validation_table)) / sd(ensemble_rf_validation_table)
  sum_ensemble_validation_rf <- sum(diag(ensemble_rf_validation_table))
  ensemble_rf_validation_prop <- diag(prop.table(ensemble_rf_validation_table, margin = 1))
  
  ensemble_rf_holdout_mean <- mean(c(ensemble_rf_test_accuracy_mean, ensemble_rf_validation_accuracy_mean))
  ensemble_rf_overfitting[i] <- ensemble_rf_holdout_mean / ensemble_rf_train_accuracy_mean
  ensemble_rf_overfitting_mean <- mean(ensemble_rf_overfitting)
  ensemble_rf_overfitting_range <- range(ensemble_rf_overfitting)
  
  ensemble_rf_table <- ensemble_rf_test_table + ensemble_rf_validation_table
  
  ensemble_rf_end <- Sys.time()
  ensemble_rf_duration[i] <- ensemble_rf_end - ensemble_rf_start
  ensemble_rf_duration_mean <- mean(ensemble_rf_duration)
  
  
  #### Ensemble Regularized discriminant analysis ####
  ensemble_rda_start <- Sys.time()
  
  tune_ensemble_train_rda <- tune.randomForest(x = ensemble_train, y = ensemble_y_train)
  ensemble_rda_train_pred <- predict(tune_ensemble_train_rda$best.model, ensemble_train, type = "class")
  ensemble_rda_train_table <- table(ensemble_rda_train_pred, ensemble_y_train)
  ensemble_rda_train_accuracy[i] <- sum(diag(ensemble_rda_train_table)) / sum(ensemble_rda_train_table)
  ensemble_rda_train_accuracy_mean <- mean(ensemble_rda_train_accuracy)
  ensemble_rda_train_diag <- sum(diag(ensemble_rda_train_table))
  ensemble_rda_train_mean <- mean(diag(ensemble_rda_train_table)) / mean(ensemble_rda_train_table)
  ensemble_rda_train_sd <- sd(diag(ensemble_rda_train_table)) / sd(ensemble_rda_train_table)
  sum_ensemble_train_rda <- sum(diag(ensemble_rda_train_table))
  ensemble_rda_train_prop <- diag(prop.table(ensemble_rda_train_table, margin = 1))
  
  ensemble_rda_test_pred <- predict(tune_ensemble_train_rda$best.model, ensemble_test)
  ensemble_rda_test_table <- table(ensemble_rda_test_pred, ensemble_y_test)
  ensemble_rda_test_accuracy[i] <- sum(diag(ensemble_rda_test_table)) / sum(ensemble_rda_test_table)
  ensemble_rda_test_accuracy_mean <- mean(ensemble_rda_test_accuracy)
  ensemble_rda_test_mean <- mean(diag(ensemble_rda_test_table)) / mean(ensemble_rda_test_table)
  ensemble_rda_test_sd <- sd(diag(ensemble_rda_test_table)) / sd(ensemble_rda_test_table)
  sum_diag_ensemble_test_rda <- sum(diag(ensemble_rda_test_table))
  ensemble_rda_test_prop <- diag(prop.table(ensemble_rda_test_table, margin = 1))
  
  ensemble_rda_validation_pred <- predict(tune_ensemble_train_rda$best.model, ensemble_validation)
  ensemble_rda_validation_table <- table(ensemble_rda_validation_pred, ensemble_y_validation)
  ensemble_rda_validation_accuracy[i] <- sum(diag(ensemble_rda_validation_table)) / sum(ensemble_rda_validation_table)
  ensemble_rda_validation_accuracy_mean <- mean(ensemble_rda_validation_accuracy)
  ensemble_rda_validation_mean <- mean(diag(ensemble_rda_validation_table)) / mean(ensemble_rda_validation_table)
  ensemble_rda_validation_sd <- sd(diag(ensemble_rda_validation_table)) / sd(ensemble_rda_validation_table)
  sum_diag_ensemble_validation_rda <- sum(diag(ensemble_rda_validation_table))
  ensemble_rda_validation_prop <- diag(prop.table(ensemble_rda_validation_table, margin = 1))
  
  ensemble_rda_holdout_mean <- mean(c(ensemble_rda_test_accuracy_mean, ensemble_rda_validation_accuracy_mean))
  ensemble_rda_overfitting[i] <- ensemble_rda_holdout_mean / ensemble_rda_train_accuracy_mean
  ensemble_rda_overfitting_mean <- mean(ensemble_rda_overfitting)
  ensemble_rda_overfitting_range <- range(ensemble_rda_overfitting)
  
  ensemble_rda_table <- ensemble_rda_test_table + ensemble_rda_validation_table
  
  ensemble_rda_end <- Sys.time()
  ensemble_rda_duration[i] <- ensemble_rda_end- ensemble_rda_start
  ensemble_rda_duration_mean <- mean(ensemble_rda_duration)
  
  
  #### Ensemble Support Vector Machines ####
  ensemble_svm_start <- Sys.time()
  
  ensemble_svm_train_fit <- svm(ensemble_y_train ~ ., data = ensemble_train, kernel = "radial", gamma = 1, cost = 1)
  ensemble_svm_train_pred <- predict(ensemble_svm_train_fit, ensemble_train, type = "class")
  ensemble_svm_train_table <- table(ensemble_svm_train_pred, ensemble_y_train)
  ensemble_svm_train_accuracy[i] <- sum(diag(ensemble_svm_train_table)) / sum(ensemble_svm_train_table)
  ensemble_svm_train_accuracy_mean <- mean(ensemble_svm_train_accuracy)
  ensemble_svm_train_diag <- sum(diag(ensemble_svm_train_table))
  ensemble_svm_train_mean <- mean(diag(ensemble_svm_train_table)) / mean(ensemble_svm_train_table)
  ensemble_svm_train_sd <- sd(diag(ensemble_svm_train_table)) / sd(ensemble_svm_train_table)
  sum_ensemble_train_svm <- sum(diag(ensemble_svm_train_table))
  ensemble_svm_train_prop <- diag(prop.table(ensemble_svm_train_table, margin = 1))
  
  ensemble_svm_test_model <- svm(ensemble_y_train ~ ., data = ensemble_train, kernel = "radial", gamma = 1, cost = 1)
  ensemble_svm_test_pred <- predict(ensemble_svm_test_model, ensemble_test, type = "class")
  ensemble_svm_test_table <- table(ensemble_svm_test_pred, ensemble_y_test)
  ensemble_svm_test_accuracy[i] <- sum(diag(ensemble_svm_test_table)) / sum(ensemble_svm_test_table)
  ensemble_svm_test_accuracy_mean <- mean(ensemble_svm_test_accuracy)
  ensemble_svm_test_diag <- sum(diag(ensemble_svm_test_table))
  ensemble_svm_test_mean <- mean(diag(ensemble_svm_test_table)) / mean(ensemble_svm_test_table)
  ensemble_svm_test_sd <- sd(diag(ensemble_svm_test_table)) / sd(ensemble_svm_test_table)
  sum_ensemble_test_svm <- sum(diag(ensemble_svm_test_table))
  ensemble_svm_test_prop <- diag(prop.table(ensemble_svm_test_table, margin = 1))
  
  ensemble_svm_validation_model <- svm(ensemble_y_train ~ ., data = ensemble_train, kernel = "radial", gamma = 1, cost = 1)
  ensemble_svm_validation_pred <- predict(ensemble_svm_validation_model, ensemble_validation, type = "class")
  ensemble_svm_validation_table <- table(ensemble_svm_validation_pred, ensemble_y_validation)
  ensemble_svm_validation_accuracy[i] <- sum(diag(ensemble_svm_validation_table)) / sum(ensemble_svm_validation_table)
  ensemble_svm_validation_accuracy_mean <- mean(ensemble_svm_validation_accuracy)
  ensemble_svm_validation_diag <- sum(diag(ensemble_svm_validation_table))
  ensemble_svm_validation_mean <- mean(diag(ensemble_svm_validation_table)) / mean(ensemble_svm_validation_table)
  ensemble_svm_validation_sd <- sd(diag(ensemble_svm_validation_table)) / sd(ensemble_svm_validation_table)
  sum_ensemble_validation_svm <- sum(diag(ensemble_svm_validation_table))
  ensemble_svm_validation_prop <- diag(prop.table(ensemble_svm_validation_table, margin = 1))
  
  ensemble_svm_holdout_mean <- mean(c(ensemble_svm_test_accuracy_mean, ensemble_svm_validation_accuracy_mean))
  ensemble_svm_overfitting[i] <- ensemble_svm_holdout_mean / ensemble_svm_train_accuracy_mean
  ensemble_svm_overfitting_mean <- mean(ensemble_svm_overfitting)
  ensemble_svm_overfitting_range <- range(ensemble_svm_overfitting)
  
  ensemble_svm_table <- ensemble_svm_test_table + ensemble_svm_validation_table
  
  ensemble_svm_end <- Sys.time()
  ensemble_svm_duration[i] <- ensemble_svm_end- ensemble_svm_start
  ensemble_svm_duration_mean <- mean(ensemble_svm_duration)
  
  
  #### Ensemble Trees ####
  ensemble_tree_start <- Sys.time()
  
  ensemble_tree_train_fit <- tree(y ~ ., data = ensemble_train)
  ensemble_tree_train_pred <- predict(ensemble_tree_train_fit, ensemble_train, type = "class")
  ensemble_tree_train_table <- table(ensemble_tree_train_pred, ensemble_y_train)
  ensemble_tree_train_accuracy[i] <- sum(diag(ensemble_tree_train_table)) / sum(ensemble_tree_train_table)
  ensemble_tree_train_accuracy_mean <- mean(ensemble_tree_train_accuracy)
  ensemble_tree_train_diag <- sum(diag(ensemble_tree_train_table))
  ensemble_tree_train_mean <- mean(diag(ensemble_tree_train_table)) / mean(ensemble_tree_train_table)
  ensemble_tree_train_sd <- sd(diag(ensemble_tree_train_table)) / sd(ensemble_tree_train_table)
  sum_ensemble_train_tree <- sum(diag(ensemble_tree_train_table))
  ensemble_tree_train_prop <- diag(prop.table(ensemble_tree_train_table, margin = 1))
  
  ensemble_tree_test_pred <- predict(ensemble_tree_train_fit, ensemble_test, type = "class")
  ensemble_tree_test_table <- table(ensemble_tree_test_pred, ensemble_y_test)
  ensemble_tree_test_accuracy[i] <- sum(diag(ensemble_tree_test_table)) / sum(ensemble_tree_test_table)
  ensemble_tree_test_accuracy_mean <- mean(ensemble_tree_test_accuracy)
  ensemble_tree_test_diag <- sum(diag(ensemble_tree_test_table))
  ensemble_tree_test_mean <- mean(diag(ensemble_tree_test_table)) / mean(ensemble_tree_test_table)
  ensemble_tree_test_sd <- sd(diag(ensemble_tree_test_table)) / sd(ensemble_tree_test_table)
  sum_ensemble_test_tree <- sum(diag(ensemble_tree_test_table))
  ensemble_tree_test_prop <- diag(prop.table(ensemble_tree_test_table, margin = 1))
  
  ensemble_tree_validation_pred <- predict(ensemble_tree_train_fit, ensemble_validation, type = "class")
  ensemble_tree_validation_table <- table(ensemble_tree_validation_pred, ensemble_y_validation)
  ensemble_tree_validation_accuracy[i] <- sum(diag(ensemble_tree_validation_table)) / sum(ensemble_tree_validation_table)
  ensemble_tree_validation_accuracy_mean <- mean(ensemble_tree_validation_accuracy)
  ensemble_tree_validation_diag <- sum(diag(ensemble_tree_validation_table))
  ensemble_tree_validation_mean <- mean(diag(ensemble_tree_validation_table)) / mean(ensemble_tree_validation_table)
  ensemble_tree_validation_sd <- sd(diag(ensemble_tree_validation_table)) / sd(ensemble_tree_validation_table)
  sum_ensemble_validation_tree <- sum(diag(ensemble_tree_validation_table))
  ensemble_tree_validation_prop <- diag(prop.table(ensemble_tree_validation_table, margin = 1))
  
  ensemble_tree_holdout_mean <- mean(c(ensemble_tree_test_accuracy_mean, ensemble_tree_validation_accuracy_mean))
  ensemble_tree_overfitting[i] <- ensemble_tree_holdout_mean / ensemble_tree_train_accuracy_mean
  ensemble_tree_overfitting_mean <- mean(ensemble_tree_overfitting)
  ensemble_tree_overfitting_range <- range(ensemble_tree_overfitting)
  
  ensemble_tree_table <- ensemble_tree_test_table + ensemble_tree_validation_table
  
  ensemble_tree_end <- Sys.time()
  ensemble_tree_duration[i] <- ensemble_tree_end - ensemble_tree_start
  ensemble_tree_duration_mean <- mean(ensemble_tree_duration)
  
  
} # Matches the opening parenthesis on line 362. :)

#### Summary table ####

results <- data.frame(
  Model = c('ADA Boost', 'Bagged CART', 'Bagged Random Forest', 'C50', 'Earth', 'Flexible Discriminant Analysis', 'K-Nearest Neighbors',
    'Linear Discriminant Analysis', 'Linear Model', 'Mixed Discriminant Analysis', 'Naive Bayes', 
    'Quadratic Discriminant Analysis', 'Partial Least Squares', 'Penalized Discriminant Analysis', 'Random Foreat', 'Ranger',
    'Regularized Discriminant Analysis', 'RPart', 'Support Vector Machines', 'Trees',
    'Ensemble ADA Bag', 'Ensemble ADA Boost', 'Ensemble Bagged CART', 'Ensemble Bagged Random Forest',
    'Ensemble C50', 'Ensemble Earth', 'Ensemble Least Squared Support Vector Machines', 'Ensemble Naive Bayes',
    'Ensemble Ranger', 'Ensemble Random Forest', 'Ensemble Regularized Discriminant Analysis',
    'Ensemble Support Vector Machines', 'Ensemble Trees'),

  'Mean_Train_Accuracy' = round(c(adaboost_train_accuracy_mean, bag_cart_train_accuracy_mean, bag_rf_train_accuracy_mean, C50_train_accuracy_mean,
    earth_train_accuracy_mean, fda_train_accuracy_mean, knn_train_accuracy_mean, lda_train_accuracy_mean,
    linear_train_accuracy_mean, mda_train_accuracy_mean, n_bayes_train_accuracy_mean,
    qda_train_accuracy_mean, pls_train_accuracy_mean, pda_train_accuracy_mean, rf_train_accuracy_mean, ranger_train_accuracy_mean,
    rda_train_accuracy_mean, rpart_train_accuracy_mean, svm_train_accuracy_mean, tree_train_accuracy_mean,
    ensemble_adabag_train_accuracy_mean, ensemble_adaboost_train_accuracy_mean, ensemble_bag_cart_train_accuracy_mean,
    ensemble_bag_rf_train_accuracy_mean, ensemble_C50_train_accuracy_mean, ensemble_earth_train_accuracy_mean,
    ensemble_lssvm_train_accuracy_mean, ensemble_n_bayes_train_accuracy_mean, ensemble_ranger_train_accuracy_mean,
    ensemble_rf_train_accuracy_mean, ensemble_rda_train_accuracy_mean, ensemble_svm_train_accuracy_mean,
    ensemble_tree_train_accuracy_mean),4),
  
  'Mean_Test_Accuracy' = round(c(adaboost_test_accuracy_mean, bag_cart_test_accuracy_mean, bag_rf_test_accuracy_mean, C50_test_accuracy_mean,
    earth_test_accuracy_mean, fda_test_accuracy_mean, knn_test_accuracy_mean, lda_test_accuracy_mean, linear_test_accuracy_mean,
    mda_test_accuracy_mean, n_bayes_test_accuracy_mean, qda_test_accuracy_mean, pls_test_accuracy_mean, pda_test_accuracy_mean, rf_test_accuracy_mean,
    ranger_test_accuracy_mean, rda_test_accuracy_mean, rpart_test_accuracy_mean, svm_test_accuracy_mean, tree_test_accuracy_mean,
    ensemble_adabag_test_accuracy_mean, ensemble_adaboost_test_accuracy_mean, ensemble_bag_cart_test_accuracy_mean,
    ensemble_bag_rf_test_accuracy_mean, ensemble_C50_test_accuracy_mean, ensemble_earth_test_accuracy_mean,
    ensemble_lssvm_test_accuracy_mean, ensemble_n_bayes_test_accuracy_mean, ensemble_ranger_test_accuracy_mean,
    ensemble_rf_test_accuracy_mean, ensemble_rda_test_accuracy_mean, ensemble_svm_test_accuracy_mean, ensemble_tree_test_accuracy_mean),4),
  
  'Mean_Validation_Accuracy' = round(c(adaboost_validation_accuracy_mean, bag_cart_validation_accuracy_mean, bag_rf_validation_accuracy_mean,
    C50_validation_accuracy_mean, earth_validation_accuracy_mean, fda_validation_accuracy_mean, knn_validation_accuracy_mean,
    lda_validation_accuracy_mean, linear_validation_accuracy_mean, mda_validation_accuracy_mean, n_bayes_validation_accuracy_mean,
    qda_validation_accuracy_mean, pls_validation_accuracy_mean, pda_validation_accuracy_mean, rf_validation_accuracy_mean, ranger_validation_accuracy_mean,
    rda_validation_accuracy_mean, rpart_validation_accuracy_mean, svm_validation_accuracy_mean, tree_validation_accuracy_mean,
    ensemble_adabag_validation_accuracy_mean, ensemble_adaboost_validation_accuracy_mean, ensemble_bag_cart_validation_accuracy_mean,
    ensemble_bag_rf_validation_accuracy_mean, ensemble_C50_validation_accuracy_mean, ensemble_earth_validation_accuracy_mean,
    ensemble_lssvm_validation_accuracy_mean, ensemble_n_bayes_validation_accuracy_mean, ensemble_ranger_validation_accuracy_mean,
    ensemble_rf_validation_accuracy_mean, ensemble_rda_validation_accuracy_mean, ensemble_svm_validation_accuracy_mean,
    ensemble_tree_validation_accuracy_mean),4),
  
  'Mean_Holdout' = round(c(adaboost_holdout_mean, bag_cart_holdout_mean, bag_rf_holdout_mean, C50_holdout_mean, earth_holdout_mean,
    fda_holdout_mean, knn_holdout_mean, lda_holdout_mean, linear_holdout_mean, mda_holdout_mean, n_bayes_holdout_mean,
    qda_holdout_mean, pls_holdout_mean, pda_holdout_mean, rf_holdout_mean, ranger_holdout_mean, rda_holdout_mean, rpart_holdout_mean,
    svm_holdout_mean, tree_holdout_mean, ensemble_adabag_holdout_mean, ensemble_adaboost_holdout_mean,
    ensemble_bag_cart_holdout_mean, ensemble_bag_rf_holdout_mean, ensemble_C50_holdout_mean, ensemble_earth_holdout_mean,
    ensemble_lssvm_holdout_mean, ensemble_n_bayes_holdout_mean, ensemble_ranger_holdout_mean, ensemble_rf_holdout_mean,
    ensemble_rda_holdout_mean, ensemble_svm_holdout_mean, ensemble_tree_holdout_mean),4),
  
  'Mean_overfitting' = round(c(adaboost_overfitting_mean, bag_cart_overfitting_mean, bag_rf_overfitting_mean, C50_overfitting_mean,
    earth_overfitting_mean, fda_overfitting_mean, knn_overfitting_mean, lda_overfitting_mean, linear_overfitting_mean,
    mda_overfitting_mean, n_bayes_overfitting_mean, qda_overfitting_mean, pls_overfitting_mean, pda_overfitting_mean, rf_overfitting_mean,
    ranger_overfitting_mean, rda_overfitting_mean, rpart_overfitting_mean, svm_overfitting_mean, tree_overfitting_mean,
    ensemble_adabag_overfitting_mean, ensemble_adaboost_overfitting_mean, ensemble_bag_cart_overfitting_mean,
    ensemble_bag_rf_overfitting_mean, ensemble_C50_overfitting_mean, ensemble_earth_overfitting_mean,
    ensemble_lssvm_overfitting_mean, ensemble_n_bayes_overfitting_mean, ensemble_ranger_overfitting_mean,
    ensemble_rf_overfitting_mean, ensemble_rda_overfitting_mean, ensemble_svm_overfitting_mean, ensemble_tree_overfitting_mean),4),
  
  'Min_overfitting' = round(c(adaboost_overfitting_range[1], bag_cart_overfitting_range[1], bag_rf_overfitting_range[1], C50_overfitting_range[1],
    earth_overfitting_range[1], fda_overfitting_range[1], knn_overfitting_range[1], lda_overfitting_range[1],
    linear_overfitting_range[1], mda_overfitting_range[1], n_bayes_overfitting_range[1], qda_overfitting_range[1],
    pls_overfitting_range[1], pda_overfitting_range[1], rf_overfitting_range[1], ranger_overfitting_range[1], rda_overfitting_range[1],
    rpart_overfitting_range[1], svm_overfitting_range[1], tree_overfitting_range[1], ensemble_adabag_overfitting_range[1],
    ensemble_adaboost_overfitting_range[1], ensemble_bag_cart_overfitting_range[1], ensemble_bag_rf_overfitting_range[1],
    ensemble_C50_overfitting_range[1], ensemble_earth_overfitting_range[1], ensemble_lssvm_overfitting_range[1],
    ensemble_n_bayes_overfitting_range[1], ensemble_ranger_overfitting_range[1], ensemble_rf_overfitting_range[1],
    ensemble_rda_overfitting_range[1], ensemble_svm_overfitting_range[1], ensemble_tree_overfitting_range[1]),4),
  
  'Max_overfitting' = round(c(adaboost_overfitting_range[2], bag_cart_overfitting_range[2], bag_rf_overfitting_range[2], C50_overfitting_range[2],
    earth_overfitting_range[2], fda_overfitting_range[2], knn_overfitting_range[2], lda_overfitting_range[2],
    linear_overfitting_range[2], mda_overfitting_range[2], n_bayes_overfitting_range[2], qda_overfitting_range[2],
    pls_overfitting_range[2], pda_overfitting_range[2], rf_overfitting_range[2], ranger_overfitting_range[2], rda_overfitting_range[2],
    rpart_overfitting_range[2], svm_overfitting_range[2], tree_overfitting_range[2], ensemble_adabag_overfitting_range[2],
    ensemble_adaboost_overfitting_range[2], ensemble_bag_cart_overfitting_range[2], ensemble_bag_rf_overfitting_range[2],
    ensemble_C50_overfitting_range[2], ensemble_earth_overfitting_range[2], ensemble_lssvm_overfitting_range[2],
    ensemble_n_bayes_overfitting_range[2], ensemble_ranger_overfitting_range[2], ensemble_rf_overfitting_range[2],
    ensemble_rda_overfitting_range[2], ensemble_svm_overfitting_range[2], ensemble_tree_overfitting_range[2]),4),
  
  'Diagonal_Sum_of_Test_Data' = round(c(sum_diag_test_adaboost, sum_diag_bag_test_cart, sum_diag_test_bag_rf, sum_diag_test_C50,
    sum_diag_test_earth, sum_diag_test_fda, sum_diag_test_knn, sum_diag_test_lda, sum_diag_test_linear,
    sum_diag_test_mda, sum_diag_n_test_bayes, sum_diag_test_qda, sum_diag_test_pls, sum_diag_test_pda, sum_diag_test_rf, sum_diag_test_ranger,
    sum_diag_test_rda, sum_diag_test_rpart, sum_diag_test_svm, sum_diag_test_tree, ensemble_adabag_test_diag,
    ensemble_adaboost_test_diag, ensemble_sum_diag_bag_test_cart, ensemble_bag_rf_test_diag, sum_diag_ensemble_test_C50,
    sum_diag_ensemble_test_earth, ensemble_lssvm_test_diag, ensemble_n_bayes_test_diag, ensemble_ranger_test_diag,
    ensemble_rf_test_diag, sum_diag_ensemble_test_rda, ensemble_svm_test_diag, ensemble_tree_test_diag),4),
  
  'Diagonal_Sum_of_Validation_Data' = round(c(sum_diag_validation_adaboost, sum_diag_bag_validation_cart, sum_diag_validation_bag_rf,
    sum_diag_validation_C50, sum_diag_validation_earth, sum_diag_validation_fda, sum_diag_validation_knn,
    sum_diag_validation_lda, sum_diag_validation_linear, sum_diag_validation_mda, sum_diag_n_validation_bayes,
    sum_diag_validation_qda, sum_diag_validation_pls, sum_diag_validation_pda, sum_diag_validation_rf, sum_diag_validation_ranger,
    sum_diag_validation_rda, sum_diag_validation_rpart, sum_diag_validation_svm, sum_diag_validation_tree,
    ensemble_adabag_validation_diag, ensemble_adaboost_validation_diag, ensemble_sum_diag_bag_validation_cart,
    ensemble_bag_rf_validation_diag, sum_diag_ensemble_validation_C50, sum_diag_ensemble_validation_earth,
    ensemble_lssvm_validation_diag, ensemble_n_bayes_validation_diag, ensemble_ranger_validation_diag,
    ensemble_rf_validation_diag, sum_diag_ensemble_validation_rda, ensemble_svm_validation_diag, ensemble_tree_validation_diag),4),
  
  'Mean_duration' = round(c(adaboost_duration_mean, bag_cart_duration_mean, bag_rf_duration_mean, C50_duration_mean, earth_duration_mean,
    fda_duration_mean, knn_duration_mean, lda_duration_mean, linear_duration_mean, mda_duration_mean, n_bayes_duration_mean,
    qda_duration_mean, pls_duration_mean, pda_duration_mean, rf_duration_mean, ranger_duration_mean, rda_duration_mean, rpart_duration_mean,
    svm_duration_mean, tree_duration_mean, ensemble_adabag_duration_mean, ensemble_adaboost_duration_mean,
    ensemble_bag_cart_duration_mean, ensemble_bag_rf_duration_mean, ensemble_C50_duration_mean, ensemble_earth_duration_mean,
    ensemble_lssvm_duration_mean, ensemble_n_bayes_duration_mean, ensemble_ranger_duration_mean, ensemble_rf_duration_mean,
    ensemble_rda_duration_mean, ensemble_svm_duration_mean, ensemble_tree_duration_mean),4),
  
  'Final_Model' = c('adaboost_train_fit', 'bag_cart_train_fit', 'bag_rf_train_fit', 'C50_train_fit', 'earth_train_fit', 'fda_train_fit',
    'knn_train_fit', 'lda_train_fit', 'linear_train_fit', 'mda_train_fit', 'n_bayes_train_fit',
    'qda_train_fit', 'pls_train_fit', 'pda_train_fit', 'rf_train_fit', 'ranger_train_fit', 'rda_train_fit', 'rpart_train_fit',
    'svm_train_fit', 'tree_train_fit', 'ensemble_adabag_train_fit', 'ensemble_adaboost_train_fit',
    'ensemble_bag_cart_train_fit', 'ensemble_bag_rf_train_fit', 'ensemble_C50_train_fit', 'ensemble_earth_train_fit',
    'ensemble_lssvm_train_fit', 'ensemble_n_bayes_train_fit', 'ensemble_ranger_train_fit', 'ensemble_rf_train_fit',
    'ensemble_rda_train_fit', 'ensemble_svm_train_fit', 'ensemble_tree_train_fit'),
  
  'Summary_Table' = c('adaboost_table', 'bag_cart_table', 'bag_rf_table', 'C50_table', 'earth_table', 'fda_table', 'knn_table', 'lda_table',
    'linear_table', 'mda_table', 'n_bayes_table', 'qda_table', 'pls_table', 'pda_table', 'rf_table', 'ranger_table', 'rda_table',
    'rpart_table', 'svm_table', 'tree_table', 'ensemble_adabag_table', 'ensemble_adaboost_table', 'ensemble_bag_cart_table',
    'ensemble_bag_rf_table', 'ensemble_C50_table', 'ensemble_earth_table', 'ensemble_lssvm_table', 'ensemble_n_bayes_table',
    'ensemble_ranger_table', 'ensemble_rf_table',
    'ensemble_rda_table', 'ensemble_svm_table', 'ensemble_tree_table')
)


results <- results %>% arrange(desc(Mean_Holdout))
reactable(results, searchable = TRUE, pagination = FALSE, wrap = TRUE, fullWidth = TRUE, filterable = TRUE, bordered = TRUE,
    striped = TRUE, highlight = TRUE, rownames = TRUE)

Test_Proportions <- rbind('LDA' = lda_test_prop, 'QDA' = qda_test_prop, 'KNN' = knn_test_prop, 'MDA' = mda_test_prop,
    'FDA' = fda_test_prop, 'RDA' = rda_test_prop, 'C50' = C50_test_prop, 'Tree' = tree_test_prop, 'Support Vector Machines' = svm_test_prop,
    'Naive Bayes' = n_bayes_test_prop, 'Bagged Cart' = bag_cart_test_prop, 'ADA Boost' = adaboost_test_prop,
    'Ensemble Trees' = ensemble_tree_test_prop, 'Ensemble Random Forest' = ensemble_rf_test_prop,
    'Ensemble Bagged Random Forest' = ensemble_bag_rf_test_prop, 'Ensemble Support Vector Machines' = ensemble_svm_test_prop,
    'Ensemble Naive Bayes' = ensemble_n_bayes_test_prop, 'Ensemble RDA' = ensemble_rda_test_prop,
    'Ensemble C50' = ensemble_C50_test_prop, 'Ensemble Bagged Cart' = ensemble_bag_cart_test_prop,
    'Ensemble ADA Boost' = ensemble_adaboost_test_prop)

Test_Proportions

Validation_Proportions <- rbind('LDA' = lda_validation_prop, 'QDA' = qda_validation_prop, 'KNN' = knn_validation_prop,
    'MDA' = mda_validation_prop, 'FDA' = fda_validation_prop, 'RDA' = rda_validation_prop, 'C50' = C50_validation_prop,
    'Tree' = tree_validation_prop, 'Support Vector Machines' = svm_validation_prop, 'Naive Bayes' = n_bayes_validation_prop,
    'Bagged Cart' = bag_cart_validation_prop, 'ADA Boost' = adaboost_validation_prop,
    'Ensemble Trees' = ensemble_tree_validation_prop, 'Ensemble Random Forest' = ensemble_rf_validation_prop,
    'Ensemble Bagged Random Forest' = ensemble_bag_rf_validation_prop,
    'Ensemble Support Vector Machines' = ensemble_svm_validation_prop, 'Ensemble Naive Bayes' = ensemble_n_bayes_validation_prop,
    'Ensemble RDA' = ensemble_rda_validation_prop, 'Ensemble C50' = ensemble_C50_validation_prop,
    'Ensemble Bagged Cart' = ensemble_bag_cart_validation_prop,'Ensemble ADA Boost' = ensemble_adaboost_validation_prop)

Validation_Proportions
