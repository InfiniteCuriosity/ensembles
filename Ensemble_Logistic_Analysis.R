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


library(adabag)
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
library(klaR)
library(MASS) # for linear discriminant analysis
library(mda)
library(naivebayes) # for Naive Bayes
library(neuralnet) # for Neural Networks
library(pls) # for Principal Components Regression
library(randomForest) # for random forests
library(reactable) # for output formats
library(tidyverse)
library(tree) # for decision tree

adabag_train_accuracy <- 0
adabag_test_accuracy <- 0
adabag_validation_accuracy <- 0
adabag_overfitting <- 0
#adabag_pred <- 0
#adabag_accuracy <- 0
adabag_no <- 0
adabag_yes <- 0
adabag_train_no <- 0
adabag_train_yes <- 0
adabag_test_no <- 0
adabag_test_yes <- 0
adabag_validation_no <- 0
adabag_validation_yes <- 0
#adabag_sum_diag <- 0
#adabag_overfitting_no <- 0
adabag_overfitting_yes <- 0
adabag_holdout <- 0

adaboost_train_accuracy <- 0
adaboost_test_accuracy <- 0
adaboost_validation_accuracy <- 0
adaboost_overfitting <- 0
#adaboost_pred <- 0
#adaboost_accuracy <- 0
adaboost_no <- 0
adaboost_yes <- 0
adaboost_train_no <- 0
adaboost_train_yes <- 0
adaboost_test_no <- 0
adaboost_test_yes <- 0
adaboost_validation_no <- 0
adaboost_validation_yes <- 0
#adaboost_sum_diag <- 0
#adaboost_overfitting_no <- 0
#adaboost_overfitting_yes <- 0
adaboost_holdout <- 0

bagging_train_accuracy <- 0
bagging_test_accuracy <- 0
bagging_validation_accuracy <- 0
bagging_overfitting <- 0
bagging_pred <- 0
bagging_accuracy <- 0
bagging_no <- 0
bagging_yes <- 0
bagging_train_no <- 0
bagging_train_yes <- 0
bagging_test_no <- 0
bagging_test_yes <- 0
bagging_validation_no <- 0
bagging_validation_yes <- 0
bagging_sum_diag <- 0
bagging_overfitting_no <- 0
bagging_overfitting_yes <- 0
bagging_holdout <- 0

#bag_cart_train_accuracy <- 0
#bag_cart_test_accuracy <- 0
bag_cart_validation_accuracy <- 0
#bag_cart_overfitting <- 0
#bag_cart_pred <- 0
bag_cart_accuracy <- 0
bag_cart_no <- 0
bag_cart_yes <- 0
bag_cart_train_no <- 0
bag_cart_train_yes <- 0
#bag_cart_test_no <- 0
#bag_cart_test_yes <- 0
bag_cart_validation_no <- 0
bag_cart_validation_yes <- 0
#bag_cart_sum_diag <- 0
#bag_cart_overfitting_no <- 0
#bag_cart_overfitting_yes <- 0
bag_cart_holdout <- 0

bag_rf_train_accuracy <- 0
bag_rf_test_accuracy <- 0
bag_rf_validation_accuracy <- 0
bag_rf_overfitting <- 0
bag_rf_pred <- 0
bag_rf_accuracy <- 0
bag_rf_no <- 0
bag_rf_yes <- 0
bag_rf_sum_diag <- 0
bag_rf_overfitting_no <- 0
bag_rf_overfitting_yes <- 0
bag_rf_holdout <- 0
bag_rf_train_no <- 0
bag_rf_train_yes <- 0
bag_rf_test_no <- 0
bag_rf_test_yes <- 0
bag_rf_validation_no <- 0
bag_rf_validation_yes <- 0

C50_train_accuracy <- 0
C50_test_accuracy <- 0
C50_validation_accuracy <- 0
C50_overfitting <- 0
C50_pred <- 0
C50_accuracy <- 0
C50_no <- 0
C50_yes <- 0
C50_sum_diag <- 0
C50_overfitting_no <- 0
C50_overfitting_yes <- 0
C50_holdout <- 0
C50_train_no <- 0
C50_train_yes <- 0
C50_test_no <- 0
C50_test_yes <- 0
C50_validation_no <- 0
C50_validation_yes <- 0

earth_train_accuracy <- 0
earth_test_accuracy <- 0
earth_validation_accuracy <- 0
earth_overfitting <- 0
earth_pred <- 0
earth_accuracy <- 0
earth_no <- 0
earth_yes <- 0
earth_sum_diag <- 0
earth_overfitting_no <- 0
earth_overfitting_yes <- 0
earth_holdout <- 0
earth_train_no <- 0
earth_train_yes <- 0
earth_test_no <- 0
earth_test_yes <- 0
earth_validation_no <- 0
earth_validation_yes <- 0

fda_train_accuracy <- 0
fda_test_accuracy <- 0
fda_validation_accuracy <- 0
fda_overfitting <- 0
fda_pred <- 0
fda_accuracy <- 0
fda_no <- 0
fda_yes <- 0
fda_sum_diag <- 0
fda_overfitting_no <- 0
fda_overfitting_yes <- 0
fda_holdout <- 0
fda_train_no <- 0
fda_train_yes <- 0
fda_test_no <- 0
fda_test_yes <- 0
fda_validation_no <- 0
fda_validation_yes <- 0

lda_train_accuracy <- 0
lda_test_accuracy <- 0
lda_validation_accuracy <- 0
lda_overfitting <- 0
lda_pred <- 0
lda_accuracy <- 0
lda_no <- 0
lda_yes <- 0
lda_sum_diag <- 0
lda_overfitting_no <- 0
lda_overfitting_yes <- 0
lda_holdout <- 0
lda_train_no <- 0
lda_train_yes <- 0
lda_test_no <- 0
lda_test_yes <- 0
lda_validation_no <- 0
lda_validation_yes <- 0

linear_train_accuracy <- 0
linear_test_accuracy <- 0
linear_validation_accuracy <- 0
linear_overfitting <- 0
linear_pred <- 0
linear_accuracy <- 0
linear_no <- 0
linear_yes <- 0
linear_sum_diag <- 0
linear_overfitting_no <- 0
linear_overfitting_yes <- 0
linear_holdout <- 0
linear_train_no <- 0
linear_train_yes <- 0
linear_test_no <- 0
linear_test_yes <- 0
linear_validation_no <- 0
linear_validation_yes <- 0

mda_train_accuracy <- 0
mda_test_accuracy <- 0
mda_validation_accuracy <- 0
mda_overfitting <- 0
mda_pred <- 0
mda_accuracy <- 0
mda_no <- 0
mda_yes <- 0
mda_sum_diag <- 0
mda_overfitting_no <- 0
mda_overfitting_yes <- 0
mda_holdout <- 0
mda_train_no <- 0
mda_train_yes <- 0
mda_test_no <- 0
mda_test_yes <- 0
mda_validation_no <- 0
mda_validation_yes <- 0

n_bayes_train_accuracy <- 0
n_bayes_test_accuracy <- 0
n_bayes_validation_accuracy <- 0
n_bayes_overfitting <- 0
n_bayes_pred <- 0
n_bayes_accuracy <- 0
n_bayes_no <- 0
n_bayes_yes <- 0
n_bayes_sum_diag <- 0
n_bayes_overfitting_no <- 0
n_bayes_overfitting_yes <- 0
n_bayes_holdout <- 0
n_bayes_train_no <- 0
n_bayes_train_yes <- 0
n_bayes_test_no <- 0
n_bayes_test_yes <- 0
n_bayes_validation_no <- 0
n_bayes_validation_yes <- 0

pls_train_accuracy <- 0
pls_test_accuracy <- 0
pls_validation_accuracy <- 0
pls_overfitting <- 0
pls_pred <- 0
pls_accuracy <- 0
pls_no <- 0
pls_yes <- 0
pls_sum_diag <- 0
pls_overfitting_no <- 0
pls_overfitting_yes <- 0
pls_holdout <- 0
pls_train_no <- 0
pls_train_yes <- 0
pls_test_no <- 0
pls_test_yes <- 0
pls_validation_no <- 0
pls_validation_yes <- 0

pda_train_accuracy <- 0
pda_test_accuracy <- 0
pda_validation_accuracy <- 0
pda_overfitting <- 0
pda_pred <- 0
pda_accuracy <- 0
pda_no <- 0
pda_yes <- 0
pda_sum_diag <- 0
pda_overfitting_no <- 0
pda_overfitting_yes <- 0
pda_holdout <- 0
pda_train_no <- 0
pda_train_yes <- 0
pda_test_no <- 0
pda_test_yes <- 0
pda_validation_no <- 0
pda_validation_yes <- 0

qda_train_accuracy <- 0
qda_test_accuracy <- 0
qda_validation_accuracy <- 0
qda_overfitting <- 0
qda_pred <- 0
qda_accuracy <- 0
qda_no <- 0
qda_yes <- 0
qda_sum_diag <- 0
qda_overfitting_no <- 0
qda_overfitting_yes <- 0
qda_holdout <- 0
qda_train_no <- 0
qda_train_yes <- 0
qda_test_no <- 0
qda_test_yes <- 0
qda_validation_no <- 0
qda_validation_yes <- 0

rf_train_accuracy <- 0
rf_test_accuracy <- 0
rf_validation_accuracy <- 0
rf_overfitting <- 0
rf_pred <- 0
rf_accuracy <- 0
rf_no <- 0
rf_yes <- 0
rf_sum_diag <- 0
rf_overfitting_no <- 0
rf_overfitting_yes <- 0
rf_holdout <- 0
rf_train_no <- 0
rf_train_yes <- 0
rf_test_no <- 0
rf_test_yes <- 0
rf_validation_no <- 0
rf_validation_yes <- 0

ranger_train_accuracy <- 0
ranger_test_accuracy <- 0
ranger_validation_accuracy <- 0
ranger_overangeritting <- 0
ranger_pred <- 0
ranger_accuracy <- 0
ranger_no <- 0
ranger_yes <- 0
ranger_sum_diag <- 0
ranger_overfitting <- 0
ranger_overfitting_no <- 0
ranger_overfitting_yes <- 0
ranger_holdout <- 0
ranger_train_no <- 0
ranger_train_yes <- 0
ranger_test_no <- 0
ranger_test_yes <- 0
ranger_validation_no <- 0
ranger_validation_yes <- 0

rda_train_accuracy <- 0
rda_test_accuracy <- 0
rda_validation_accuracy <- 0
rda_overdaitting <- 0
rda_pred <- 0
rda_accuracy <- 0
rda_no <- 0
rda_yes <- 0
rda_sum_diag <- 0
rda_overfitting <- 0
rda_overfitting_no <- 0
rda_overfitting_yes <- 0
rda_holdout <- 0
rda_train_no <- 0
rda_train_yes <- 0
rda_test_no <- 0
rda_test_yes <- 0
rda_validation_no <- 0
rda_validation_yes <- 0

rpart_train_accuracy <- 0
rpart_test_accuracy <- 0
rpart_validation_accuracy <- 0
rpart_overpartitting <- 0
rpart_pred <- 0
rpart_accuracy <- 0
rpart_no <- 0
rpart_yes <- 0
rpart_sum_diag <- 0
rpart_overfitting <- 0
rpart_overfitting_no <- 0
rpart_overftitting_yes <- 0
rpart_holdout <- 0
rpart_train_no <- 0
rpart_train_yes <- 0
rpart_test_no <- 0
rpart_test_yes <- 0
rpart_validation_no <- 0
rpart_validation_yes <- 0

svm_train_accuracy <- 0
svm_test_accuracy <- 0
svm_validation_accuracy <- 0
svm_ovesvmitting <- 0
svm_pred <- 0
svm_accuracy <- 0
svm_no <- 0
svm_yes <- 0
svm_sum_diag <- 0
svm_overfitting <- 0
svm_overfiting_no <- 0
svm_overfitting_yes <- 0
svm_holdout <- 0
svm_train_no <- 0
svm_train_yes <- 0
svm_test_no <- 0
svm_test_yes <- 0
svm_validation_no <- 0
svm_validation_yes <- 0

tree_train_accuracy <- 0
tree_test_accuracy <- 0
tree_validation_accuracy <- 0
tree_ovetreeitting <- 0
tree_pred <- 0
tree_accuracy <- 0
tree_no <- 0
tree_yes <- 0
tree_sum_diag <- 0
tree_overfitting <- 0
tree_overfiting_no <- 0
tree_overfitting_yes <- 0
tree_holdout <- 0
tree_train_no <- 0
tree_train_yes <- 0
tree_test_no <- 0
tree_test_yes <- 0
tree_validation_no <- 0
tree_validation_yes <- 0

ensemble_adabag_train_accuracy <- 0
ensemble_adabag_test_accuracy <- 0
ensemble_adabag_validation_accuracy <- 0
ensemble_adabag_overfitting <- 0
#ensemble_adabag_pred <- 0
#ensemble_adabag_accuracy <- 0
ensemble_adabag_no <- 0
ensemble_adabag_yes <- 0
#ensemble_adabag_sum_diag <- 0
#ensemble_adabag_overfitting_no <- 0
ensemble_adabag_overfitting_yes <- 0
ensemble_adabag_holdout <- 0
ensemble_adabag_train_no <- 0
ensemble_adabag_train_yes <- 0
ensemble_adabag_test_no <- 0
ensemble_adabag_test_yes <- 0
ensemble_adabag_validation_no <- 0
ensemble_adabag_validation_yes <- 0

ensemble_adaboost_train_accuracy <- 0
ensemble_adaboost_test_accuracy <- 0
ensemble_adaboost_validation_accuracy <- 0
ensemble_adaboost_overfitting <- 0
ensemble_adaboost_pred <- 0
ensemble_adaboost_accuracy <- 0
ensemble_adaboost_no <- 0
ensemble_adaboost_yes <- 0
ensemble_adaboost_sum_diag <- 0
ensemble_adaboost_overfitting_no <- 0
ensemble_adaboost_overfitting_yes <- 0
ensemble_adaboost_holdout <- 0
ensemble_adaboost_train_no <- 0
ensemble_adaboost_train_yes <- 0
ensemble_adaboost_test_no <- 0
ensemble_adaboost_test_yes <- 0
ensemble_adaboost_validation_no <- 0
ensemble_adaboost_validation_yes <- 0

ensemble_bagging_train_accuracy <- 0
ensemble_bagging_test_accuracy <- 0
ensemble_bagging_validation_accuracy <- 0
ensemble_bagging_overfitting <- 0
ensemble_bagging_pred <- 0
ensemble_bagging_accuracy <- 0
ensemble_bagging_no <- 0
ensemble_bagging_yes <- 0
ensemble_bagging_sum_diag <- 0
ensemble_bagging_overfitting_no <- 0
ensemble_bagging_overfitting_yes <- 0
ensemble_bagging_holdout <- 0
ensemble_bagging_train_no <- 0
ensemble_bagging_train_yes <- 0
ensemble_bagging_test_no <- 0
ensemble_bagging_test_yes <- 0
ensemble_bagging_validation_no <- 0
ensemble_bagging_validation_yes <- 0

ensemble_bag_rf_train_accuracy <- 0
ensemble_bag_rf_test_accuracy <- 0
ensemble_bag_rf_validation_accuracy <- 0
ensemble_bag_rf_overfitting <- 0
ensemble_bag_rf_pred <- 0
ensemble_bag_rf_accuracy <- 0
ensemble_bag_rf_no <- 0
ensemble_bag_rf_yes <- 0
ensemble_bag_rf_sum_diag <- 0
ensemble_bag_rf_overfitting_no <- 0
ensemble_bag_rf_overfitting_yes <- 0
ensemble_bag_rf_holdout <- 0
ensemble_bag_rf_train_no <- 0
ensemble_bag_rf_train_yes <- 0
ensemble_bag_rf_test_no <- 0
ensemble_bag_rf_test_yes <- 0
ensemble_bag_rf_validation_no <- 0
ensemble_bag_rf_validation_yes <- 0

ensemble_C50_train_accuracy <- 0
ensemble_C50_test_accuracy <- 0
ensemble_C50_validation_accuracy <- 0
ensemble_C50_overfitting <- 0
ensemble_C50_pred <- 0
ensemble_C50_accuracy <- 0
ensemble_C50_no <- 0
ensemble_C50_yes <- 0
ensemble_C50_sum_diag <- 0
ensemble_C50_overfitting_no <- 0
ensemble_C50_overfitting_yes <- 0
ensemble_C50_holdout <- 0
ensemble_C50_train_no <- 0
ensemble_C50_train_yes <- 0
ensemble_C50_test_no <- 0
ensemble_C50_test_yes <- 0
ensemble_C50_validation_no <- 0
ensemble_C50_validation_yes <- 0

ensemble_earth_train_accuracy <- 0
ensemble_earth_test_accuracy <- 0
ensemble_earth_validation_accuracy <- 0
ensemble_earth_overfitting <- 0
ensemble_earth_pred <- 0
ensemble_earth_accuracy <- 0
ensemble_earth_no <- 0
ensemble_earth_yes <- 0
ensemble_earth_sum_diag <- 0
ensemble_earth_overfitting_no <- 0
ensemble_earth_overfitting_yes <- 0
ensemble_earth_holdout <- 0
ensemble_earth_train_no <- 0
ensemble_earth_train_yes <- 0
ensemble_earth_test_no <- 0
ensemble_earth_test_yes <- 0
ensemble_earth_validation_no <- 0
ensemble_earth_validation_yes <- 0

ensemble_fda_train_accuracy <- 0
ensemble_fda_test_accuracy <- 0
ensemble_fda_validation_accuracy <- 0
ensemble_fda_overfitting <- 0
ensemble_fda_pred <- 0
ensemble_fda_accuracy <- 0
ensemble_fda_no <- 0
ensemble_fda_yes <- 0
ensemble_fda_sum_diag <- 0
ensemble_fda_overfitting_no <- 0
ensemble_fda_overfitting_yes <- 0
ensemble_fda_holdout <- 0
ensemble_fda_train_no <- 0
ensemble_fda_train_yes <- 0
ensemble_fda_test_no <- 0
ensemble_fda_test_yes <- 0
ensemble_fda_validation_no <- 0
ensemble_fda_validation_yes <- 0

ensemble_linear_train_accuracy <- 0
ensemble_linear_test_accuracy <- 0
ensemble_linear_validation_accuracy <- 0
ensemble_linear_overfitting <- 0
ensemble_linear_pred <- 0
ensemble_linear_accuracy <- 0
ensemble_linear_no <- 0
ensemble_linear_yes <- 0
ensemble_linear_sum_diag <- 0
ensemble_linear_overfitting_no <- 0
ensemble_linear_overfitting_yes <- 0
ensemble_linear_holdout <- 0
ensemble_linear_train_no <- 0
ensemble_linear_train_yes <- 0
ensemble_linear_test_no <- 0
ensemble_linear_test_yes <- 0
ensemble_linear_validation_no <- 0
ensemble_linear_validation_yes <- 0

ensemble_mda_train_accuracy <- 0
ensemble_mda_test_accuracy <- 0
ensemble_mda_validation_accuracy <- 0
ensemble_mda_overfitting <- 0
ensemble_mda_pred <- 0
ensemble_mda_accuracy <- 0
ensemble_mda_no <- 0
ensemble_mda_yes <- 0
ensemble_mda_sum_diag <- 0
ensemble_mda_overfitting_no <- 0
ensemble_mda_overfitting_yes <- 0
ensemble_mda_holdout <- 0
ensemble_mda_train_no <- 0
ensemble_mda_train_yes <- 0
ensemble_mda_test_no <- 0
ensemble_mda_test_yes <- 0
ensemble_mda_validation_no <- 0
ensemble_mda_validation_yes <- 0

ensemble_n_bayes_train_accuracy <- 0
ensemble_n_bayes_test_accuracy <- 0
ensemble_n_bayes_validation_accuracy <- 0
ensemble_n_bayes_overfitting <- 0
ensemble_n_bayes_pred <- 0
ensemble_n_bayes_accuracy <- 0
ensemble_n_bayes_no <- 0
ensemble_n_bayes_yes <- 0
ensemble_n_bayes_sum_diag <- 0
ensemble_n_bayes_overfitting_no <- 0
ensemble_n_bayes_overfitting_yes <- 0
ensemble_n_bayes_holdout <- 0
ensemble_n_bayes_train_no <- 0
ensemble_n_bayes_train_yes <- 0
ensemble_n_bayes_test_no <- 0
ensemble_n_bayes_test_yes <- 0
ensemble_n_bayes_validation_no <- 0
ensemble_n_bayes_validation_yes <- 0

ensemble_n_bayes_train_accuracy <- 0
ensemble_n_bayes_test_accuracy <- 0
ensemble_n_bayes_validation_accuracy <- 0
ensemble_n_bayes_overfitting <- 0
ensemble_n_bayes_pred <- 0
ensemble_n_bayes_accuracy <- 0
ensemble_n_bayes_no <- 0
ensemble_n_bayes_yes <- 0
ensemble_n_bayes_sum_diag <- 0
ensemble_n_bayes_overfitting_no <- 0
ensemble_n_bayes_overfitting_yes <- 0
ensemble_n_bayes_holdout <- 0
ensemble_n_bayes_train_no <- 0
ensemble_n_bayes_train_yes <- 0
ensemble_n_bayes_test_no <- 0
ensemble_n_bayes_test_yes <- 0
ensemble_n_bayes_validation_no <- 0
ensemble_n_bayes_validation_yes <- 0

ensemble_pls_train_accuracy <- 0
ensemble_pls_test_accuracy <- 0
ensemble_pls_validation_accuracy <- 0
ensemble_pls_overfitting <- 0
ensemble_pls_pred <- 0
ensemble_pls_accuracy <- 0
ensemble_pls_no <- 0
ensemble_pls_yes <- 0
ensemble_pls_sum_diag <- 0
ensemble_pls_overfitting_no <- 0
ensemble_pls_overfitting_yes <- 0
ensemble_pls_holdout <- 0
ensemble_pls_train_no <- 0
ensemble_pls_train_yes <- 0
ensemble_pls_test_no <- 0
ensemble_pls_test_yes <- 0
ensemble_pls_validation_no <- 0
ensemble_pls_validation_yes <- 0

ensemble_pda_train_accuracy <- 0
ensemble_pda_test_accuracy <- 0
ensemble_pda_validation_accuracy <- 0
ensemble_pda_overfitting <- 0
ensemble_pda_pred <- 0
ensemble_pda_accuracy <- 0
ensemble_pda_no <- 0
ensemble_pda_yes <- 0
ensemble_pda_sum_diag <- 0
ensemble_pda_overfitting_no <- 0
ensemble_pda_overfitting_yes <- 0
ensemble_pda_holdout <- 0
ensemble_pda_train_no <- 0
ensemble_pda_train_yes <- 0
ensemble_pda_test_no <- 0
ensemble_pda_test_yes <- 0
ensemble_pda_validation_no <- 0
ensemble_pda_validation_yes <- 0

ensemble_rf_train_accuracy <- 0
ensemble_rf_test_accuracy <- 0
ensemble_rf_validation_accuracy <- 0
ensemble_rf_overfitting <- 0
ensemble_rf_pred <- 0
ensemble_rf_accuracy <- 0
ensemble_rf_no <- 0
ensemble_rf_yes <- 0
ensemble_rf_sum_diag <- 0
ensemble_rf_overfitting_no <- 0
ensemble_rf_overfitting_yes <- 0
ensemble_rf_holdout <- 0
ensemble_rf_train_no <- 0
ensemble_rf_train_yes <- 0
ensemble_rf_test_no <- 0
ensemble_rf_test_yes <- 0
ensemble_rf_validation_no <- 0
ensemble_rf_validation_yes <- 0

ensemble_ranger_train_accuracy <- 0
ensemble_ranger_test_accuracy <- 0
ensemble_ranger_validation_accuracy <- 0
ensemble_ranger_overfitting <- 0
ensemble_ranger_pred <- 0
ensemble_ranger_accuracy <- 0
ensemble_ranger_no <- 0
ensemble_ranger_yes <- 0
ensemble_ranger_sum_diag <- 0
ensemble_ranger_overfitting_no <- 0
ensemble_ranger_overfitting_yes <- 0
ensemble_ranger_holdout <- 0
ensemble_ranger_train_no <- 0
ensemble_ranger_train_yes <- 0
ensemble_ranger_test_no <- 0
ensemble_ranger_test_yes <- 0
ensemble_ranger_validation_no <- 0
ensemble_ranger_validation_yes <- 0

ensemble_rda_train_accuracy <- 0
ensemble_rda_test_accuracy <- 0
ensemble_rda_validation_accuracy <- 0
ensemble_rda_overfitting <- 0
ensemble_rda_pred <- 0
ensemble_rda_accuracy <- 0
ensemble_rda_no <- 0
ensemble_rda_yes <- 0
ensemble_rda_sum_diag <- 0
ensemble_rda_overfitting_no <- 0
ensemble_rda_overfitting_yes <- 0
ensemble_rda_holdout <- 0
ensemble_rda_train_no <- 0
ensemble_rda_train_yes <- 0
ensemble_rda_test_no <- 0
ensemble_rda_test_yes <- 0
ensemble_rda_validation_no <- 0
ensemble_rda_validation_yes <- 0

ensemble_rpart_train_accuracy <- 0
ensemble_rpart_test_accuracy <- 0
ensemble_rpart_validation_accuracy <- 0
ensemble_rpart_overfitting <- 0
ensemble_rpart_pred <- 0
ensemble_rpart_accuracy <- 0
ensemble_rpart_no <- 0
ensemble_rpart_yes <- 0
ensemble_rpart_sum_diag <- 0
ensemble_rpart_overfitting_no <- 0
ensemble_rpart_overfitting_yes <- 0
ensemble_rpart_holdout <- 0
ensemble_rpart_train_no <- 0
ensemble_rpart_train_yes <- 0
ensemble_rpart_test_no <- 0
ensemble_rpart_test_yes <- 0
ensemble_rpart_validation_no <- 0
ensemble_rpart_validation_yes <- 0

ensemble_svm_train_accuracy <- 0
ensemble_svm_test_accuracy <- 0
ensemble_svm_validation_accuracy <- 0
ensemble_svm_overfitting <- 0
ensemble_svm_pred <- 0
ensemble_svm_accuracy <- 0
ensemble_svm_no <- 0
ensemble_svm_yes <- 0
ensemble_svm_sum_diag <- 0
ensemble_svm_overfitting_no <- 0
ensemble_svm_overfitting_yes <- 0
ensemble_svm_holdout <- 0
ensemble_svm_train_no <- 0
ensemble_svm_train_yes <- 0
ensemble_svm_test_no <- 0
ensemble_svm_test_yes <- 0
ensemble_svm_validation_no <- 0
ensemble_svm_validation_yes <- 0

ensemble_tree_train_accuracy <- 0
ensemble_tree_test_accuracy <- 0
ensemble_tree_validation_accuracy <- 0
ensemble_tree_overfitting <- 0
ensemble_tree_pred <- 0
ensemble_tree_accuracy <- 0
ensemble_tree_no <- 0
ensemble_tree_yes <- 0
ensemble_tree_sum_diag <- 0
ensemble_tree_overfitting_no <- 0
ensemble_tree_overfitting_yes <- 0
ensemble_tree_holdout <- 0
ensemble_tree_train_no <- 0
ensemble_tree_train_yes <- 0
ensemble_tree_test_no <- 0
ensemble_tree_test_yes <- 0
ensemble_tree_validation_no <- 0
ensemble_tree_validation_yes <- 0

set.seed(31415) # to make results exactly reproducible

# Best models for a logistic feature

# Get the data

df <- bind_rows(Pima.te, Pima.tr)
y = df$type

View(df) # To see the data

#### Barchart of the data against type ####
df %>% 
  mutate(across(-type, as.numeric)) %>% 
  pivot_longer(-type, names_to = "var", values_to = "value") %>%
  ggplot(aes(x = type, y = value)) +
  geom_col() +
  facet_wrap(~var, scales = "free") +
  labs(title="Numerical values against type")

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
pairs(df, main = "Scatterplot matrices of the data")

#### Boxplots of the numeric data ####
df1 %>%
  gather(key = "var", value = "value") %>%
  ggplot(aes(x = '',y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~ var, scales = "free") +
  theme_bw() +
  labs(title = "Boxplots of the numeric data")
# Thanks to https://rstudio-pubs-static.s3.amazonaws.com/388596_e21196f1adf04e0ea7cd68edd9eba966.html

#### Histograms of the numeric data ####
df1 <- df %>% select_if(is.numeric)
ggplot(gather(df1, cols, value), aes(x = value)) +
  geom_histogram(bins = round(nrow(df1)/10)) + 
  facet_wrap(.~cols, scales = "free") +
  labs(title = "Histograms of each numeric column. Each bar = 10 rows of data")



#### Break into test and train set ####

for (i in 1:5) {
  
  index <- sample(c(1:3), nrow(df), replace=TRUE, prob=c(0.6, 0.2, 0.2))
  
  train  <- df[index == 1, ]
  test   <- df[index == 2, ]
  validation = df[index == 3,]
  
  train01 <- train # needed to run xgboost
  test01 <- test # needed to run xgboost
  validation01 <- validation
  
  y_train <- train$type
  y_test <- test$type
  y_validation <- validation$type
  
  train  <- df[index == 1, ] %>% dplyr::select(-type)
  test   <- df[index == 2, ] %>% dplyr::select(-type)
  validation <- df[index == 3, ] %>% dplyr::select(-type)
  
  #### Adabag ####
  
  adabag_train_fit <- adabag::bagging(type ~ ., data = train01)
  
  adabag_train_pred <- predict(adabag_train_fit, train01)
  adabag_train_table <- table(adabag_train_pred$class, y_train)
  adabag_train_accuracy[i] <- sum(diag(adabag_train_table)) / sum(adabag_train_table)
  adabag_train_accuracy_mean <- mean(adabag_train_accuracy)
  adabag_train_mean <- mean(diag(adabag_train_table)) / mean(adabag_train_table)
  adabag_train_sd <- sd(diag(adabag_train_table)) / sd(adabag_train_table)
  adabag_train_sum_diag <- sum(diag(adabag_train_table))
  adabag_train_prop <- diag(prop.table(adabag_train_table, margin = 1))
  adabag_train_no[i] <- as.numeric(adabag_train_prop[1])
  adabag_train_no_mean <- mean(adabag_train_no)
  adabag_train_yes[i] <- as.numeric(adabag_train_prop[2])
  adabag_train_yes_mean <- mean(adabag_train_yes)
  
  adabag_test_pred <- predict(adabag_train_fit, test01)
  adabag_test_table <- table(adabag_test_pred$class, y_test)
  adabag_test_accuracy[i] <- sum(diag(adabag_test_table)) / sum(adabag_test_table)
  adabag_test_accuracy_mean <- mean(adabag_test_accuracy)
  adabag_test_mean <- mean(diag(adabag_test_table)) / mean(adabag_test_table)
  adabag_test_sd <- sd(diag(adabag_test_table)) / sd(adabag_test_table)
  adabag_test_sum_diag <- sum(diag(adabag_test_table))
  adabag_test_prop <- diag(prop.table(adabag_test_table, margin = 1))
  adabag_test_no[i] <- as.numeric(adabag_test_prop[1])
  adabag_test_no_mean <- mean(adabag_test_no)
  adabag_test_yes[i] <- as.numeric(adabag_test_prop[2])
  adabag_test_yes_mean <- mean(adabag_test_yes)
  
  adabag_validation_pred <- predict(adabag_train_fit, validation01)
  adabag_validation_table <- table(adabag_validation_pred$class, y_validation)
  adabag_validation_accuracy[i] <- sum(diag(adabag_validation_table)) / sum(adabag_validation_table)
  adabag_validation_accuracy_mean <- mean(adabag_validation_accuracy)
  adabag_validation_mean <- mean(diag(adabag_validation_table)) / mean(adabag_validation_table)
  adabag_validation_sd <- sd(diag(adabag_validation_table)) / sd(adabag_validation_table)
  adabag_validation_sum_diag <- sum(diag(adabag_validation_table))
  adabag_validation_prop <- diag(prop.table(adabag_validation_table, margin = 1))
  adabag_validation_no[i] <- as.numeric(adabag_validation_prop[1])
  adabag_validation_no_mean <- mean(adabag_validation_no)
  adabag_validation_yes[i] <- as.numeric(adabag_validation_prop[2])
  adabag_validation_yes_mean <- mean(adabag_validation_yes)
  
  adabag_no[i] <- mean(c(adabag_test_no, adabag_validation_no))
  adabag_no_mean <- mean(c(adabag_no))
  
  adabag_yes[i] <- mean(c(adabag_test_yes, adabag_validation_yes))
  adabag_yes_mean <- mean(c(adabag_yes))
  
  adabag_holdout[i] <- mean(c(adabag_test_accuracy_mean, adabag_validation_accuracy_mean))
  adabag_holdout_mean <- mean(adabag_holdout)
  adabag_overfitting[i] <- adabag_holdout_mean / adabag_train_accuracy_mean
  adabag_overfitting_mean <- mean(adabag_overfitting)
  adabag_overfitting_range <- range(adabag_overfitting)
  
  adabag_no_mean <- mean(c(adabag_test_no, adabag_validation_no))
  adabag_yes_mean <- mean(c(adabag_test_yes, adabag_validation_yes))
  
  adabag_table <- adabag_test_table + adabag_validation_table
  
  adabag_table_sum_diag <- sum(diag(adabag_table))
  
  
  #### Adaboost ####
  adaboost_train_fit <- MachineShop::fit(formula = type ~ ., data = train01, model = "AdaBoostModel")
  
  adaboost_train_pred <- predict(adaboost_train_fit, train01)
  adaboost_train_table <- table(adaboost_train_pred, y_train)
  adaboost_train_accuracy[i] <- sum(diag(adaboost_train_table)) / sum(adaboost_train_table)
  adaboost_train_accuracy_mean <- mean(adaboost_train_accuracy)
  adaboost_train_mean <- mean(diag(adaboost_train_table)) / mean(adaboost_train_table)
  adaboost_train_sd <- sd(diag(adaboost_train_table)) / sd(adaboost_train_table)
  adaboost_train_sum_diag <- sum(diag(adaboost_train_table))
  adaboost_train_prop <- diag(prop.table(adaboost_train_table, margin = 1))
  adaboost_train_no[i] <- as.numeric(adaboost_train_prop[1])
  adaboost_train_no_mean <- mean(adaboost_train_no)
  adaboost_train_yes[i] <- as.numeric(adaboost_train_prop[2])
  adaboost_train_yes_mean <- mean(adaboost_train_yes)
  
  adaboost_test_pred <- predict(adaboost_train_fit, test01)
  adaboost_test_table <- table(adaboost_test_pred, y_test)
  adaboost_test_accuracy[i] <- sum(diag(adaboost_test_table)) / sum(adaboost_test_table)
  adaboost_test_accuracy_mean <- mean(adaboost_test_accuracy)
  adaboost_test_mean <- mean(diag(adaboost_test_table)) / mean(adaboost_test_table)
  adaboost_test_sd <- sd(diag(adaboost_test_table)) / sd(adaboost_test_table)
  adaboost_test_sum_diag <- sum(diag(adaboost_test_table))
  adaboost_test_prop <- diag(prop.table(adaboost_test_table, margin = 1))
  adaboost_test_no[i] <- as.numeric(adaboost_test_prop[1])
  adaboost_test_no_mean <- mean(adaboost_test_no)
  adaboost_test_yes[i] <- as.numeric(adaboost_test_prop[2])
  adaboost_test_yes_mean <- mean(adaboost_test_yes)
  
  adaboost_validation_pred <- predict(adaboost_train_fit, validation01)
  adaboost_validation_table <- table(adaboost_validation_pred, y_validation)
  adaboost_validation_accuracy[i] <- sum(diag(adaboost_validation_table)) / sum(adaboost_validation_table)
  adaboost_validation_accuracy_mean <- mean(adaboost_validation_accuracy)
  adaboost_validation_mean <- mean(diag(adaboost_validation_table)) / mean(adaboost_validation_table)
  adaboost_validation_sd <- sd(diag(adaboost_validation_table)) / sd(adaboost_validation_table)
  adaboost_validation_sum_diag <- sum(diag(adaboost_validation_table))
  adaboost_validation_prop <- diag(prop.table(adaboost_validation_table, margin = 1))
  adaboost_validation_no[i] <- as.numeric(adaboost_validation_prop[1])
  adaboost_validation_no_mean <- mean(adaboost_validation_no)
  adaboost_validation_yes[i] <- as.numeric(adaboost_validation_prop[2])
  adaboost_validation_yes_mean <- mean(adaboost_validation_yes)
  
  adaboost_no[i] <- mean(c(adaboost_test_no, adaboost_validation_no))
  adaboost_no_mean <- mean(adaboost_no)
  
  adaboost_yes[i] <- mean(c(adaboost_test_yes, adaboost_validation_yes))
  adaboost_yes_mean <- mean(adaboost_yes)
  
  adaboost_holdout[i] <- mean(c(adaboost_test_accuracy, adaboost_validation_accuracy))
  adaboost_holdout_mean <- mean(adaboost_holdout)
  adaboost_overfitting[i] <- adaboost_holdout_mean / adaboost_train_accuracy_mean
  adaboost_overfitting_mean <- mean(adaboost_overfitting)
  adaboost_overfitting_range <- range(adaboost_overfitting)
  
  adaboost_yes_mean <- mean(c(adaboost_test_yes_mean, adaboost_validation_yes_mean))
  adaboost_no_mean <- mean(c(adaboost_test_no_mean, adaboost_validation_no_mean))
  
  adaboost_table <- adaboost_test_table + adaboost_validation_table
  
  adaboost_table_sum_diag <- sum(diag(adaboost_table))
  
  
  #### Bagging ####
  bagging_train_fit <- ipred::bagging(type ~ ., data = train01, coob = TRUE)
  
  bagging_train_pred <- predict(bagging_train_fit, train01)
  bagging_train_table <- table(bagging_train_pred, y_train)
  bagging_train_accuracy[i] <- sum(diag(bagging_train_table)) / sum(bagging_train_table)
  bagging_train_accuracy_mean <- mean(bagging_train_accuracy)
  bagging_train_mean <- mean(diag(bagging_train_table)) / mean(bagging_train_table)
  bagging_train_sd <- sd(diag(bagging_train_table)) / sd(bagging_train_table)
  bagging_train_sum_diag <- sum(diag(bagging_train_table))
  bagging_train_prop <- diag(prop.table(bagging_train_table, margin = 1))
  bagging_train_no[i] <- as.numeric(bagging_train_prop[1])
  bagging_train_no_mean <- mean(bagging_train_no)
  bagging_train_yes[i] <- as.numeric(bagging_train_prop[2])
  bagging_train_yes_mean <- mean(bagging_train_yes)
  
  bagging_test_pred <- predict(bagging_train_fit, test01)
  bagging_test_table <- table(bagging_test_pred, y_test)
  bagging_test_accuracy[i] <- sum(diag(bagging_test_table)) / sum(bagging_test_table)
  bagging_test_accuracy_mean <- mean(bagging_test_accuracy)
  bagging_test_mean <- mean(diag(bagging_test_table)) / mean(bagging_test_table)
  bagging_test_sd <- sd(diag(bagging_test_table)) / sd(bagging_test_table)
  bagging_test_sum_diag <- sum(diag(bagging_test_table))
  bagging_test_prop <- diag(prop.table(bagging_test_table, margin = 1))
  bagging_test_no[i] <- as.numeric(bagging_test_prop[1])
  bagging_test_no_mean <- mean(bagging_test_no)
  bagging_test_yes[i] <- as.numeric(bagging_test_prop[2])
  bagging_test_yes_mean <- mean(bagging_test_yes)
  
  bagging_validation_pred <- predict(bagging_train_fit, validation01)
  bagging_validation_table <- table(bagging_validation_pred, y_validation)
  bagging_validation_accuracy[i] <- sum(diag(bagging_validation_table)) / sum(bagging_validation_table)
  bagging_validation_accuracy_mean <- mean(bagging_validation_accuracy)
  bagging_validation_mean <- mean(diag(bagging_validation_table)) / mean(bagging_validation_table)
  bagging_validation_sd <- sd(diag(bagging_validation_table)) / sd(bagging_validation_table)
  bagging_validation_sum_diag <- sum(diag(bagging_validation_table))
  bagging_validation_prop <- diag(prop.table(bagging_validation_table, margin = 1))
  bagging_validation_no[i] <- as.numeric(bagging_validation_prop[1])
  bagging_validation_no_mean <- mean(bagging_validation_no)
  bagging_validation_yes[i] <- as.numeric(bagging_validation_prop[2])
  bagging_validation_yes_mean <- mean(bagging_validation_yes)
  
  bagging_no[i] <- mean(c(bagging_test_no, bagging_validation_no))
  bagging_no_mean <- mean(bagging_no)
  
  bagging_yes[i] <- mean(c(bagging_test_yes, bagging_validation_yes))
  bagging_yes_mean <- mean(bagging_yes)
  
  bagging_holdout[i] <- mean(c(bagging_test_accuracy_mean, bagging_validation_accuracy_mean))
  bagging_holdout_mean <- mean(bagging_holdout)
  bagging_overfitting[i] <- bagging_holdout_mean / bagging_train_accuracy_mean
  bagging_overfitting_mean <- mean(bagging_overfitting)
  bagging_overfitting_range <- range(bagging_overfitting)
  
  bagging_no_mean <- mean(c(bagging_test_no, bagging_validation_no))
  bagging_yes_mean <- mean(c(bagging_test_yes, bagging_validation_yes))
  
  bagging_table <- bagging_test_table + bagging_validation_table
  
  bagging_table_sum_diag <- sum(diag(bagging_table))
  
  
  #### Bagged Random Forest ####
  bag_rf_train_fit <- randomForest(y_train ~ ., data = train01)
  
  bag_rf_train_pred <- predict(bag_rf_train_fit, train01)
  bag_rf_train_table <- table(bag_rf_train_pred, y_train)
  bag_rf_train_accuracy[i] <- sum(diag(bag_rf_train_table)) / sum(bag_rf_train_table)
  bag_rf_train_accuracy_mean <- mean(bag_rf_train_accuracy)
  bag_rf_train_mean <- mean(diag(bag_rf_train_table)) / mean(bag_rf_train_table)
  bag_rf_train_sd <- sd(diag(bag_rf_train_table)) / sd(bag_rf_train_table)
  bag_rf_train_sum_diag <- sum(diag(bag_rf_train_table))
  bag_rf_train_prop <- diag(prop.table(bag_rf_train_table, margin = 1))
  bag_rf_train_no[i] <- as.numeric(bag_rf_train_prop[1])
  bag_rf_train_no_mean <- mean( bag_rf_train_no)
  bag_rf_train_yes[i] <- as.numeric(bag_rf_train_prop[2])
  bag_rf_train_yes_mean <- mean(bag_rf_train_yes)
  
  bag_rf_test_pred <- predict(bag_rf_train_fit, test01)
  bag_rf_test_table <- table(bag_rf_test_pred, y_test)
  bag_rf_test_accuracy[i] <- sum(diag(bag_rf_test_table)) / sum(bag_rf_test_table)
  bag_rf_test_accuracy_mean <- mean(bag_rf_test_accuracy)
  bag_rf_test_mean <- mean(diag(bag_rf_test_table)) / mean(bag_rf_test_table)
  bag_rf_test_sd <- sd(diag(bag_rf_test_table)) / sd(bag_rf_test_table)
  bag_rf_test_sum_diag <- sum(diag(bag_rf_test_table))
  bag_rf_test_prop <- diag(prop.table(bag_rf_test_table, margin = 1))
  bag_rf_test_no[i] <- as.numeric(bag_rf_test_prop[1])
  bag_rf_test_no_mean <- mean(bag_rf_test_no)
  bag_rf_test_yes[i] <- as.numeric(bag_rf_test_prop[2])
  bag_rf_test_yes_mean <- mean(bag_rf_test_yes)
  
  bag_rf_validation_pred <- predict(bag_rf_train_fit, validation01)
  bag_rf_validation_table <- table(bag_rf_validation_pred, y_validation)
  bag_rf_validation_accuracy[i] <- sum(diag(bag_rf_validation_table)) / sum(bag_rf_validation_table)
  bag_rf_validation_accuracy_mean <- mean(bag_rf_validation_accuracy)
  bag_rf_validation_mean <- mean(diag(bag_rf_validation_table)) / mean(bag_rf_validation_table)
  bag_rf_validation_sd <- sd(diag(bag_rf_validation_table)) / sd(bag_rf_validation_table)
  bag_rf_validation_sum_diag <- sum(diag(bag_rf_validation_table))
  bag_rf_validation_prop <- diag(prop.table(bag_rf_validation_table, margin = 1))
  bag_rf_validation_no[i] <- as.numeric(bag_rf_validation_prop[1])
  bag_rf_validation_no_mean <- mean(bag_rf_validation_no)
  bag_rf_validation_yes[i] <- as.numeric(bag_rf_validation_prop[2])
  bag_rf_validation_yes_mean <- mean(bag_rf_validation_yes)
  
  bag_rf_holdout[i] <- mean(c(bag_rf_test_accuracy_mean, bag_rf_validation_accuracy_mean))
  bag_rf_holdout_mean <- mean(bag_rf_holdout)
  bag_rf_overfitting[i] <- bag_rf_holdout_mean / bag_rf_train_accuracy_mean
  bag_rf_overfitting_mean <- mean(bag_rf_overfitting)
  bag_rf_overfitting_range <- range(bag_rf_overfitting)
  
  bag_rf_no <- mean(c(bag_rf_test_no, bag_rf_validation_no))
  bag_rf_yes <- mean(c(bag_rf_test_yes, bag_rf_validation_yes))
  
  bag_rf_table <- bag_rf_test_table + bag_rf_validation_table
  
  bag_rf_table_sum_diag <- sum(diag(bag_rf_table))
  
  
  #### C50 ####
  C50_train_fit <- C5.0(as.factor(y_train) ~ ., data = train)
  
  C50_train_pred <- predict(C50_train_fit, train01)
  C50_train_table <- table(C50_train_pred, y_train)
  C50_train_accuracy[i] <- sum(diag(C50_train_table)) / sum(C50_train_table)
  C50_train_accuracy_mean <- mean(C50_train_accuracy)
  C50_train_mean <- mean(diag(C50_train_table)) / mean(C50_train_table)
  C50_train_sd <- sd(diag(C50_train_table)) / sd(C50_train_table)
  C50_train_sum_diag <- sum(diag(C50_train_table))
  C50_train_prop <- diag(prop.table(C50_train_table, margin = 1))
  C50_train_no[i] <- as.numeric(C50_train_prop[1])
  C50_train_no_mean <- mean(C50_train_no)
  C50_train_yes[i] <- as.numeric(C50_train_prop[2])
  C50_train_yes_mean <- mean(C50_train_yes)
  
  C50_test_pred <- predict(C50_train_fit, test01)
  C50_test_table <- table(C50_test_pred, y_test)
  C50_test_accuracy[i] <- sum(diag(C50_test_table)) / sum(C50_test_table)
  C50_test_accuracy_mean <- mean(C50_test_accuracy)
  C50_test_mean <- mean(diag(C50_test_table)) / mean(C50_test_table)
  C50_test_sd <- sd(diag(C50_test_table)) / sd(C50_test_table)
  C50_test_sum_diag <- sum(diag(C50_test_table))
  C50_test_prop <- diag(prop.table(C50_test_table, margin = 1))
  C50_test_no[i] <- as.numeric(C50_test_prop[1])
  C50_test_no_mean <- mean(C50_test_no)
  C50_test_yes[i] <- as.numeric(C50_test_prop[2])
  C50_test_yes_mean <- mean(C50_test_yes)
  
  C50_validation_pred <- predict(C50_train_fit, validation01)
  C50_validation_table <- table(C50_validation_pred, y_validation)
  C50_validation_accuracy[i] <- sum(diag(C50_validation_table)) / sum(C50_validation_table)
  C50_validation_accuracy_mean <- mean(C50_validation_accuracy)
  C50_validation_mean <- mean(diag(C50_validation_table)) / mean(C50_validation_table)
  C50_validation_sd <- sd(diag(C50_validation_table)) / sd(C50_validation_table)
  C50_validation_sum_diag <- sum(diag(C50_validation_table))
  C50_validation_prop <- diag(prop.table(C50_validation_table, margin = 1))
  C50_validation_no[i] <- as.numeric(C50_validation_prop[1])
  C50_validation_no_mean <- mean(C50_validation_no)
  C50_validation_yes[i] <- as.numeric(C50_validation_prop[2])
  C50_validation_yes_mean <- mean(C50_validation_yes)
  
  C50_no[i] <- mean(c(C50_test_no, C50_validation_no))
  C50_no_mean <- mean(C50_no)
  
  C50_yes[i] <- mean(c(C50_test_yes, C50_validation_yes))
  C50_yes_mean <- mean(C50_yes)
  
  C50_holdout[i] <- mean(c(C50_test_accuracy_mean, C50_validation_accuracy_mean))
  C50_holdout_mean <- mean(C50_holdout)
  C50_overfitting[i] <- C50_holdout_mean / C50_train_accuracy_mean
  C50_overfitting_mean <- mean(C50_overfitting)
  C50_overfitting_range <- range(C50_overfitting)
  
  C50_table <- C50_test_table + C50_validation_table
  
  C50_table_sum_diag <- sum(diag(C50_table))
  
  
  #### Earth model ####
  earth_train_fit <- MachineShop::fit(type ~ ., data = train01, model = "EarthModel")
  
  earth_train_pred <- predict(earth_train_fit, train01)
  earth_train_table <- table(earth_train_pred, y_train)
  earth_train_accuracy[i] <- sum(diag(earth_train_table)) / sum(earth_train_table)
  earth_train_accuracy_mean <- mean(earth_train_accuracy)
  earth_train_mean <- mean(diag(earth_train_table)) / mean(earth_train_table)
  earth_train_sd <- sd(diag(earth_train_table)) / sd(earth_train_table)
  earth_train_sum_diag <- sum(diag(earth_train_table))
  earth_train_prop <- diag(prop.table(earth_train_table, margin = 1))
  earth_train_no[i] <- as.numeric(earth_train_prop[1])
  earth_train_no_mean <- mean(earth_train_no)
  earth_train_yes[i] <- as.numeric(earth_train_prop[2])
  earth_train_yes_mean <- mean(earth_train_yes)
  
  earth_test_pred <- predict(earth_train_fit, test01)
  earth_test_table <- table(earth_test_pred, y_test)
  earth_test_accuracy[i] <- sum(diag(earth_test_table)) / sum(earth_test_table)
  earth_test_accuracy_mean <- mean(earth_test_accuracy)
  earth_test_mean <- mean(diag(earth_test_table)) / mean(earth_test_table)
  earth_test_sd <- sd(diag(earth_test_table)) / sd(earth_test_table)
  earth_test_sum_diag <- sum(diag(earth_test_table))
  earth_test_prop <- diag(prop.table(earth_test_table, margin = 1))
  earth_test_no[i] <- as.numeric(earth_test_prop[1])
  earth_test_no_mean <- mean(earth_test_no)
  earth_test_yes[i] <- as.numeric(earth_test_prop[2])
  earth_test_yes_mean <- mean(earth_test_yes)
  
  earth_validation_pred <- predict(earth_train_fit, validation01)
  earth_validation_table <- table(earth_validation_pred, y_validation)
  earth_validation_accuracy[i] <- sum(diag(earth_validation_table)) / sum(earth_validation_table)
  earth_validation_accuracy_mean <- mean(earth_validation_accuracy)
  earth_validation_mean <- mean(diag(earth_validation_table)) / mean(earth_validation_table)
  earth_validation_sd <- sd(diag(earth_validation_table)) / sd(earth_validation_table)
  earth_validation_sum_diag <- sum(diag(earth_validation_table))
  earth_validation_prop <- diag(prop.table(earth_validation_table, margin = 1))
  earth_validation_no[i] <- as.numeric(earth_validation_prop[1])
  earth_validation_no_mean <- mean(earth_validation_no)
  earth_validation_yes[i] <- as.numeric(earth_validation_prop[2])
  earth_validation_yes_mean <- mean(earth_validation_yes)
  
  earth_no[i] <- mean(c(earth_test_no, earth_validation_no))
  earth_no_mean <- mean(earth_no)
  
  earth_yes[i] <- mean(c(earth_test_yes, earth_validation_yes))
  earth_yes_mean <- mean(earth_yes)
  
  earth_holdout[i] <- mean(c(earth_test_accuracy_mean, earth_validation_accuracy_mean))
  earth_holdout_mean <- mean(earth_holdout)
  earth_overfitting[i] <- earth_holdout_mean / earth_train_accuracy_mean
  earth_overfitting_mean <- mean(earth_overfitting)
  earth_overfitting_range <- range(earth_overfitting)
  
  earth_table <- earth_test_table + earth_validation_table
  
  earth_table_sum_diag <- sum(diag(earth_table))
  
  
  #### Flexible discriminant analysis ####
  fda_train_fit <- mda::fda(y_train ~ ., data = train)
  
  fda_train_pred <- predict(fda_train_fit, train01)
  fda_train_table <- table(fda_train_pred, y_train)
  fda_train_accuracy[i] <- sum(diag(fda_train_table)) / sum(fda_train_table)
  fda_train_accuracy_mean <- mean(fda_train_accuracy)
  fda_train_mean <- mean(diag(fda_train_table)) / mean(fda_train_table)
  fda_train_sd <- sd(diag(fda_train_table)) / sd(fda_train_table)
  fda_train_sum_diag <- sum(diag(fda_train_table))
  fda_train_prop <- diag(prop.table(fda_train_table, margin = 1))
  fda_train_no[i] <- as.numeric(fda_train_prop[1])
  fda_train_no_mean <- mean(fda_train_no)
  fda_train_yes[i] <- as.numeric(fda_train_prop[2])
  fda_train_yes_mean <- mean(fda_train_yes)
  
  fda_test_pred <- predict(fda_train_fit, test01)
  fda_test_table <- table(fda_test_pred, y_test)
  fda_test_accuracy[i] <- sum(diag(fda_test_table)) / sum(fda_test_table)
  fda_test_accuracy_mean <- mean(fda_test_accuracy)
  fda_test_mean <- mean(diag(fda_test_table)) / mean(fda_test_table)
  fda_test_sd <- sd(diag(fda_test_table)) / sd(fda_test_table)
  fda_test_sum_diag <- sum(diag(fda_test_table))
  fda_test_prop <- diag(prop.table(fda_test_table, margin = 1))
  fda_test_no[i] <- as.numeric(fda_test_prop[1])
  fda_test_no_mean <- mean(fda_test_no)
  fda_test_yes[i] <- as.numeric(fda_test_prop[2])
  fda_test_yes_mean <- mean(fda_test_yes)
  
  fda_validation_pred <- predict(fda_train_fit, validation01)
  fda_validation_table <- table(fda_validation_pred, y_validation)
  fda_validation_accuracy[i] <- sum(diag(fda_validation_table)) / sum(fda_validation_table)
  fda_validation_accuracy_mean <- mean(fda_validation_accuracy)
  fda_validation_mean <- mean(diag(fda_validation_table)) / mean(fda_validation_table)
  fda_validation_sd <- sd(diag(fda_validation_table)) / sd(fda_validation_table)
  fda_validation_sum_diag <- sum(diag(fda_validation_table))
  fda_validation_prop <- diag(prop.table(fda_validation_table, margin = 1))
  fda_validation_no[i] <- as.numeric(fda_validation_prop[1])
  fda_validation_no_mean <- mean(fda_validation_no)
  fda_validation_yes[i] <- as.numeric(fda_validation_prop[2])
  fda_validation_yes_mean <- mean(fda_validation_yes)
  
  fda_no[i] <- mean(c(fda_test_no, fda_validation_no))
  fda_no_mean <- mean(fda_no)
  
  fda_yes[i] <- mean(c(fda_test_yes, fda_validation_yes))
  fda_yes_mean <- mean(fda_yes)
  
  fda_holdout[i] <- mean(c(fda_test_accuracy_mean, fda_validation_accuracy_mean))
  fda_holdout_mean <- mean(fda_holdout)
  fda_overfitting[i] <- fda_holdout_mean / fda_train_accuracy_mean
  fda_overfitting_mean <- mean(fda_overfitting)
  fda_overfitting_range <- range(fda_overfitting)
  
  fda_table <- fda_test_table + fda_validation_table
  
  fda_table_sum_diag <- sum(diag(fda_table))
  
  
  #### Linear Discriminant Analysis ####
  lda_train_fit <- MASS::lda(y_train ~ ., data = train)
  
  lda_train_pred <- predict(lda_train_fit, train01)
  lda_train_table <- table(lda_train_pred$class, y_train)
  lda_train_accuracy[i] <- sum(diag(lda_train_table)) / sum(lda_train_table)
  lda_train_accuracy_mean <- mean(lda_train_accuracy)
  lda_train_mean <- mean(diag(lda_train_table)) / mean(lda_train_table)
  lda_train_sd <- sd(diag(lda_train_table)) / sd(lda_train_table)
  lda_train_sum_diag <- sum(diag(lda_train_table))
  lda_train_prop <- diag(prop.table(lda_train_table, margin = 1))
  lda_train_no[i] <- as.numeric(lda_train_prop[1])
  lda_train_no_mean <- mean(lda_train_no)
  lda_train_yes[i] <- as.numeric(lda_train_prop[2])
  lda_train_yes_mean <- mean(lda_train_yes)
  
  lda_test_pred <- predict(lda_train_fit, test01)
  lda_test_table <- table(lda_test_pred$class, y_test)
  lda_test_accuracy[i] <- sum(diag(lda_test_table)) / sum(lda_test_table)
  lda_test_accuracy_mean <- mean(lda_test_accuracy)
  lda_test_mean <- mean(diag(lda_test_table)) / mean(lda_test_table)
  lda_test_sd <- sd(diag(lda_test_table)) / sd(lda_test_table)
  lda_test_sum_diag <- sum(diag(lda_test_table))
  lda_test_prop <- diag(prop.table(lda_test_table, margin = 1))
  lda_test_no[i] <- as.numeric(lda_test_prop[1])
  lda_test_no_mean <- mean(lda_test_no)
  lda_test_yes[i] <- as.numeric(lda_test_prop[2])
  lda_test_yes_mean <- mean(lda_test_yes)
  
  lda_validation_pred <- predict(lda_train_fit, validation01)
  lda_validation_table <- table(lda_validation_pred$class, y_validation)
  lda_validation_accuracy[i] <- sum(diag(lda_validation_table)) / sum(lda_validation_table)
  lda_validation_accuracy_mean <- mean(lda_validation_accuracy)
  lda_validation_mean <- mean(diag(lda_validation_table)) / mean(lda_validation_table)
  lda_validation_sd <- sd(diag(lda_validation_table)) / sd(lda_validation_table)
  lda_validation_sum_diag <- sum(diag(lda_validation_table))
  lda_validation_prop <- diag(prop.table(lda_validation_table, margin = 1))
  lda_validation_no[i] <- as.numeric(lda_validation_prop[1])
  lda_validation_no_mean <- mean(lda_validation_no)
  lda_validation_yes[i] <- as.numeric(lda_validation_prop[2])
  lda_validation_yes_mean <- mean(lda_validation_yes)
  
  lda_no[i] <- mean(c(lda_test_no, lda_validation_no))
  lda_no_mean <- mean(lda_no)
  
  lda_yes[i] <- mean(c(lda_test_yes, lda_validation_yes))
  lda_yes_mean <- mean(lda_yes)
  
  lda_holdout[i] <- mean(c(lda_test_accuracy_mean, lda_validation_accuracy_mean))
  lda_holdout_mean <- mean(lda_holdout)
  lda_overfitting[i] <- lda_holdout_mean / lda_train_accuracy_mean
  lda_overfitting_mean <- mean(lda_overfitting)
  lda_overfitting_range <- range(lda_overfitting)
  
  lda_table <- lda_test_table + lda_validation_table
  
  lda_table_sum_diag <- sum(diag(lda_table))
  
  
  #### Linear ####
  linear_train_fit <- MachineShop::fit(type ~ ., data = train01, model = "LMModel")
  
  linear_train_pred <- predict(linear_train_fit, train01)
  linear_train_table <- table(linear_train_pred, y_train)
  linear_train_accuracy[i] <- sum(diag(linear_train_table)) / sum(linear_train_table)
  linear_train_accuracy_mean <- mean(linear_train_accuracy)
  linear_train_mean <- mean(diag(linear_train_table)) / mean(linear_train_table)
  linear_train_sd <- sd(diag(linear_train_table)) / sd(linear_train_table)
  linear_train_sum_diag <- sum(diag(linear_train_table))
  linear_train_prop <- diag(prop.table(linear_train_table, margin = 1))
  linear_train_no[i] <- as.numeric(linear_train_prop[1])
  linear_train_no_mean <- mean(linear_train_no)
  linear_train_yes[i] <- as.numeric(linear_train_prop[2])
  linear_train_yes_mean <- mean(linear_train_yes)
  
  linear_test_pred <- predict(linear_train_fit, test01)
  linear_test_table <- table(linear_test_pred, y_test)
  linear_test_accuracy[i] <- sum(diag(linear_test_table)) / sum(linear_test_table)
  linear_test_accuracy_mean <- mean(linear_test_accuracy)
  linear_test_mean <- mean(diag(linear_test_table)) / mean(linear_test_table)
  linear_test_sd <- sd(diag(linear_test_table)) / sd(linear_test_table)
  linear_test_sum_diag <- sum(diag(linear_test_table))
  linear_test_prop <- diag(prop.table(linear_test_table, margin = 1))
  linear_test_no[i] <- as.numeric(linear_test_prop[1])
  linear_test_no_mean <- mean(linear_test_no)
  linear_test_yes[i] <- as.numeric(linear_test_prop[2])
  linear_test_yes_mean <- mean(linear_test_yes)
  
  linear_validation_pred <- predict(linear_train_fit, validation01)
  linear_validation_table <- table(linear_validation_pred, y_validation)
  linear_validation_accuracy[i] <- sum(diag(linear_validation_table)) / sum(linear_validation_table)
  linear_validation_accuracy_mean <- mean(linear_validation_accuracy)
  linear_validation_mean <- mean(diag(linear_validation_table)) / mean(linear_validation_table)
  linear_validation_sd <- sd(diag(linear_validation_table)) / sd(linear_validation_table)
  linear_validation_sum_diag <- sum(diag(linear_validation_table))
  linear_validation_prop <- diag(prop.table(linear_validation_table, margin = 1))
  linear_validation_no[i] <- as.numeric(linear_validation_prop[1])
  linear_validation_no_mean <- mean(linear_validation_no)
  linear_validation_yes[i] <- as.numeric(linear_validation_prop[2])
  linear_validation_yes_mean <- mean(linear_validation_yes)
  
  linear_no[i] <- mean(c(linear_test_no, linear_validation_no))
  linear_no_mean <- mean(linear_no)
  
  linear_yes[i] <- mean(c(linear_test_yes, linear_validation_yes))
  linear_yes_mean <- mean(linear_yes)
  
  linear_holdout[i] <- mean(c(linear_test_accuracy_mean, linear_validation_accuracy_mean))
  linear_holdout_mean <- mean(linear_holdout)
  linear_overfitting[i] <- linear_holdout_mean / linear_train_accuracy_mean
  linear_overfitting_mean <- mean(linear_overfitting)
  linear_overfitting_range <- range(linear_overfitting)
  
  linear_table <- linear_test_table + linear_validation_table
  
  linear_table_sum_diag <- sum(diag(linear_table))
  
  
  #### Mixed discriminant analysis ####
  mda_train_fit <- mda::mda(formula = type ~ ., data = test01)
  
  mda_train_pred <- predict(mda_train_fit, train01)
  mda_train_table <- table(mda_train_pred, y_train)
  mda_train_accuracy[i] <- sum(diag(mda_train_table)) / sum(mda_train_table)
  mda_train_accuracy_mean <- mean(mda_train_accuracy)
  mda_train_mean <- mean(diag(mda_train_table)) / mean(mda_train_table)
  mda_train_sd <- sd(diag(mda_train_table)) / sd(mda_train_table)
  mda_train_sum_diag <- sum(diag(mda_train_table))
  mda_train_prop <- diag(prop.table(mda_train_table, margin = 1))
  mda_train_no[i] <- as.numeric(mda_train_prop[1])
  mda_train_no_mean <- mean(mda_train_no)
  mda_train_yes[i] <- as.numeric(mda_train_prop[2])
  mda_train_yes_mean <- mean(mda_train_yes)
  
  mda_test_pred <- predict(mda_train_fit, test01)
  mda_test_table <- table(mda_test_pred, y_test)
  mda_test_accuracy[i] <- sum(diag(mda_test_table)) / sum(mda_test_table)
  mda_test_accuracy_mean <- mean(mda_test_accuracy)
  mda_test_mean <- mean(diag(mda_test_table)) / mean(mda_test_table)
  mda_test_sd <- sd(diag(mda_test_table)) / sd(mda_test_table)
  mda_test_sum_diag <- sum(diag(mda_test_table))
  mda_test_prop <- diag(prop.table(mda_test_table, margin = 1))
  mda_test_no[i] <- as.numeric(mda_test_prop[1])
  mda_test_no_mean <- mean(mda_test_no)
  mda_test_yes[i] <- as.numeric(mda_test_prop[2])
  mda_test_yes_mean <- mean(mda_test_yes)
  
  mda_validation_pred <- predict(mda_train_fit, validation01)
  mda_validation_table <- table(mda_validation_pred, y_validation)
  mda_validation_accuracy[i] <- sum(diag(mda_validation_table)) / sum(mda_validation_table)
  mda_validation_accuracy_mean <- mean(mda_validation_accuracy)
  mda_validation_mean <- mean(diag(mda_validation_table)) / mean(mda_validation_table)
  mda_validation_sd <- sd(diag(mda_validation_table)) / sd(mda_validation_table)
  mda_validation_sum_diag <- sum(diag(mda_validation_table))
  mda_validation_prop <- diag(prop.table(mda_validation_table, margin = 1))
  mda_validation_no[i] <- as.numeric(mda_validation_prop[1])
  mda_validation_no_mean <- mean(mda_validation_no)
  mda_validation_yes[i] <- as.numeric(mda_validation_prop[2])
  mda_validation_yes_mean <- mean(mda_validation_yes)
  
  mda_no[i] <- mean(c(mda_test_no, mda_validation_no))
  mda_no_mean <- mean(mda_no)
  
  mda_yes[i] <- mean(c(mda_test_yes, mda_validation_yes))
  mda_yes_mean <- mean(mda_yes)
  
  mda_holdout[i] <- mean(c(mda_test_accuracy_mean, mda_validation_accuracy_mean))
  mda_holdout_mean <- mean(mda_holdout)
  mda_overfitting[i] <- mda_holdout_mean / mda_train_accuracy_mean
  mda_overfitting_mean <- mean(mda_overfitting)
  mda_overfitting_range <- range(mda_overfitting)
  
  mda_table <- mda_test_table + mda_validation_table
  
  mda_table_sum_diag <- sum(diag(mda_table))
  
  
  #### Naive Bayes ####
  n_bayes_train_fit <- naiveBayes(y_train ~ ., data = train)
  
  n_bayes_train_pred <- predict(n_bayes_train_fit, train01)
  n_bayes_train_table <- table(n_bayes_train_pred, y_train)
  n_bayes_train_accuracy[i] <- sum(diag(n_bayes_train_table)) / sum(n_bayes_train_table)
  n_bayes_train_accuracy_mean <- mean(n_bayes_train_accuracy)
  n_bayes_train_mean <- mean(diag(n_bayes_train_table)) / mean(n_bayes_train_table)
  n_bayes_train_sd <- sd(diag(n_bayes_train_table)) / sd(n_bayes_train_table)
  n_bayes_train_sum_diag <- sum(diag(n_bayes_train_table))
  n_bayes_train_prop <- diag(prop.table(n_bayes_train_table, margin = 1))
  n_bayes_train_no[i] <- as.numeric(n_bayes_train_prop[1])
  n_bayes_train_no_mean <- mean(n_bayes_train_no)
  n_bayes_train_yes[i] <- as.numeric(n_bayes_train_prop[2])
  n_bayes_train_yes_mean <- mean(n_bayes_train_yes)
  
  n_bayes_test_pred <- predict(n_bayes_train_fit, test01)
  n_bayes_test_table <- table(n_bayes_test_pred, y_test)
  n_bayes_test_accuracy[i] <- sum(diag(n_bayes_test_table)) / sum(n_bayes_test_table)
  n_bayes_test_accuracy_mean <- mean(n_bayes_test_accuracy)
  n_bayes_test_mean <- mean(diag(n_bayes_test_table)) / mean(n_bayes_test_table)
  n_bayes_test_sd <- sd(diag(n_bayes_test_table)) / sd(n_bayes_test_table)
  n_bayes_test_sum_diag <- sum(diag(n_bayes_test_table))
  n_bayes_test_prop <- diag(prop.table(n_bayes_test_table, margin = 1))
  n_bayes_test_no[i] <- as.numeric(n_bayes_test_prop[1])
  n_bayes_test_no_mean <- mean(n_bayes_test_no)
  n_bayes_test_yes[i] <- as.numeric(n_bayes_test_prop[2])
  n_bayes_test_yes_mean <- mean(n_bayes_test_yes)
  
  n_bayes_validation_pred <- predict(n_bayes_train_fit, validation01)
  n_bayes_validation_table <- table(n_bayes_validation_pred, y_validation)
  n_bayes_validation_accuracy[i] <- sum(diag(n_bayes_validation_table)) / sum(n_bayes_validation_table)
  n_bayes_validation_accuracy_mean <- mean(n_bayes_validation_accuracy)
  n_bayes_validation_mean <- mean(diag(n_bayes_validation_table)) / mean(n_bayes_validation_table)
  n_bayes_validation_sd <- sd(diag(n_bayes_validation_table)) / sd(n_bayes_validation_table)
  n_bayes_validation_sum_diag <- sum(diag(n_bayes_validation_table))
  n_bayes_validation_prop <- diag(prop.table(n_bayes_validation_table, margin = 1))
  n_bayes_validation_no[i] <- as.numeric(n_bayes_validation_prop[1])
  n_bayes_validation_no_mean <- mean(n_bayes_validation_no)
  n_bayes_validation_yes[i] <- as.numeric(n_bayes_validation_prop[2])
  n_bayes_validation_yes_mean <- mean(n_bayes_validation_yes)
  
  n_bayes_no[i] <- mean(c(n_bayes_test_no, n_bayes_validation_no))
  n_bayes_no_mean <- mean(n_bayes_no)
  
  n_bayes_yes[i] <- mean(c(n_bayes_test_yes, n_bayes_validation_yes))
  n_bayes_yes_mean <- mean(n_bayes_yes)
  
  n_bayes_holdout[i] <- mean(c(n_bayes_test_accuracy_mean, n_bayes_validation_accuracy_mean))
  n_bayes_holdout_mean <- mean(n_bayes_holdout)
  n_bayes_overfitting[i] <- n_bayes_holdout_mean / n_bayes_train_accuracy_mean
  n_bayes_overfitting_mean <- mean(n_bayes_overfitting)
  n_bayes_overfitting_range <- range(n_bayes_overfitting)
  
  n_bayes_table <- n_bayes_test_table + n_bayes_validation_table
  
  n_bayes_table_sum_diag <- sum(diag(n_bayes_table))
  
  
  #### Partial Least Squares ####
  pls_train_fit <- MachineShop::fit(type ~ ., data = train01, model = "PLSModel")
  
  pls_train_pred <- predict(pls_train_fit, train01)
  pls_train_table <- table(pls_train_pred, y_train)
  pls_train_accuracy[i] <- sum(diag(pls_train_table)) / sum(pls_train_table)
  pls_train_accuracy_mean <- mean(pls_train_accuracy)
  pls_train_mean <- mean(diag(pls_train_table)) / mean(pls_train_table)
  pls_train_sd <- sd(diag(pls_train_table)) / sd(pls_train_table)
  pls_train_sum_diag <- sum(diag(pls_train_table))
  pls_train_prop <- diag(prop.table(pls_train_table, margin = 1))
  pls_train_no[i] <- as.numeric(pls_train_prop[1])
  pls_train_no_mean <- mean(pls_train_no)
  pls_train_yes[i] <- as.numeric(pls_train_prop[2])
  pls_train_yes_mean <- mean(pls_train_yes)
  
  pls_test_pred <- predict(pls_train_fit, test01)
  pls_test_table <- table(pls_test_pred, y_test)
  pls_test_accuracy[i] <- sum(diag(pls_test_table)) / sum(pls_test_table)
  pls_test_accuracy_mean <- mean(pls_test_accuracy)
  pls_test_mean <- mean(diag(pls_test_table)) / mean(pls_test_table)
  pls_test_sd <- sd(diag(pls_test_table)) / sd(pls_test_table)
  pls_test_sum_diag <- sum(diag(pls_test_table))
  pls_test_prop <- diag(prop.table(pls_test_table, margin = 1))
  pls_test_no[i] <- as.numeric(pls_test_prop[1])
  pls_test_no_mean <- mean(pls_test_no)
  pls_test_yes[i] <- as.numeric(pls_test_prop[2])
  pls_test_yes_mean <- mean(pls_test_yes)
  
  pls_validation_pred <- predict(pls_train_fit, validation01)
  pls_validation_table <- table(pls_validation_pred, y_validation)
  pls_validation_accuracy[i] <- sum(diag(pls_validation_table)) / sum(pls_validation_table)
  pls_validation_accuracy_mean <- mean(pls_validation_accuracy)
  pls_validation_mean <- mean(diag(pls_validation_table)) / mean(pls_validation_table)
  pls_validation_sd <- sd(diag(pls_validation_table)) / sd(pls_validation_table)
  pls_validation_sum_diag <- sum(diag(pls_validation_table))
  pls_validation_prop <- diag(prop.table(pls_validation_table, margin = 1))
  pls_validation_no[i] <- as.numeric(pls_validation_prop[1])
  pls_validation_no_mean <- mean(pls_validation_no)
  pls_validation_yes[i] <- as.numeric(pls_validation_prop[2])
  pls_validation_yes_mean <- mean(pls_validation_yes)
  
  pls_no[i] <- mean(c(pls_test_no, pls_validation_no))
  pls_no_mean <- mean(pls_no)
  
  pls_yes[i] <- mean(c(pls_test_yes, pls_validation_yes))
  pls_yes_mean <- mean(pls_yes)
  
  pls_holdout[i] <- mean(c(pls_test_accuracy_mean, pls_validation_accuracy_mean))
  pls_holdout_mean <- mean(pls_holdout)
  pls_overfitting[i] <- pls_holdout_mean / pls_train_accuracy_mean
  pls_overfitting_mean <- mean(pls_overfitting)
  pls_overfitting_range <- range(pls_overfitting)
  
  pls_table <- pls_test_table + pls_validation_table
  
  pls_table_sum_diag <- sum(diag(pls_table))
  
  
  #### Penalized Discriminant Analysis ####
  pda_train_fit <- MachineShop::fit(type ~ ., data = train01, model = "PDAModel")
  
  pda_train_pred <- predict(pda_train_fit, train01)
  pda_train_table <- table(pda_train_pred, y_train)
  pda_train_accuracy[i] <- sum(diag(pda_train_table)) / sum(pda_train_table)
  pda_train_accuracy_mean <- mean(pda_train_accuracy)
  pda_train_mean <- mean(diag(pda_train_table)) / mean(pda_train_table)
  pda_train_sd <- sd(diag(pda_train_table)) / sd(pda_train_table)
  pda_train_sum_diag <- sum(diag(pda_train_table))
  pda_train_prop <- diag(prop.table(pda_train_table, margin = 1))
  pda_train_no[i] <- as.numeric(pda_train_prop[1])
  pda_train_no_mean <- mean(pda_train_no)
  pda_train_yes[i] <- as.numeric(pda_train_prop[2])
  pda_train_yes_mean <- mean(pda_train_yes)
  
  pda_test_pred <- predict(pda_train_fit, test01)
  pda_test_table <- table(pda_test_pred, y_test)
  pda_test_accuracy[i] <- sum(diag(pda_test_table)) / sum(pda_test_table)
  pda_test_accuracy_mean <- mean(pda_test_accuracy)
  pda_test_mean <- mean(diag(pda_test_table)) / mean(pda_test_table)
  pda_test_sd <- sd(diag(pda_test_table)) / sd(pda_test_table)
  pda_test_sum_diag <- sum(diag(pda_test_table))
  pda_test_prop <- diag(prop.table(pda_test_table, margin = 1))
  pda_test_no[i] <- as.numeric(pda_test_prop[1])
  pda_test_no_mean <- mean(pda_test_no)
  pda_test_yes[i] <- as.numeric(pda_test_prop[2])
  pda_test_yes_mean <- mean(pda_test_yes)
  
  pda_validation_pred <- predict(pda_train_fit, validation01)
  pda_validation_table <- table(pda_validation_pred, y_validation)
  pda_validation_accuracy[i] <- sum(diag(pda_validation_table)) / sum(pda_validation_table)
  pda_validation_accuracy_mean <- mean(pda_validation_accuracy)
  pda_validation_mean <- mean(diag(pda_validation_table)) / mean(pda_validation_table)
  pda_validation_sd <- sd(diag(pda_validation_table)) / sd(pda_validation_table)
  pda_validation_sum_diag <- sum(diag(pda_validation_table))
  pda_validation_prop <- diag(prop.table(pda_validation_table, margin = 1))
  pda_validation_no[i] <- as.numeric(pda_validation_prop[1])
  pda_validation_no_mean <- mean(pda_validation_no)
  pda_validation_yes[i] <- as.numeric(pda_validation_prop[2])
  pda_validation_yes_mean <- mean(pda_validation_yes)
  
  pda_no[i] <- mean(c(pda_test_no, pda_validation_no))
  pda_no_mean <- mean(pda_no)
  
  pda_yes[i] <- mean(c(pda_test_yes, pda_validation_yes))
  pda_yes_mean <- mean(pda_yes)
  
  pda_holdout[i] <- mean(c(pda_test_accuracy_mean, pda_validation_accuracy_mean))
  pda_holdout_mean <- mean(pda_holdout)
  pda_overfitting[i] <- pda_holdout_mean / pda_train_accuracy_mean
  pda_overfitting_mean <- mean(pda_overfitting)
  pda_overfitting_range <- range(pda_overfitting)
  
  pda_table <- pda_test_table + pda_validation_table
  
  pda_table_sum_diag <- sum(diag(pda_table))
  
  
  #### Quadratic Discriminant Analysis ####
  qda_train_fit <- MASS::qda(type ~ ., data = train01)
  
  qda_train_pred <- predict(qda_train_fit, train01)
  qda_train_table <- table(qda_train_pred$class, y_train)
  qda_train_accuracy[i] <- sum(diag(qda_train_table)) / sum(qda_train_table)
  qda_train_accuracy_mean <- mean(qda_train_accuracy)
  qda_train_mean <- mean(diag(qda_train_table)) / mean(qda_train_table)
  qda_train_sd <- sd(diag(qda_train_table)) / sd(qda_train_table)
  qda_train_sum_diag <- sum(diag(qda_train_table))
  qda_train_prop <- diag(prop.table(qda_train_table, margin = 1))
  qda_train_no[i] <- as.numeric(qda_train_prop[1])
  qda_train_no_mean <- mean(qda_train_no)
  qda_train_yes[i] <- as.numeric(qda_train_prop[2])
  qda_train_yes_mean <- mean(qda_train_yes)
  
  qda_test_pred <- predict(qda_train_fit, test01)
  qda_test_table <- table(qda_test_pred$class, y_test)
  qda_test_accuracy[i] <- sum(diag(qda_test_table)) / sum(qda_test_table)
  qda_test_accuracy_mean <- mean(qda_test_accuracy)
  qda_test_mean <- mean(diag(qda_test_table)) / mean(qda_test_table)
  qda_test_sd <- sd(diag(qda_test_table)) / sd(qda_test_table)
  qda_test_sum_diag <- sum(diag(qda_test_table))
  qda_test_prop <- diag(prop.table(qda_test_table, margin = 1))
  qda_test_no[i] <- as.numeric(qda_test_prop[1])
  qda_test_no_mean <- mean(qda_test_no)
  qda_test_yes[i] <- as.numeric(qda_test_prop[2])
  qda_test_yes_mean <- mean(qda_test_yes)
  
  qda_validation_pred <- predict(qda_train_fit, validation01)
  qda_validation_table <- table(qda_validation_pred$class, y_validation)
  qda_validation_accuracy[i] <- sum(diag(qda_validation_table)) / sum(qda_validation_table)
  qda_validation_accuracy_mean <- mean(qda_validation_accuracy)
  qda_validation_mean <- mean(diag(qda_validation_table)) / mean(qda_validation_table)
  qda_validation_sd <- sd(diag(qda_validation_table)) / sd(qda_validation_table)
  qda_validation_sum_diag <- sum(diag(qda_validation_table))
  qda_validation_prop <- diag(prop.table(qda_validation_table, margin = 1))
  qda_validation_no[i] <- as.numeric(qda_validation_prop[1])
  qda_validation_no_mean <- mean(qda_validation_no)
  qda_validation_yes[i] <- as.numeric(qda_validation_prop[2])
  qda_validation_yes_mean <- mean(qda_validation_yes)
  
  qda_no[i] <- mean(c(qda_test_no, qda_validation_no))
  qda_no_mean <- mean(qda_no)
  
  qda_yes[i] <- mean(c(qda_test_yes, qda_validation_yes))
  qda_yes_mean <- mean(qda_yes)
  
  qda_holdout[i] <- mean(c(qda_test_accuracy, qda_validation_accuracy_mean))
  qda_holdout_mean <- mean(qda_holdout)
  qda_overfitting[i] <- qda_holdout_mean / qda_train_accuracy_mean
  qda_overfitting_mean <- mean(qda_overfitting)
  qda_overfitting_range <- range(qda_overfitting)
  
  qda_table <- qda_test_table + qda_validation_table
  
  qda_table_sum_diag <- sum(diag(qda_table))
  
  
  #### Random Forest ####
  tune_train_rf <- tune.randomForest(x = train, y = y_train, data = df, mtry = 1:(ncol(train)-1))
  
  rf_train_pred <- predict(tune_train_rf$best.model, train01)
  rf_train_table <- table(rf_train_pred, y_train)
  rf_train_accuracy[i] <- sum(diag(rf_train_table)) / sum(rf_train_table)
  rf_train_accuracy_mean <- mean(rf_train_accuracy)
  rf_train_mean <- mean(diag(rf_train_table)) / mean(rf_train_table)
  rf_train_sd <- sd(diag(rf_train_table)) / sd(rf_train_table)
  rf_train_sum_diag <- sum(diag(rf_train_table))
  rf_train_prop <- diag(prop.table(rf_train_table, margin = 1))
  rf_train_no[i] <- as.numeric(rf_train_prop[1])
  rf_train_no_mean <- mean(rf_train_no)
  rf_train_yes[i] <- as.numeric(rf_train_prop[2])
  rf_train_yes_mean <- mean(rf_train_yes)
  
  rf_test_pred <- predict(tune_train_rf$best.model, test01)
  rf_test_table <- table(rf_test_pred, y_test)
  rf_test_accuracy[i] <- sum(diag(rf_test_table)) / sum(rf_test_table)
  rf_test_accuracy_mean <- mean(rf_test_accuracy)
  rf_test_mean <- mean(diag(rf_test_table)) / mean(rf_test_table)
  rf_test_sd <- sd(diag(rf_test_table)) / sd(rf_test_table)
  rf_test_sum_diag <- sum(diag(rf_test_table))
  rf_test_prop <- diag(prop.table(rf_test_table, margin = 1))
  rf_test_no[i] <- as.numeric(rf_test_prop[1])
  rf_test_no_mean <- mean(rf_test_no)
  rf_test_yes[i] <- as.numeric(rf_test_prop[2])
  rf_test_yes_mean <- mean(rf_test_yes)
  
  rf_validation_pred <- predict(tune_train_rf$best.model, validation01)
  rf_validation_table <- table(rf_validation_pred, y_validation)
  rf_validation_accuracy[i] <- sum(diag(rf_validation_table)) / sum(rf_validation_table)
  rf_validation_accuracy_mean <- mean(rf_validation_accuracy)
  rf_validation_mean <- mean(diag(rf_validation_table)) / mean(rf_validation_table)
  rf_validation_sd <- sd(diag(rf_validation_table)) / sd(rf_validation_table)
  rf_validation_sum_diag <- sum(diag(rf_validation_table))
  rf_validation_prop <- diag(prop.table(rf_validation_table, margin = 1))
  rf_validation_no[i] <- as.numeric(rf_validation_prop[1])
  rf_validation_no_mean <- mean(rf_validation_no)
  rf_validation_yes[i] <- as.numeric(rf_validation_prop[2])
  rf_validation_yes_mean <- mean(rf_validation_yes)
  
  rf_no[i] <- mean(c(rf_test_no, rf_validation_no))
  rf_no_mean <- mean(rf_no)
  
  rf_yes[i] <- mean(c(rf_test_yes, rf_validation_yes))
  rf_yes_mean <- mean(rf_yes)
  
  rf_holdout[i] <- mean(c(rf_test_accuracy_mean, rf_validation_accuracy_mean))
  rf_holdout_mean <- mean(rf_holdout)
  rf_overfitting[i] <- rf_holdout_mean / rf_train_accuracy_mean
  rf_overfitting_mean <- mean(rf_overfitting)
  rf_overfitting_range <- range(rf_overfitting)
  
  rf_table <- rf_test_table + rf_validation_table
  
  rf_table_sum_diag <- sum(diag(rf_table))
  
  
  #### Ranger ####
  ranger_train_fit <- MachineShop::fit(type ~ ., data = train01, model = "RangerModel")
  
  ranger_train_pred <- predict(ranger_train_fit, train01)
  ranger_train_table <- table(ranger_train_pred, y_train)
  ranger_train_accuracy[i] <- sum(diag(ranger_train_table)) / sum(ranger_train_table)
  ranger_train_accuracy_mean <- mean(ranger_train_accuracy)
  ranger_train_mean <- mean(diag(ranger_train_table)) / mean(ranger_train_table)
  ranger_train_sd <- sd(diag(ranger_train_table)) / sd(ranger_train_table)
  ranger_train_sum_diag <- sum(diag(ranger_train_table))
  ranger_train_prop <- diag(prop.table(ranger_train_table, margin = 1))
  ranger_train_no[i] <- as.numeric(ranger_train_prop[1])
  ranger_train_no_mean <- mean(ranger_train_no)
  ranger_train_yes[i] <- as.numeric(ranger_train_prop[2])
  ranger_train_yes_mean <- mean(ranger_train_yes)
  
  ranger_test_pred <- predict(ranger_train_fit, test01)
  ranger_test_table <- table(ranger_test_pred, y_test)
  ranger_test_accuracy[i] <- sum(diag(ranger_test_table)) / sum(ranger_test_table)
  ranger_test_accuracy_mean <- mean(ranger_test_accuracy)
  ranger_test_mean <- mean(diag(ranger_test_table)) / mean(ranger_test_table)
  ranger_test_sd <- sd(diag(ranger_test_table)) / sd(ranger_test_table)
  ranger_test_sum_diag <- sum(diag(ranger_test_table))
  ranger_test_prop <- diag(prop.table(ranger_test_table, margin = 1))
  ranger_test_no[i] <- as.numeric(ranger_test_prop[1])
  ranger_test_no_mean <- mean(ranger_test_no)
  ranger_test_yes[i] <- as.numeric(ranger_test_prop[2])
  ranger_test_yes_mean <- mean(ranger_test_yes)
  
  ranger_validation_pred <- predict(ranger_train_fit, validation01)
  ranger_validation_table <- table(ranger_validation_pred, y_validation)
  ranger_validation_accuracy[i] <- sum(diag(ranger_validation_table)) / sum(ranger_validation_table)
  ranger_validation_accuracy_mean <- mean(ranger_validation_accuracy)
  ranger_validation_mean <- mean(diag(ranger_validation_table)) / mean(ranger_validation_table)
  ranger_validation_sd <- sd(diag(ranger_validation_table)) / sd(ranger_validation_table)
  ranger_validation_sum_diag <- sum(diag(ranger_validation_table))
  ranger_validation_prop <- diag(prop.table(ranger_validation_table, margin = 1))
  ranger_validation_no[i] <- as.numeric(ranger_validation_prop[1])
  ranger_validation_no_mean <- mean(ranger_validation_no)
  ranger_validation_yes[i] <- as.numeric(ranger_validation_prop[2])
  ranger_validation_yes_mean <- mean(ranger_validation_yes)
  
  ranger_no[i] <- mean(c(ranger_test_no, ranger_validation_no))
  ranger_no_mean <- mean(ranger_no)
  
  ranger_yes[i] <- mean(c(ranger_test_yes, ranger_validation_yes))
  ranger_yes_mean <- mean(ranger_yes)
  
  ranger_holdout[i] <- mean(c(ranger_test_accuracy_mean, ranger_validation_accuracy_mean))
  ranger_holdout_mean <- mean(ranger_holdout)
  ranger_overfitting[i] <- ranger_holdout_mean / ranger_train_accuracy_mean
  ranger_overfitting_mean <- mean(ranger_overfitting)
  ranger_overfitting_range <- range(ranger_overfitting)
  
  ranger_table <- ranger_test_table + ranger_validation_table
  
  ranger_table_sum_diag <- sum(diag(ranger_table))
  
  
  #### Regularized Discrmininat Analysis ####
  rda_train_fit <- klaR::rda(y_train ~ ., data = train)
  
  rda_train_pred <- predict(rda_train_fit, train01)
  rda_train_table <- table(rda_train_pred$class, y_train)
  rda_train_accuracy[i] <- sum(diag(rda_train_table)) / sum(rda_train_table)
  rda_train_accuracy_mean <- mean(rda_train_accuracy)
  rda_train_mean <- mean(diag(rda_train_table)) / mean(rda_train_table)
  rda_train_sd <- sd(diag(rda_train_table)) / sd(rda_train_table)
  rda_train_sum_diag <- sum(diag(rda_train_table))
  rda_train_prop <- diag(prop.table(rda_train_table, margin = 1))
  rda_train_no[i] <- as.numeric(rda_train_prop[1])
  rda_train_no_mean <- mean(rda_train_no)
  rda_train_yes[i] <- as.numeric(rda_train_prop[2])
  
  rda_test_pred <- predict(rda_train_fit, test01)
  rda_test_table <- table(rda_test_pred$class, y_test)
  rda_test_accuracy[i] <- sum(diag(rda_test_table)) / sum(rda_test_table)
  rda_test_accuracy_mean <- mean(rda_test_accuracy)
  rda_test_mean <- mean(diag(rda_test_table)) / mean(rda_test_table)
  rda_test_sd <- sd(diag(rda_test_table)) / sd(rda_test_table)
  rda_test_sum_diag <- sum(diag(rda_test_table))
  rda_test_prop <- diag(prop.table(rda_test_table, margin = 1))
  rda_test_no[i] <- as.numeric(rda_test_prop[1])
  rda_test_no_mean <- mean(rda_test_no)
  rda_test_yes[i] <- as.numeric(rda_test_prop[2])
  rda_test_yes_mean <- mean(rda_test_yes)
  
  rda_validation_pred <- predict(rda_train_fit, validation01)
  rda_validation_table <- table(rda_validation_pred$class, y_validation)
  rda_validation_accuracy[i] <- sum(diag(rda_validation_table)) / sum(rda_validation_table)
  rda_validation_accuracy_mean <- mean(rda_validation_accuracy)
  rda_validation_mean <- mean(diag(rda_validation_table)) / mean(rda_validation_table)
  rda_validation_sd <- sd(diag(rda_validation_table)) / sd(rda_validation_table)
  rda_validation_sum_diag <- sum(diag(rda_validation_table))
  rda_validation_prop <- diag(prop.table(rda_validation_table, margin = 1))
  rda_validation_no[i] <- as.numeric(rda_validation_prop[1])
  rda_validation_no_mean <- mean(rda_validation_no)
  rda_validation_yes[i] <- as.numeric(rda_validation_prop[2])
  rda_validation_yes_mean <- mean( rda_validation_yes)
  
  rda_no[i] <- mean(c(rda_test_no, rda_validation_no))
  rda_no_mean <- mean(rda_no)
  
  rda_yes[i] <- mean(c(rda_test_yes, rda_validation_yes))
  rda_yes_mean <- mean(rda_yes)
  
  rda_holdout[i] <- mean(c(rda_test_accuracy_mean, rda_validation_accuracy_mean))
  rda_holdout_mean <- mean(rda_holdout)
  rda_overfitting[i] <- rda_holdout_mean/rda_train_accuracy_mean
  rda_overfitting_mean <- mean(rda_overfitting)
  rda_overfitting_range <- range(rda_overfitting)
  
  rda_table <- rda_test_table + rda_validation_table
  
  rda_table_sum_diag <- sum(diag(rda_table))

  
  #### Rpart ####
  rda_train_fit <- klaR::rda(y_train ~ ., data = train)
  
  rda_train_pred <- predict(rda_train_fit, train01)
  rda_train_table <- table(rda_train_pred$class, y_train)
  rda_train_accuracy[i] <- sum(diag(rda_train_table)) / sum(rda_train_table)
  rda_train_accuracy_mean <- mean(rda_train_accuracy)
  rda_train_mean <- mean(diag(rda_train_table)) / mean(rda_train_table)
  rda_train_sd <- sd(diag(rda_train_table)) / sd(rda_train_table)
  rda_train_sum_diag <- sum(diag(rda_train_table))
  rda_train_prop <- diag(prop.table(rda_train_table, margin = 1))
  rda_train_no[i] <- as.numeric(rda_train_prop[1])
  rda_train_no_mean <- mean(rda_train_no)
  rda_train_yes[i] <- as.numeric(rda_train_prop[2])
  
  rda_test_pred <- predict(rda_train_fit, test01)
  rda_test_table <- table(rda_test_pred$class, y_test)
  rda_test_accuracy[i] <- sum(diag(rda_test_table)) / sum(rda_test_table)
  rda_test_accuracy_mean <- mean(rda_test_accuracy)
  rda_test_mean <- mean(diag(rda_test_table)) / mean(rda_test_table)
  rda_test_sd <- sd(diag(rda_test_table)) / sd(rda_test_table)
  rda_test_sum_diag <- sum(diag(rda_test_table))
  rda_test_prop <- diag(prop.table(rda_test_table, margin = 1))
  rda_test_no[i] <- as.numeric(rda_test_prop[1])
  rda_test_no_mean <- mean(rda_test_no)
  rda_test_yes[i] <- as.numeric(rda_test_prop[2])
  rda_test_yes_mean <- mean(rda_test_yes)
  
  rda_validation_pred <- predict(rda_train_fit, validation01)
  rda_validation_table <- table(rda_validation_pred$class, y_validation)
  rda_validation_accuracy[i] <- sum(diag(rda_validation_table)) / sum(rda_validation_table)
  rda_validation_accuracy_mean <- mean(rda_validation_accuracy)
  rda_validation_mean <- mean(diag(rda_validation_table)) / mean(rda_validation_table)
  rda_validation_sd <- sd(diag(rda_validation_table)) / sd(rda_validation_table)
  rda_validation_sum_diag <- sum(diag(rda_validation_table))
  rda_validation_prop <- diag(prop.table(rda_validation_table, margin = 1))
  rda_validation_no[i] <- as.numeric(rda_validation_prop[1])
  rda_validation_no_mean <- mean(rda_validation_no)
  rda_validation_yes[i] <- as.numeric(rda_validation_prop[2])
  rda_validation_yes_mean <- mean( rda_validation_yes)
  
  rda_no[i] <- mean(c(rda_test_no, rda_validation_no))
  rda_no_mean <- mean(rda_no)
  
  rda_yes[i] <- mean(c(rda_test_yes, rda_validation_yes))
  rda_yes_mean <- mean(rda_yes)
  
  rda_holdout[i] <- mean(c(rda_test_accuracy_mean, rda_validation_accuracy_mean))
  rda_holdout_mean <- mean(rda_holdout)
  rda_overfitting[i] <- rda_holdout_mean/rda_train_accuracy_mean
  rda_overfitting_mean <- mean(rda_overfitting)
  rda_overfitting_range <- range(rda_overfitting)
  
  rda_table <- rda_test_table + rda_validation_table
  
  rda_table_sum_diag <- sum(diag(rda_table))

  
  #### Support Vector Machines (SVM) ####
  svm_train_fit <- svm(y_train ~ ., data = train, kernel = "radial", gamma = 1, cost = 1)
  
  svm_train_pred <- predict(svm_train_fit, train01)
  svm_train_table <- table(svm_train_pred, y_train)
  svm_train_accuracy[i] <- sum(diag(svm_train_table)) / sum(svm_train_table)
  svm_train_accuracy_mean <- mean(svm_train_accuracy)
  svm_train_mean <- mean(diag(svm_train_table)) / mean(svm_train_table)
  svm_train_sd <- sd(diag(svm_train_table)) / sd(svm_train_table)
  svm_train_sum_diag <- sum(diag(svm_train_table))
  svm_train_prop <- diag(prop.table(svm_train_table, margin = 1))
  svm_train_no[i] <- as.numeric(svm_train_prop[1])
  svm_train_no_mean <- mean(svm_train_no)
  svm_train_yes[i] <- as.numeric(svm_train_prop[2])
  svm_train_yes_mean <- mean(svm_train_yes)
  
  svm_test_pred <- predict(svm_train_fit, test01)
  svm_test_table <- table(svm_test_pred, y_test)
  svm_test_accuracy[i] <- sum(diag(svm_test_table)) / sum(svm_test_table)
  svm_test_accuracy_mean <- mean(svm_test_accuracy)
  svm_test_mean <- mean(diag(svm_test_table)) / mean(svm_test_table)
  svm_test_sd <- sd(diag(svm_test_table)) / sd(svm_test_table)
  svm_test_sum_diag <- sum(diag(svm_test_table))
  svm_test_prop <- diag(prop.table(svm_test_table, margin = 1))
  svm_test_no[i] <- as.numeric(svm_test_prop[1])
  svm_test_no_mean <- mean(svm_test_no)
  svm_test_yes[i] <- as.numeric(svm_test_prop[2])
  svm_test_yes_mean <- mean(svm_test_yes)
  
  svm_validation_pred <- predict(svm_train_fit, validation01)
  svm_validation_table <- table(svm_validation_pred, y_validation)
  svm_validation_accuracy[i] <- sum(diag(svm_validation_table)) / sum(svm_validation_table)
  svm_validation_accuracy_mean <- mean(svm_validation_accuracy)
  svm_validation_mean <- mean(diag(svm_validation_table)) / mean(svm_validation_table)
  svm_validation_sd <- sd(diag(svm_validation_table)) / sd(svm_validation_table)
  svm_validation_sum_diag <- sum(diag(svm_validation_table))
  svm_validation_prop <- diag(prop.table(svm_validation_table, margin = 1))
  svm_validation_no[i] <- as.numeric(svm_validation_prop[1])
  svm_validation_no_mean <- mean(svm_validation_no)
  svm_validation_yes[i] <- as.numeric(svm_validation_prop[2])
  svm_validation_yes_mean <- mean(svm_validation_yes)
  
  svm_no[i] <- mean(c(svm_test_no, svm_validation_no))
  svm_no_mean <- mean(svm_no)
  
  svm_yes[i] <- mean(c(svm_test_yes, svm_validation_yes))
  svm_yes_mean <- mean(svm_yes)
  
  svm_holdout[i] <- mean(c(svm_test_accuracy_mean, svm_validation_accuracy_mean))
  svm_holdout_mean <- mean(svm_holdout)
  svm_overfitting[i] <- svm_holdout_mean/svm_train_accuracy_mean
  svm_overfitting_mean <- mean(svm_overfitting)
  svm_overfitting_range <- range(svm_overfitting)
  
  svm_table <- svm_test_table + svm_validation_table
  
  svm_table_sum_diag <- sum(diag(svm_table))
  
  
  #### Trees ####
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
  tree_train_prop <- diag(prop.table(tree_train_table, margin = 1))
  tree_train_no[i] <- as.numeric(tree_train_prop[1])
  tree_train_no_mean <- mean(tree_train_no)
  tree_train_yes[i] <- as.numeric(tree_train_prop[2])
  tree_train_yes_mean <- mean(tree_train_yes)
  
  tree_test_pred <- predict(prune_train_tree, test, type = "class")
  tree_test_table <- table(tree_test_pred, y_test)
  tree_test_accuracy[i] <- sum(diag(tree_test_table)) / sum(tree_test_table)
  tree_test_accuracy_mean <- mean(tree_test_accuracy)
  tree_test_diag <- sum(diag(tree_test_table))
  tree_test_mean <- mean(diag(tree_test_table)) / mean(tree_test_table)
  tree_test_sd <- sd(diag(tree_test_table)) / sd(tree_test_table)
  sum_diag_test_tree <- sum(diag(tree_test_table))
  tree_test_prop <- diag(prop.table(tree_test_table, margin = 1))
  tree_test_prop <- diag(prop.table(tree_test_table, margin = 1))
  tree_test_no[i] <- as.numeric(tree_test_prop[1])
  tree_test_no_mean <- mean(tree_test_no)
  tree_test_yes[i] <- as.numeric(tree_test_prop[2])
  tree_test_yes_mean <- mean(tree_test_yes)
  
  tree_validation_pred <- predict(prune_train_tree, validation, type = "class")
  tree_validation_table <- table(tree_validation_pred, y_validation)
  tree_validation_accuracy[i] <- sum(diag(tree_validation_table)) / sum(tree_validation_table)
  tree_validation_accuracy_mean <- mean(tree_validation_accuracy)
  tree_validation_diag <- sum(diag(tree_validation_table))
  tree_validation_mean <- mean(diag(tree_validation_table)) / mean(tree_validation_table)
  tree_validation_sd <- sd(diag(tree_validation_table)) / sd(tree_validation_table)
  sum_diag_validation_tree <- sum(diag(tree_validation_table))
  tree_validation_prop <- diag(prop.table(tree_validation_table, margin = 1))
  tree_validation_prop <- diag(prop.table(tree_validation_table, margin = 1))
  tree_validation_no[i] <- as.numeric(tree_validation_prop[1])
  tree_validation_no_mean <- mean(tree_validation_no)
  tree_validation_yes[i] <- as.numeric(tree_validation_prop[2])
  tree_validation_yes_mean <- mean(tree_validation_yes)
  
  tree_no[i] <- mean(c(tree_test_no, tree_validation_no))
  tree_no_mean <- mean(tree_no)
  
  tree_yes[i] <- mean(c(tree_test_yes, tree_validation_yes))
  tree_yes_mean <- mean(tree_yes)
  
  tree_holdout_mean <- mean(c(tree_test_accuracy_mean, tree_validation_accuracy_mean))
  tree_overfitting[i] <- tree_holdout_mean / tree_train_accuracy_mean
  tree_overfitting_mean <- mean(tree_overfitting)
  tree_overfitting_range <- range(tree_overfitting)
  
  tree_table <- tree_test_table + tree_validation_table
  
  tree_table_sum_diag <- sum(diag(tree_table))
  
  
  ##############################################################################################################################
  
  ######################################## Ensembles start here #################################################################
  
  ################################################################################################################################
  
ensemble1 <- data.frame('ADA bag' = c(as.factor(adabag_test_pred$class), as.factor(adabag_validation_pred$class)),
                        'ADA boost' = c(adaboost_test_pred,adaboost_validation_pred),
                        'Bagged Random Forest' = c(bag_rf_test_pred, bag_rf_validation_pred),
                        'C50' = c(C50_test_pred,C50_validation_pred),
                        'Earth' = c(earth_test_pred, earth_validation_pred),
                        'Flexible Discriminant Analysis' = c(fda_test_pred, fda_validation_pred),
                        'Linear Discriminant Analysis' = c(lda_test_pred$class, lda_validation_pred$class),
                        'Linear' = c(linear_test_pred, linear_validation_pred),
                        'Mixed Discriminant Analysis' = c(mda_test_pred,mda_validation_pred),
                        'Naive Bayes' = c(n_bayes_test_pred, n_bayes_validation_pred),
                        'Quadratic Discriminant Analysis' = c(qda_test_pred$class, qda_validation_pred$class),
                        'Partial Least Squares' = c(pls_test_pred, pls_validation_pred),
                        'Penalized Discriminant Analysis' = c(pda_test_pred, pda_validation_pred),
                        'Random Forest' = c(rf_test_pred, rf_validation_pred),
                        'Ranger' = c(ranger_test_pred, ranger_validation_pred),
                        'Regularized Discriminant Analysis' = c(rda_test_pred$class, rda_validation_pred$class),
                        'Support Vector Machines' = c(svm_test_pred, svm_validation_pred),
                        'Trees' = c(tree_test_pred, tree_validation_pred)
)

  ensemble_row_numbers <- as.numeric(row.names(ensemble1))
  ensemble1$y <- df[ensemble_row_numbers,"type"]
  
  ensemble_index <- sample(c(1:3), nrow(ensemble1), replace=TRUE, prob=c(0.6,0.2, 0.2))
  ensemble_train  <- ensemble1[ensemble_index == 1, ]
  ensemble_test   <- ensemble1[ensemble_index ==2, ]
  ensemble_validation <- ensemble1[ensemble_index == 3, ]
  ensemble_y_train <- ensemble_train$y
  ensemble_y_test <- ensemble_test$y
  ensemble_y_validation <- ensemble_validation$y
  
  
  ### Ensemble with adabag ####
  ensemble_adabag_fit_train <- adabag::bagging(y ~ ., data = ensemble_train)
  ensemble_adabag_train_pred <- predict(ensemble_adabag_fit_train, ensemble_train)
  ensemble_adabag_train_table <- table(ensemble_adabag_train_pred$class, ensemble_y_train)
  ensemble_adabag_train_accuracy[i] <- sum(diag(ensemble_adabag_train_table)) / sum(ensemble_adabag_train_table)
  ensemble_adabag_train_accuracy_mean <- mean(ensemble_adabag_train_accuracy)
  ensemble_adabag_train_mean <- mean(diag(ensemble_adabag_train_table)) / mean(ensemble_adabag_train_table)
  ensemble_adabag_train_sd <- sd(diag(ensemble_adabag_train_table)) / sd(ensemble_adabag_train_table)
  ensemble_adabag_train_sum_diag <- sum(diag(ensemble_adabag_train_table))
  ensemble_adabag_train_prop <- diag(prop.table(ensemble_adabag_train_table, margin = 1))
  ensemble_adabag_train_no[i] <- as.numeric(ensemble_adabag_train_prop[1])
  ensemble_adabag_train_no_mean <- mean(ensemble_adabag_train_no)
  ensemble_adabag_train_yes[i] <- as.numeric(ensemble_adabag_train_prop[2])
  ensemble_adabag_train_yes_mean <- mean(ensemble_adabag_train_yes)
  
  ensemble_adabag_test_pred <- predict(ensemble_adabag_fit_train, ensemble_test)
  ensemble_adabag_test_table <- table(ensemble_adabag_test_pred$class, ensemble_y_test)
  ensemble_adabag_test_accuracy[i] <- sum(diag(ensemble_adabag_test_table)) / sum(ensemble_adabag_test_table)
  ensemble_adabag_test_accuracy_mean <- mean(ensemble_adabag_test_accuracy)
  ensemble_adabag_test_mean <- mean(diag(ensemble_adabag_test_table)) / mean(ensemble_adabag_test_table)
  ensemble_adabag_test_sd <- sd(diag(ensemble_adabag_test_table)) / sd(ensemble_adabag_test_table)
  ensemble_adabag_test_sum_diag <- sum(diag(ensemble_adabag_test_table))
  ensemble_adabag_test_prop <- diag(prop.table(ensemble_adabag_test_table, margin = 1))
  ensemble_adabag_test_no[i] <- as.numeric(ensemble_adabag_test_prop[1])
  ensemble_adabag_test_no_mean <- mean(ensemble_adabag_test_no)
  ensemble_adabag_test_yes[i] <- as.numeric(ensemble_adabag_test_prop[2])
  ensemble_adabag_test_yes_mean <- mean(ensemble_adabag_test_yes)
  
  ensemble_adabag_validation_pred <- predict(ensemble_adabag_fit_train, ensemble_validation)
  ensemble_adabag_validation_table <- table(ensemble_adabag_validation_pred$class, ensemble_y_validation)
  ensemble_adabag_validation_accuracy[i] <- sum(diag(ensemble_adabag_validation_table)) / sum(ensemble_adabag_validation_table)
  ensemble_adabag_validation_accuracy_mean <- mean(ensemble_adabag_validation_accuracy)
  ensemble_adabag_validation_mean <- mean(diag(ensemble_adabag_validation_table)) / mean(ensemble_adabag_validation_table)
  ensemble_adabag_validation_sd <- sd(diag(ensemble_adabag_validation_table)) / sd(ensemble_adabag_validation_table)
  ensemble_adabag_validation_sum_diag <- sum(diag(ensemble_adabag_validation_table))
  ensemble_adabag_validation_prop <- diag(prop.table(ensemble_adabag_validation_table, margin = 1))
  ensemble_adabag_validation_no[i] <- as.numeric(ensemble_adabag_validation_prop[1])
  ensemble_adabag_validation_no_mean <- mean(ensemble_adabag_validation_no)
  ensemble_adabag_validation_yes[i] <- as.numeric(ensemble_adabag_validation_prop[2])
  ensemble_adabag_validation_yes_mean <- mean(ensemble_adabag_validation_yes)
  
  ensemble_adabag_no[i] <- mean(c(ensemble_adabag_test_no, ensemble_adabag_validation_no))
  ensemble_adabag_no_mean <- mean(c(ensemble_adabag_no))
  
  ensemble_adabag_yes[i] <- mean(c(ensemble_adabag_test_yes, ensemble_adabag_validation_yes))
  ensemble_adabag_yes_mean <- mean(c(ensemble_adabag_yes))
  
  ensemble_adabag_holdout[i] <- mean(c(ensemble_adabag_test_accuracy_mean, ensemble_adabag_validation_accuracy_mean))
  ensemble_adabag_holdout_mean <- mean(ensemble_adabag_holdout)
  ensemble_adabag_overfitting[i] <- ensemble_adabag_holdout_mean / ensemble_adabag_train_accuracy_mean
  ensemble_adabag_overfitting_mean <- mean(ensemble_adabag_overfitting)
  ensemble_adabag_overfitting_range <- range(ensemble_adabag_overfitting)
  
  ensemble_adabag_no_mean <- mean(c(ensemble_adabag_test_no, ensemble_adabag_validation_no))
  ensemble_adabag_yes_mean <- mean(c(ensemble_adabag_test_yes, ensemble_adabag_validation_yes))
  
  ensemble_adabag_table <- ensemble_adabag_test_table + ensemble_adabag_validation_table
  
  ensemble_adabag_table_sum_diag <- sum(diag(ensemble_adabag_table))
  

  #### Ensemble Using Adaboost ####
  ensemble_adaboost_train_fit <- MachineShop::fit(formula = y ~ ., data = ensemble_train, model = "AdaBoostModel")
  ensemble_adaboost_train_pred <- predict(ensemble_adaboost_train_fit, ensemble_train)
  ensemble_adaboost_train_table <- table(ensemble_adaboost_train_pred, ensemble_y_train)
  ensemble_adaboost_train_accuracy[i] <- sum(diag(ensemble_adaboost_train_table)) / sum(ensemble_adaboost_train_table)
  ensemble_adaboost_train_accuracy_mean <- mean(ensemble_adaboost_train_accuracy)
  ensemble_adaboost_train_mean <- mean(diag(ensemble_adaboost_train_table)) / mean(ensemble_adaboost_train_table)
  ensemble_adaboost_train_sd <- sd(diag(ensemble_adaboost_train_table)) / sd(ensemble_adaboost_train_table)
  ensemble_adaboost_train_sum_diag <- sum(diag(ensemble_adaboost_train_table))
  ensemble_adaboost_train_prop <- diag(prop.table(ensemble_adaboost_train_table, margin = 1))
  ensemble_adaboost_train_no[i] <- as.numeric(ensemble_adaboost_train_prop[1])
  ensemble_adaboost_train_no_mean <- mean(ensemble_adaboost_train_no)
  ensemble_adaboost_train_yes[i] <- as.numeric(ensemble_adaboost_train_prop[2])
  ensemble_adaboost_train_yes_mean <- mean(ensemble_adaboost_train_yes)
  
  ensemble_adaboost_test_pred <- predict(ensemble_adaboost_train_fit, ensemble_test)
  ensemble_adaboost_test_table <- table(ensemble_adaboost_test_pred, ensemble_y_test)
  ensemble_adaboost_test_accuracy[i] <- sum(diag(ensemble_adaboost_test_table)) / sum(ensemble_adaboost_test_table)
  ensemble_adaboost_test_accuracy_mean <- mean(ensemble_adaboost_test_accuracy)
  ensemble_adaboost_test_mean <- mean(diag(ensemble_adaboost_test_table)) / mean(ensemble_adaboost_test_table)
  ensemble_adaboost_test_sd <- sd(diag(ensemble_adaboost_test_table)) / sd(ensemble_adaboost_test_table)
  ensemble_adaboost_test_sum_diag <- sum(diag(ensemble_adaboost_test_table))
  ensemble_adaboost_test_prop <- diag(prop.table(ensemble_adaboost_test_table, margin = 1))
  ensemble_adaboost_test_no[i] <- as.numeric(ensemble_adaboost_test_prop[1])
  ensemble_adaboost_test_no_mean <- mean(ensemble_adaboost_test_no)
  ensemble_adaboost_test_yes[i] <- as.numeric(ensemble_adaboost_test_prop[2])
  ensemble_adaboost_test_yes_mean <- mean(ensemble_adaboost_test_yes)
  
  ensemble_adaboost_validation_pred <- predict(ensemble_adaboost_train_fit, ensemble_validation)
  ensemble_adaboost_validation_table <- table(ensemble_adaboost_validation_pred, ensemble_y_validation)
  ensemble_adaboost_validation_accuracy[i] <- sum(diag(ensemble_adaboost_validation_table)) / sum(ensemble_adaboost_validation_table)
  ensemble_adaboost_validation_accuracy_mean <- mean(ensemble_adaboost_validation_accuracy)
  ensemble_adaboost_validation_mean <- mean(diag(ensemble_adaboost_validation_table)) / mean(ensemble_adaboost_validation_table)
  ensemble_adaboost_validation_sd <- sd(diag(ensemble_adaboost_validation_table)) / sd(ensemble_adaboost_validation_table)
  ensemble_adaboost_validation_sum_diag <- sum(diag(ensemble_adaboost_validation_table))
  ensemble_adaboost_validation_prop <- diag(prop.table(ensemble_adaboost_validation_table, margin = 1))
  ensemble_adaboost_validation_no[i] <- as.numeric(ensemble_adaboost_validation_prop[1])
  ensemble_adaboost_validation_no_mean <- mean(ensemble_adaboost_validation_no)
  ensemble_adaboost_validation_yes[i] <- as.numeric(ensemble_adaboost_validation_prop[2])
  ensemble_adaboost_validation_yes_mean <- mean(ensemble_adaboost_validation_yes)
  
  ensemble_adaboost_no[i] <- mean(c(ensemble_adaboost_test_no, ensemble_adaboost_validation_no))
  ensemble_adaboost_no_mean <- mean(ensemble_adaboost_no)
  
  ensemble_adaboost_yes[i] <- mean(c(ensemble_adaboost_test_yes, ensemble_adaboost_validation_yes))
  ensemble_adaboost_yes_mean <- mean(ensemble_adaboost_yes)
  
  ensemble_adaboost_holdout[i] <- mean(c(ensemble_adaboost_test_accuracy, ensemble_adaboost_validation_accuracy))
  ensemble_adaboost_holdout_mean <- mean(ensemble_adaboost_holdout)
  ensemble_adaboost_overfitting[i] <- ensemble_adaboost_holdout_mean / ensemble_adaboost_train_accuracy_mean
  ensemble_adaboost_overfitting_mean <- mean(ensemble_adaboost_overfitting)
  ensemble_adaboost_overfitting_range <- range(ensemble_adaboost_overfitting)
  
  ensemble_adaboost_yes_mean <- mean(c(ensemble_adaboost_test_yes_mean, ensemble_adaboost_validation_yes_mean))
  ensemble_adaboost_no_mean <- mean(c(ensemble_adaboost_test_no_mean, ensemble_adaboost_validation_no_mean))
  
  ensemble_adaboost_table <- ensemble_adaboost_test_table + ensemble_adaboost_validation_table
  
  ensemble_adaboost_table_sum_diag <- sum(diag(ensemble_adaboost_table))
  

  #### Ensemble Using Bagging ####
  ensemble_bagging_train_fit <- ipred::bagging(y ~ ., data = ensemble_train, coob = TRUE)
  
  ensemble_bagging_train_pred <- predict(ensemble_bagging_train_fit, ensemble_train)
  ensemble_bagging_train_table <- table(ensemble_bagging_train_pred, ensemble_y_train)
  ensemble_bagging_train_accuracy[i] <- sum(diag(ensemble_bagging_train_table)) / sum(ensemble_bagging_train_table)
  ensemble_bagging_train_accuracy_mean <- mean(ensemble_bagging_train_accuracy)
  ensemble_bagging_train_mean <- mean(diag(ensemble_bagging_train_table)) / mean(ensemble_bagging_train_table)
  ensemble_bagging_train_sd <- sd(diag(ensemble_bagging_train_table)) / sd(ensemble_bagging_train_table)
  ensemble_bagging_train_sum_diag <- sum(diag(ensemble_bagging_train_table))
  ensemble_bagging_train_prop <- diag(prop.table(ensemble_bagging_train_table, margin = 1))
  ensemble_bagging_train_no[i] <- as.numeric(ensemble_bagging_train_prop[1])
  ensemble_bagging_train_no_mean <- mean(ensemble_bagging_train_no)
  ensemble_bagging_train_yes[i] <- as.numeric(ensemble_bagging_train_prop[2])
  ensemble_bagging_train_yes_mean <- mean(ensemble_bagging_train_yes)
  
  ensemble_bagging_test_pred <- predict(ensemble_bagging_train_fit, ensemble_test)
  ensemble_bagging_test_table <- table(ensemble_bagging_test_pred, ensemble_y_test)
  ensemble_bagging_test_accuracy[i] <- sum(diag(ensemble_bagging_test_table)) / sum(ensemble_bagging_test_table)
  ensemble_bagging_test_accuracy_mean <- mean(ensemble_bagging_test_accuracy)
  ensemble_bagging_test_mean <- mean(diag(ensemble_bagging_test_table)) / mean(ensemble_bagging_test_table)
  ensemble_bagging_test_sd <- sd(diag(ensemble_bagging_test_table)) / sd(ensemble_bagging_test_table)
  ensemble_bagging_test_sum_diag <- sum(diag(ensemble_bagging_test_table))
  ensemble_bagging_test_prop <- diag(prop.table(ensemble_bagging_test_table, margin = 1))
  ensemble_bagging_test_no[i] <- as.numeric(ensemble_bagging_test_prop[1])
  ensemble_bagging_test_no_mean <- mean(ensemble_bagging_test_no)
  ensemble_bagging_test_yes[i] <- as.numeric(ensemble_bagging_test_prop[2])
  ensemble_bagging_test_yes_mean <- mean(ensemble_bagging_test_yes)
  
  ensemble_bagging_validation_pred <- predict(ensemble_bagging_train_fit, ensemble_validation)
  ensemble_bagging_validation_table <- table(ensemble_bagging_validation_pred, ensemble_y_validation)
  ensemble_bagging_validation_accuracy[i] <- sum(diag(ensemble_bagging_validation_table)) / sum(ensemble_bagging_validation_table)
  ensemble_bagging_validation_accuracy_mean <- mean(ensemble_bagging_validation_accuracy)
  ensemble_bagging_validation_mean <- mean(diag(ensemble_bagging_validation_table)) / mean(ensemble_bagging_validation_table)
  ensemble_bagging_validation_sd <- sd(diag(ensemble_bagging_validation_table)) / sd(ensemble_bagging_validation_table)
  ensemble_bagging_validation_sum_diag <- sum(diag(ensemble_bagging_validation_table))
  ensemble_bagging_validation_prop <- diag(prop.table(ensemble_bagging_validation_table, margin = 1))
  ensemble_bagging_validation_no[i] <- as.numeric(ensemble_bagging_validation_prop[1])
  ensemble_bagging_validation_no_mean <- mean(ensemble_bagging_validation_no)
  ensemble_bagging_validation_yes[i] <- as.numeric(ensemble_bagging_validation_prop[2])
  ensemble_bagging_validation_yes_mean <- mean(ensemble_bagging_validation_yes)
  
  ensemble_bagging_no[i] <- mean(c(ensemble_bagging_test_no, ensemble_bagging_validation_no))
  ensemble_bagging_no_mean <- mean(ensemble_bagging_no)
  
  ensemble_bagging_yes[i] <- mean(c(ensemble_bagging_test_yes, ensemble_bagging_validation_yes))
  ensemble_bagging_yes_mean <- mean(ensemble_bagging_yes)
  
  ensemble_bagging_holdout[i] <- mean(c(ensemble_bagging_test_accuracy_mean, ensemble_bagging_validation_accuracy_mean))
  ensemble_bagging_holdout_mean <- mean(ensemble_bagging_holdout)
  ensemble_bagging_overfitting[i] <- ensemble_bagging_holdout_mean / ensemble_bagging_train_accuracy_mean
  ensemble_bagging_overfitting_mean <- mean(ensemble_bagging_overfitting)
  ensemble_bagging_overfitting_range <- range(ensemble_bagging_overfitting)
  
  ensemble_bagging_no_mean <- mean(c(ensemble_bagging_test_no, ensemble_bagging_validation_no))
  ensemble_bagging_yes_mean <- mean(c(ensemble_bagging_test_yes, ensemble_bagging_validation_yes))
  
  ensemble_bagging_table <- ensemble_bagging_test_table + ensemble_bagging_validation_table
  
  ensemble_bagging_table_sum_diag <- sum(diag(ensemble_bagging_table))
  

  #### Ensemble Using Bagged Random Forest ####
  ensemble_bag_rf_train_fit <- randomForest(ensemble_y_train ~ ., data = ensemble_train)
  
  ensemble_bag_rf_train_pred <- predict(ensemble_bag_rf_train_fit, ensemble_train)
  ensemble_bag_rf_train_table <- table(ensemble_bag_rf_train_pred, ensemble_y_train)
  ensemble_bag_rf_train_accuracy[i] <- sum(diag(ensemble_bag_rf_train_table)) / sum(ensemble_bag_rf_train_table)
  ensemble_bag_rf_train_accuracy_mean <- mean(ensemble_bag_rf_train_accuracy)
  ensemble_bag_rf_train_mean <- mean(diag(ensemble_bag_rf_train_table)) / mean(ensemble_bag_rf_train_table)
  ensemble_bag_rf_train_sd <- sd(diag(ensemble_bag_rf_train_table)) / sd(ensemble_bag_rf_train_table)
  ensemble_bag_rf_train_sum_diag <- sum(diag(ensemble_bag_rf_train_table))
  ensemble_bag_rf_train_prop <- diag(prop.table(ensemble_bag_rf_train_table, margin = 1))
  ensemble_bag_rf_train_no[i] <- as.numeric(ensemble_bag_rf_train_prop[1])
  ensemble_bag_rf_train_no_mean <- mean( ensemble_bag_rf_train_no)
  ensemble_bag_rf_train_yes[i] <- as.numeric(ensemble_bag_rf_train_prop[2])
  ensemble_bag_rf_train_yes_mean <- mean(ensemble_bag_rf_train_yes)
  
  ensemble_bag_rf_test_pred <- predict(ensemble_bag_rf_train_fit, ensemble_test)
  ensemble_bag_rf_test_table <- table(ensemble_bag_rf_test_pred, ensemble_y_test)
  ensemble_bag_rf_test_accuracy[i] <- sum(diag(ensemble_bag_rf_test_table)) / sum(ensemble_bag_rf_test_table)
  ensemble_bag_rf_test_accuracy_mean <- mean(ensemble_bag_rf_test_accuracy)
  ensemble_bag_rf_test_mean <- mean(diag(ensemble_bag_rf_test_table)) / mean(ensemble_bag_rf_test_table)
  ensemble_bag_rf_test_sd <- sd(diag(ensemble_bag_rf_test_table)) / sd(ensemble_bag_rf_test_table)
  ensemble_bag_rf_test_sum_diag <- sum(diag(ensemble_bag_rf_test_table))
  ensemble_bag_rf_test_prop <- diag(prop.table(ensemble_bag_rf_test_table, margin = 1))
  ensemble_bag_rf_test_no[i] <- as.numeric(ensemble_bag_rf_test_prop[1])
  ensemble_bag_rf_test_no_mean <- mean(ensemble_bag_rf_test_no)
  ensemble_bag_rf_test_yes[i] <- as.numeric(ensemble_bag_rf_test_prop[2])
  ensemble_bag_rf_test_yes_mean <- mean(ensemble_bag_rf_test_yes)
  
  ensemble_bag_rf_validation_pred <- predict(ensemble_bag_rf_train_fit, ensemble_validation)
  ensemble_bag_rf_validation_table <- table(ensemble_bag_rf_validation_pred, ensemble_y_validation)
  ensemble_bag_rf_validation_accuracy[i] <- sum(diag(ensemble_bag_rf_validation_table)) / sum(ensemble_bag_rf_validation_table)
  ensemble_bag_rf_validation_accuracy_mean <- mean(ensemble_bag_rf_validation_accuracy)
  ensemble_bag_rf_validation_mean <- mean(diag(ensemble_bag_rf_validation_table)) / mean(ensemble_bag_rf_validation_table)
  ensemble_bag_rf_validation_sd <- sd(diag(ensemble_bag_rf_validation_table)) / sd(ensemble_bag_rf_validation_table)
  ensemble_bag_rf_validation_sum_diag <- sum(diag(ensemble_bag_rf_validation_table))
  ensemble_bag_rf_validation_prop <- diag(prop.table(ensemble_bag_rf_validation_table, margin = 1))
  ensemble_bag_rf_validation_no[i] <- as.numeric(ensemble_bag_rf_validation_prop[1])
  ensemble_bag_rf_validation_no_mean <- mean(ensemble_bag_rf_validation_no)
  ensemble_bag_rf_validation_yes[i] <- as.numeric(ensemble_bag_rf_validation_prop[2])
  ensemble_bag_rf_validation_yes_mean <- mean(ensemble_bag_rf_validation_yes)
  
  ensemble_bag_rf_holdout[i] <- mean(c(ensemble_bag_rf_test_accuracy_mean, ensemble_bag_rf_validation_accuracy_mean))
  ensemble_bag_rf_holdout_mean <- mean(ensemble_bag_rf_holdout)
  ensemble_bag_rf_overfitting[i] <- ensemble_bag_rf_holdout_mean / ensemble_bag_rf_train_accuracy_mean
  ensemble_bag_rf_overfitting_mean <- mean(ensemble_bag_rf_overfitting)
  ensemble_bag_rf_overfitting_range <- range(ensemble_bag_rf_overfitting)
  
  ensemble_bag_rf_no <- mean(c(ensemble_bag_rf_test_no, ensemble_bag_rf_validation_no))
  ensemble_bag_rf_yes <- mean(c(ensemble_bag_rf_test_yes, ensemble_bag_rf_validation_yes))
  
  ensemble_bag_rf_table <- ensemble_bag_rf_test_table + ensemble_bag_rf_validation_table
  
  ensemble_bag_rf_table_sum_diag <- sum(diag(ensemble_bag_rf_table))
  

  #### Ensemble Using C50
  ensemble_C50_train_fit <- C5.0(as.factor(y) ~ ., data = ensemble_train)
  
  ensemble_C50_train_pred <- predict(ensemble_C50_train_fit, ensemble_train)
  ensemble_C50_train_table <- table(ensemble_C50_train_pred, ensemble_y_train)
  ensemble_C50_train_accuracy[i] <- sum(diag(ensemble_C50_train_table)) / sum(ensemble_C50_train_table)
  ensemble_C50_train_accuracy_mean <- mean(ensemble_C50_train_accuracy)
  ensemble_C50_train_mean <- mean(diag(ensemble_C50_train_table)) / mean(ensemble_C50_train_table)
  ensemble_C50_train_sd <- sd(diag(ensemble_C50_train_table)) / sd(ensemble_C50_train_table)
  ensemble_C50_train_sum_diag <- sum(diag(ensemble_C50_train_table))
  ensemble_C50_train_prop <- diag(prop.table(ensemble_C50_train_table, margin = 1))
  ensemble_C50_train_no[i] <- as.numeric(ensemble_C50_train_prop[1])
  ensemble_C50_train_no_mean <- mean(ensemble_C50_train_no)
  ensemble_C50_train_yes[i] <- as.numeric(ensemble_C50_train_prop[2])
  ensemble_C50_train_yes_mean <- mean(ensemble_C50_train_yes)
  
  ensemble_C50_test_pred <- predict(ensemble_C50_train_fit, ensemble_test)
  ensemble_C50_test_table <- table(ensemble_C50_test_pred, ensemble_y_test)
  ensemble_C50_test_accuracy[i] <- sum(diag(ensemble_C50_test_table)) / sum(ensemble_C50_test_table)
  ensemble_C50_test_accuracy_mean <- mean(ensemble_C50_test_accuracy)
  ensemble_C50_test_mean <- mean(diag(ensemble_C50_test_table)) / mean(ensemble_C50_test_table)
  ensemble_C50_test_sd <- sd(diag(ensemble_C50_test_table)) / sd(ensemble_C50_test_table)
  ensemble_C50_test_sum_diag <- sum(diag(ensemble_C50_test_table))
  ensemble_C50_test_prop <- diag(prop.table(ensemble_C50_test_table, margin = 1))
  ensemble_C50_test_no[i] <- as.numeric(ensemble_C50_test_prop[1])
  ensemble_C50_test_no_mean <- mean(ensemble_C50_test_no)
  ensemble_C50_test_yes[i] <- as.numeric(ensemble_C50_test_prop[2])
  ensemble_C50_test_yes_mean <- mean(ensemble_C50_test_yes)
  
  ensemble_C50_validation_pred <- predict(ensemble_C50_train_fit, ensemble_validation)
  ensemble_C50_validation_table <- table(ensemble_C50_validation_pred, ensemble_y_validation)
  ensemble_C50_validation_accuracy[i] <- sum(diag(ensemble_C50_validation_table)) / sum(ensemble_C50_validation_table)
  ensemble_C50_validation_accuracy_mean <- mean(ensemble_C50_validation_accuracy)
  ensemble_C50_validation_mean <- mean(diag(ensemble_C50_validation_table)) / mean(ensemble_C50_validation_table)
  ensemble_C50_validation_sd <- sd(diag(ensemble_C50_validation_table)) / sd(ensemble_C50_validation_table)
  ensemble_C50_validation_sum_diag <- sum(diag(ensemble_C50_validation_table))
  ensemble_C50_validation_prop <- diag(prop.table(ensemble_C50_validation_table, margin = 1))
  ensemble_C50_validation_no[i] <- as.numeric(ensemble_C50_validation_prop[1])
  ensemble_C50_validation_no_mean <- mean(ensemble_C50_validation_no)
  ensemble_C50_validation_yes[i] <- as.numeric(ensemble_C50_validation_prop[2])
  ensemble_C50_validation_yes_mean <- mean(ensemble_C50_validation_yes)
  
  ensemble_C50_no[i] <- mean(c(ensemble_C50_test_no, ensemble_C50_validation_no))
  ensemble_C50_no_mean <- mean(ensemble_C50_no)
  
  ensemble_C50_yes[i] <- mean(c(ensemble_C50_test_yes, ensemble_C50_validation_yes))
  ensemble_C50_yes_mean <- mean(ensemble_C50_yes)
  
  ensemble_C50_holdout[i] <- mean(c(ensemble_C50_test_accuracy_mean, ensemble_C50_validation_accuracy_mean))
  ensemble_C50_holdout_mean <- mean(ensemble_C50_holdout)
  ensemble_C50_overfitting[i] <- ensemble_C50_holdout_mean / ensemble_C50_train_accuracy_mean
  ensemble_C50_overfitting_mean <- mean(ensemble_C50_overfitting)
  ensemble_C50_overfitting_range <- range(ensemble_C50_overfitting)
  
  ensemble_C50_table <- ensemble_C50_test_table + ensemble_C50_validation_table
  
  ensemble_C50_table_sum_diag <- sum(diag(ensemble_C50_table))
  

  #### Ensemble Using Earth model ####
  ensemble_earth_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "EarthModel")
  
  ensemble_earth_train_pred <- predict(ensemble_earth_train_fit, ensemble_train)
  ensemble_earth_train_table <- table(ensemble_earth_train_pred, ensemble_y_train)
  ensemble_earth_train_accuracy[i] <- sum(diag(ensemble_earth_train_table)) / sum(ensemble_earth_train_table)
  ensemble_earth_train_accuracy_mean <- mean(ensemble_earth_train_accuracy)
  ensemble_earth_train_mean <- mean(diag(ensemble_earth_train_table)) / mean(ensemble_earth_train_table)
  ensemble_earth_train_sd <- sd(diag(ensemble_earth_train_table)) / sd(ensemble_earth_train_table)
  ensemble_earth_train_sum_diag <- sum(diag(ensemble_earth_train_table))
  ensemble_earth_train_prop <- diag(prop.table(ensemble_earth_train_table, margin = 1))
  ensemble_earth_train_no[i] <- as.numeric(ensemble_earth_train_prop[1])
  ensemble_earth_train_no_mean <- mean(ensemble_earth_train_no)
  ensemble_earth_train_yes[i] <- as.numeric(ensemble_earth_train_prop[2])
  ensemble_earth_train_yes_mean <- mean(ensemble_earth_train_yes)
  
  ensemble_earth_test_pred <- predict(ensemble_earth_train_fit, ensemble_test)
  ensemble_earth_test_table <- table(ensemble_earth_test_pred, ensemble_y_test)
  ensemble_earth_test_accuracy[i] <- sum(diag(ensemble_earth_test_table)) / sum(ensemble_earth_test_table)
  ensemble_earth_test_accuracy_mean <- mean(ensemble_earth_test_accuracy)
  ensemble_earth_test_mean <- mean(diag(ensemble_earth_test_table)) / mean(ensemble_earth_test_table)
  ensemble_earth_test_sd <- sd(diag(ensemble_earth_test_table)) / sd(ensemble_earth_test_table)
  ensemble_earth_test_sum_diag <- sum(diag(ensemble_earth_test_table))
  ensemble_earth_test_prop <- diag(prop.table(ensemble_earth_test_table, margin = 1))
  ensemble_earth_test_no[i] <- as.numeric(ensemble_earth_test_prop[1])
  ensemble_earth_test_no_mean <- mean(ensemble_earth_test_no)
  ensemble_earth_test_yes[i] <- as.numeric(ensemble_earth_test_prop[2])
  ensemble_earth_test_yes_mean <- mean(ensemble_earth_test_yes)
  
  ensemble_earth_validation_pred <- predict(ensemble_earth_train_fit, ensemble_validation)
  ensemble_earth_validation_table <- table(ensemble_earth_validation_pred, ensemble_y_validation)
  ensemble_earth_validation_accuracy[i] <- sum(diag(ensemble_earth_validation_table)) / sum(ensemble_earth_validation_table)
  ensemble_earth_validation_accuracy_mean <- mean(ensemble_earth_validation_accuracy)
  ensemble_earth_validation_mean <- mean(diag(ensemble_earth_validation_table)) / mean(ensemble_earth_validation_table)
  ensemble_earth_validation_sd <- sd(diag(ensemble_earth_validation_table)) / sd(ensemble_earth_validation_table)
  ensemble_earth_validation_sum_diag <- sum(diag(ensemble_earth_validation_table))
  ensemble_earth_validation_prop <- diag(prop.table(ensemble_earth_validation_table, margin = 1))
  ensemble_earth_validation_no[i] <- as.numeric(ensemble_earth_validation_prop[1])
  ensemble_earth_validation_no_mean <- mean(ensemble_earth_validation_no)
  ensemble_earth_validation_yes[i] <- as.numeric(ensemble_earth_validation_prop[2])
  ensemble_earth_validation_yes_mean <- mean(ensemble_earth_validation_yes)
  
  ensemble_earth_no[i] <- mean(c(ensemble_earth_test_no, ensemble_earth_validation_no))
  ensemble_earth_no_mean <- mean(ensemble_earth_no)
  
  ensemble_earth_yes[i] <- mean(c(ensemble_earth_test_yes, ensemble_earth_validation_yes))
  ensemble_earth_yes_mean <- mean(ensemble_earth_yes)
  
  ensemble_earth_holdout[i] <- mean(c(ensemble_earth_test_accuracy_mean, ensemble_earth_validation_accuracy_mean))
  ensemble_earth_holdout_mean <- mean(ensemble_earth_holdout)
  ensemble_earth_overfitting[i] <- ensemble_earth_holdout_mean / ensemble_earth_train_accuracy_mean
  ensemble_earth_overfitting_mean <- mean(ensemble_earth_overfitting)
  ensemble_earth_overfitting_range <- range(ensemble_earth_overfitting)
  
  ensemble_earth_table <- ensemble_earth_test_table + ensemble_earth_validation_table
  
  ensemble_earth_table_sum_diag <- sum(diag(ensemble_earth_table))


  #### Ensemble Using Flexible discriminant analysis ####
  ensemble_fda_train_fit <- mda::fda(y ~ ., data = ensemble_train)
  
  ensemble_fda_train_table <- ensemble_fda_train_fit$confusion
  ensemble_fda_train_accuracy[i] <- sum(diag(ensemble_fda_train_table)) / sum(ensemble_fda_train_table)
  ensemble_fda_train_accuracy_mean <- mean(ensemble_fda_train_accuracy)
  ensemble_fda_train_mean <- mean(diag(ensemble_fda_train_table)) / mean(ensemble_fda_train_table)
  ensemble_fda_train_sd <- sd(diag(ensemble_fda_train_table)) / sd(ensemble_fda_train_table)
  ensemble_fda_train_sum_diag <- sum(diag(ensemble_fda_train_table))
  ensemble_fda_train_prop <- diag(prop.table(ensemble_fda_train_table, margin = 1))
  ensemble_fda_train_no[i] <- as.numeric(ensemble_fda_train_prop[1])
  ensemble_fda_train_no_mean <- mean(ensemble_fda_train_no)
  ensemble_fda_train_yes[i] <- as.numeric(ensemble_fda_train_prop[2])
  ensemble_fda_train_yes_mean <- mean(ensemble_fda_train_yes)
  
  ensemble_fda_test_model <- mda::fda(y ~ ., data = ensemble_test)
  ensemble_fda_test_table <- ensemble_fda_test_model$confusion
  ensemble_fda_test_accuracy[i] <- sum(diag(ensemble_fda_test_table)) / sum(ensemble_fda_test_table)
  ensemble_fda_test_accuracy_mean <- mean(ensemble_fda_test_accuracy)
  ensemble_fda_test_mean <- mean(diag(ensemble_fda_test_table)) / mean(ensemble_fda_test_table)
  ensemble_fda_test_sd <- sd(diag(ensemble_fda_test_table)) / sd(ensemble_fda_test_table)
  ensemble_fda_test_sum_diag <- sum(diag(ensemble_fda_test_table))
  ensemble_fda_test_prop <- diag(prop.table(ensemble_fda_test_table, margin = 1))
  ensemble_fda_test_no[i] <- as.numeric(ensemble_fda_test_prop[1])
  ensemble_fda_test_no_mean <- mean(ensemble_fda_test_no)
  ensemble_fda_test_yes[i] <- as.numeric(ensemble_fda_test_prop[2])
  ensemble_fda_test_yes_mean <- mean(ensemble_fda_test_yes)
  
  ensemble_fda_validation_model <- mda::fda(y ~ ., data = ensemble_validation)
  ensemble_fda_validation_table <- ensemble_fda_validation_model$confusion
  ensemble_fda_validation_accuracy[i] <- sum(diag(ensemble_fda_validation_table)) / sum(ensemble_fda_validation_table)
  ensemble_fda_validation_accuracy_mean <- mean(ensemble_fda_validation_accuracy)
  ensemble_fda_validation_mean <- mean(diag(ensemble_fda_validation_table)) / mean(ensemble_fda_validation_table)
  ensemble_fda_validation_sd <- sd(diag(ensemble_fda_validation_table)) / sd(ensemble_fda_validation_table)
  ensemble_fda_validation_sum_diag <- sum(diag(ensemble_fda_validation_table))
  ensemble_fda_validation_prop <- diag(prop.table(ensemble_fda_validation_table, margin = 1))
  ensemble_fda_validation_no[i] <- as.numeric(ensemble_fda_validation_prop[1])
  ensemble_fda_validation_no_mean <- mean(ensemble_fda_validation_no)
  ensemble_fda_validation_yes[i] <- as.numeric(ensemble_fda_validation_prop[2])
  ensemble_fda_validation_yes_mean <- mean(ensemble_fda_validation_yes)
  
  ensemble_fda_no[i] <- mean(c(ensemble_fda_test_no, ensemble_fda_validation_no))
  ensemble_fda_no_mean <- mean(ensemble_fda_no)
  
  ensemble_fda_yes[i] <- mean(c(ensemble_fda_test_yes, ensemble_fda_validation_yes))
  ensemble_fda_yes_mean <- mean(ensemble_fda_yes)
  
  ensemble_fda_holdout[i] <- mean(c(ensemble_fda_test_accuracy_mean, ensemble_fda_validation_accuracy_mean))
  ensemble_fda_holdout_mean <- mean(ensemble_fda_holdout)
  ensemble_fda_overfitting[i] <- ensemble_fda_holdout_mean / ensemble_fda_train_accuracy_mean
  ensemble_fda_overfitting_mean <- mean(ensemble_fda_overfitting)
  ensemble_fda_overfitting_range <- range(ensemble_fda_overfitting)
  
  ensemble_fda_table <- ensemble_fda_test_table + ensemble_fda_validation_table
  
  ensemble_fda_table_sum_diag <- sum(diag(ensemble_fda_table))


  #### Ensembles using Linear Modeling ####
  ensemble_linear_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "LMModel")
  
  ensemble_linear_train_pred <- predict(ensemble_linear_train_fit, ensemble_train)
  ensemble_linear_train_table <- table(ensemble_linear_train_pred, ensemble_y_train)
  ensemble_linear_train_accuracy[i] <- sum(diag(ensemble_linear_train_table)) / sum(ensemble_linear_train_table)
  ensemble_linear_train_accuracy_mean <- mean(ensemble_linear_train_accuracy)
  ensemble_linear_train_mean <- mean(diag(ensemble_linear_train_table)) / mean(ensemble_linear_train_table)
  ensemble_linear_train_sd <- sd(diag(ensemble_linear_train_table)) / sd(ensemble_linear_train_table)
  ensemble_linear_train_sum_diag <- sum(diag(ensemble_linear_train_table))
  ensemble_linear_train_prop <- diag(prop.table(ensemble_linear_train_table, margin = 1))
  ensemble_linear_train_no[i] <- as.numeric(ensemble_linear_train_prop[1])
  ensemble_linear_train_no_mean <- mean(ensemble_linear_train_no)
  ensemble_linear_train_yes[i] <- as.numeric(ensemble_linear_train_prop[2])
  ensemble_linear_train_yes_mean <- mean(ensemble_linear_train_yes)
  
  ensemble_linear_test_pred <- predict(ensemble_linear_train_fit, ensemble_test)
  ensemble_linear_test_table <- table(ensemble_linear_test_pred, ensemble_y_test)
  ensemble_linear_test_accuracy[i] <- sum(diag(ensemble_linear_test_table)) / sum(ensemble_linear_test_table)
  ensemble_linear_test_accuracy_mean <- mean(ensemble_linear_test_accuracy)
  ensemble_linear_test_mean <- mean(diag(ensemble_linear_test_table)) / mean(ensemble_linear_test_table)
  ensemble_linear_test_sd <- sd(diag(ensemble_linear_test_table)) / sd(ensemble_linear_test_table)
  ensemble_linear_test_sum_diag <- sum(diag(ensemble_linear_test_table))
  ensemble_linear_test_prop <- diag(prop.table(ensemble_linear_test_table, margin = 1))
  ensemble_linear_test_no[i] <- as.numeric(ensemble_linear_test_prop[1])
  ensemble_linear_test_no_mean <- mean(ensemble_linear_test_no)
  ensemble_linear_test_yes[i] <- as.numeric(ensemble_linear_test_prop[2])
  ensemble_linear_test_yes_mean <- mean(ensemble_linear_test_yes)
  
  ensemble_linear_validation_pred <- predict(ensemble_linear_train_fit, ensemble_validation)
  ensemble_linear_validation_table <- table(ensemble_linear_validation_pred, ensemble_y_validation)
  ensemble_linear_validation_accuracy[i] <- sum(diag(ensemble_linear_validation_table)) / sum(ensemble_linear_validation_table)
  ensemble_linear_validation_accuracy_mean <- mean(ensemble_linear_validation_accuracy)
  ensemble_linear_validation_mean <- mean(diag(ensemble_linear_validation_table)) / mean(ensemble_linear_validation_table)
  ensemble_linear_validation_sd <- sd(diag(ensemble_linear_validation_table)) / sd(ensemble_linear_validation_table)
  ensemble_linear_validation_sum_diag <- sum(diag(ensemble_linear_validation_table))
  ensemble_linear_validation_prop <- diag(prop.table(ensemble_linear_validation_table, margin = 1))
  ensemble_linear_validation_no[i] <- as.numeric(ensemble_linear_validation_prop[1])
  ensemble_linear_validation_no_mean <- mean(ensemble_linear_validation_no)
  ensemble_linear_validation_yes[i] <- as.numeric(ensemble_linear_validation_prop[2])
  ensemble_linear_validation_yes_mean <- mean(ensemble_linear_validation_yes)
  
  ensemble_linear_no[i] <- mean(c(ensemble_linear_test_no, ensemble_linear_validation_no))
  ensemble_linear_no_mean <- mean(ensemble_linear_no)
  
  ensemble_linear_yes[i] <- mean(c(ensemble_linear_test_yes, ensemble_linear_validation_yes))
  ensemble_linear_yes_mean <- mean(ensemble_linear_yes)
  
  ensemble_linear_holdout[i] <- mean(c(ensemble_linear_test_accuracy_mean, ensemble_linear_validation_accuracy_mean))
  ensemble_linear_holdout_mean <- mean(ensemble_linear_holdout)
  ensemble_linear_overfitting[i] <- ensemble_linear_holdout_mean / ensemble_linear_train_accuracy_mean
  ensemble_linear_overfitting_mean <- mean(ensemble_linear_overfitting)
  ensemble_linear_overfitting_range <- range(ensemble_linear_overfitting)
  
  ensemble_linear_table <- ensemble_linear_test_table + ensemble_linear_validation_table
  
  ensemble_linear_table_sum_diag <- sum(diag(ensemble_linear_table))


  #### Ensemble Using Naive Bayes ####
  ensemble_n_bayes_train_fit <- naiveBayes(y ~ ., data = ensemble_train)
  
  ensemble_n_bayes_train_pred <- predict(ensemble_n_bayes_train_fit, ensemble_train)
  ensemble_n_bayes_train_table <- table(ensemble_n_bayes_train_pred, ensemble_y_train)
  ensemble_n_bayes_train_accuracy[i] <- sum(diag(ensemble_n_bayes_train_table)) / sum(ensemble_n_bayes_train_table)
  ensemble_n_bayes_train_accuracy_mean <- mean(ensemble_n_bayes_train_accuracy)
  ensemble_n_bayes_train_mean <- mean(diag(ensemble_n_bayes_train_table)) / mean(ensemble_n_bayes_train_table)
  ensemble_n_bayes_train_sd <- sd(diag(ensemble_n_bayes_train_table)) / sd(ensemble_n_bayes_train_table)
  ensemble_n_bayes_train_sum_diag <- sum(diag(ensemble_n_bayes_train_table))
  ensemble_n_bayes_train_prop <- diag(prop.table(ensemble_n_bayes_train_table, margin = 1))
  ensemble_n_bayes_train_no[i] <- as.numeric(ensemble_n_bayes_train_prop[1])
  ensemble_n_bayes_train_no_mean <- mean(ensemble_n_bayes_train_no)
  ensemble_n_bayes_train_yes[i] <- as.numeric(ensemble_n_bayes_train_prop[2])
  ensemble_n_bayes_train_yes_mean <- mean(ensemble_n_bayes_train_yes)
  
  ensemble_n_bayes_test_pred <- predict(ensemble_n_bayes_train_fit, ensemble_test)
  ensemble_n_bayes_test_table <- table(ensemble_n_bayes_test_pred, ensemble_y_test)
  ensemble_n_bayes_test_accuracy[i] <- sum(diag(ensemble_n_bayes_test_table)) / sum(ensemble_n_bayes_test_table)
  ensemble_n_bayes_test_accuracy_mean <- mean(ensemble_n_bayes_test_accuracy)
  ensemble_n_bayes_test_mean <- mean(diag(ensemble_n_bayes_test_table)) / mean(ensemble_n_bayes_test_table)
  ensemble_n_bayes_test_sd <- sd(diag(ensemble_n_bayes_test_table)) / sd(ensemble_n_bayes_test_table)
  ensemble_n_bayes_test_sum_diag <- sum(diag(ensemble_n_bayes_test_table))
  ensemble_n_bayes_test_prop <- diag(prop.table(ensemble_n_bayes_test_table, margin = 1))
  ensemble_n_bayes_test_no[i] <- as.numeric(ensemble_n_bayes_test_prop[1])
  ensemble_n_bayes_test_no_mean <- mean(ensemble_n_bayes_test_no)
  ensemble_n_bayes_test_yes[i] <- as.numeric(ensemble_n_bayes_test_prop[2])
  ensemble_n_bayes_test_yes_mean <- mean(ensemble_n_bayes_test_yes)
  
  ensemble_n_bayes_validation_pred <- predict(ensemble_n_bayes_train_fit, ensemble_validation)
  ensemble_n_bayes_validation_table <- table(ensemble_n_bayes_validation_pred, ensemble_y_validation)
  ensemble_n_bayes_validation_accuracy[i] <- sum(diag(ensemble_n_bayes_validation_table)) / sum(ensemble_n_bayes_validation_table)
  ensemble_n_bayes_validation_accuracy_mean <- mean(ensemble_n_bayes_validation_accuracy)
  ensemble_n_bayes_validation_mean <- mean(diag(ensemble_n_bayes_validation_table)) / mean(ensemble_n_bayes_validation_table)
  ensemble_n_bayes_validation_sd <- sd(diag(ensemble_n_bayes_validation_table)) / sd(ensemble_n_bayes_validation_table)
  ensemble_n_bayes_validation_sum_diag <- sum(diag(ensemble_n_bayes_validation_table))
  ensemble_n_bayes_validation_prop <- diag(prop.table(ensemble_n_bayes_validation_table, margin = 1))
  ensemble_n_bayes_validation_no[i] <- as.numeric(ensemble_n_bayes_validation_prop[1])
  ensemble_n_bayes_validation_no_mean <- mean(ensemble_n_bayes_validation_no)
  ensemble_n_bayes_validation_yes[i] <- as.numeric(ensemble_n_bayes_validation_prop[2])
  ensemble_n_bayes_validation_yes_mean <- mean(ensemble_n_bayes_validation_yes)
  
  ensemble_n_bayes_no[i] <- mean(c(ensemble_n_bayes_test_no, ensemble_n_bayes_validation_no))
  ensemble_n_bayes_no_mean <- mean(ensemble_n_bayes_no)
  
  ensemble_n_bayes_yes[i] <- mean(c(ensemble_n_bayes_test_yes, ensemble_n_bayes_validation_yes))
  ensemble_n_bayes_yes_mean <- mean(ensemble_n_bayes_yes)
  
  ensemble_n_bayes_holdout[i] <- mean(c(ensemble_n_bayes_test_accuracy_mean, ensemble_n_bayes_validation_accuracy_mean))
  ensemble_n_bayes_holdout_mean <- mean(ensemble_n_bayes_holdout)
  ensemble_n_bayes_overfitting[i] <- ensemble_n_bayes_holdout_mean / ensemble_n_bayes_train_accuracy_mean
  ensemble_n_bayes_overfitting_mean <- mean(ensemble_n_bayes_overfitting)
  ensemble_n_bayes_overfitting_range <- range(ensemble_n_bayes_overfitting)
  
  ensemble_n_bayes_table <- ensemble_n_bayes_test_table + ensemble_n_bayes_validation_table
  
  ensemble_n_bayes_table_sum_diag <- sum(diag(ensemble_n_bayes_table))
  

  #### Ensemble Using Partial Least Squares ####
  ensemble_pls_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "PLSModel")
  
  ensemble_pls_train_pred <- predict(ensemble_pls_train_fit, ensemble_train)
  ensemble_pls_train_table <- table(ensemble_pls_train_pred, ensemble_y_train)
  ensemble_pls_train_accuracy[i] <- sum(diag(ensemble_pls_train_table)) / sum(ensemble_pls_train_table)
  ensemble_pls_train_accuracy_mean <- mean(ensemble_pls_train_accuracy)
  ensemble_pls_train_mean <- mean(diag(ensemble_pls_train_table)) / mean(ensemble_pls_train_table)
  ensemble_pls_train_sd <- sd(diag(ensemble_pls_train_table)) / sd(ensemble_pls_train_table)
  ensemble_pls_train_sum_diag <- sum(diag(ensemble_pls_train_table))
  ensemble_pls_train_prop <- diag(prop.table(ensemble_pls_train_table, margin = 1))
  ensemble_pls_train_no[i] <- as.numeric(ensemble_pls_train_prop[1])
  ensemble_pls_train_no_mean <- mean(ensemble_pls_train_no)
  ensemble_pls_train_yes[i] <- as.numeric(ensemble_pls_train_prop[2])
  ensemble_pls_train_yes_mean <- mean(ensemble_pls_train_yes)
  
  ensemble_pls_test_pred <- predict(ensemble_pls_train_fit, ensemble_test)
  ensemble_pls_test_table <- table(ensemble_pls_test_pred, ensemble_y_test)
  ensemble_pls_test_accuracy[i] <- sum(diag(ensemble_pls_test_table)) / sum(ensemble_pls_test_table)
  ensemble_pls_test_accuracy_mean <- mean(ensemble_pls_test_accuracy)
  ensemble_pls_test_mean <- mean(diag(ensemble_pls_test_table)) / mean(ensemble_pls_test_table)
  ensemble_pls_test_sd <- sd(diag(ensemble_pls_test_table)) / sd(ensemble_pls_test_table)
  ensemble_pls_test_sum_diag <- sum(diag(ensemble_pls_test_table))
  ensemble_pls_test_prop <- diag(prop.table(ensemble_pls_test_table, margin = 1))
  ensemble_pls_test_no[i] <- as.numeric(ensemble_pls_test_prop[1])
  ensemble_pls_test_no_mean <- mean(ensemble_pls_test_no)
  ensemble_pls_test_yes[i] <- as.numeric(ensemble_pls_test_prop[2])
  ensemble_pls_test_yes_mean <- mean(ensemble_pls_test_yes)
  
  ensemble_pls_validation_pred <- predict(ensemble_pls_train_fit, ensemble_validation)
  ensemble_pls_validation_table <- table(ensemble_pls_validation_pred, ensemble_y_validation)
  ensemble_pls_validation_accuracy[i] <- sum(diag(ensemble_pls_validation_table)) / sum(ensemble_pls_validation_table)
  ensemble_pls_validation_accuracy_mean <- mean(ensemble_pls_validation_accuracy)
  ensemble_pls_validation_mean <- mean(diag(ensemble_pls_validation_table)) / mean(ensemble_pls_validation_table)
  ensemble_pls_validation_sd <- sd(diag(ensemble_pls_validation_table)) / sd(ensemble_pls_validation_table)
  ensemble_pls_validation_sum_diag <- sum(diag(ensemble_pls_validation_table))
  ensemble_pls_validation_prop <- diag(prop.table(ensemble_pls_validation_table, margin = 1))
  ensemble_pls_validation_no[i] <- as.numeric(ensemble_pls_validation_prop[1])
  ensemble_pls_validation_no_mean <- mean(ensemble_pls_validation_no)
  ensemble_pls_validation_yes[i] <- as.numeric(ensemble_pls_validation_prop[2])
  ensemble_pls_validation_yes_mean <- mean(ensemble_pls_validation_yes)
  
  ensemble_pls_no[i] <- mean(c(ensemble_pls_test_no, ensemble_pls_validation_no))
  ensemble_pls_no_mean <- mean(ensemble_pls_no)
  
  ensemble_pls_yes[i] <- mean(c(ensemble_pls_test_yes, ensemble_pls_validation_yes))
  ensemble_pls_yes_mean <- mean(ensemble_pls_yes)
  
  ensemble_pls_holdout[i] <- mean(c(ensemble_pls_test_accuracy_mean, ensemble_pls_validation_accuracy_mean))
  ensemble_pls_holdout_mean <- mean(ensemble_pls_holdout)
  ensemble_pls_overfitting[i] <- ensemble_pls_holdout_mean / ensemble_pls_train_accuracy_mean
  ensemble_pls_overfitting_mean <- mean(ensemble_pls_overfitting)
  ensemble_pls_overfitting_range <- range(ensemble_pls_overfitting)
  
  ensemble_pls_table <- ensemble_pls_test_table + ensemble_pls_validation_table
  
  ensemble_pls_table_sum_diag <- sum(diag(ensemble_pls_table))
  

  #### Ensembles Using Penalized Discriminant Analysis ####
  ensemble_pda_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "PDAModel")
  
  ensemble_pda_train_pred <- predict(ensemble_pda_train_fit, ensemble_train)
  ensemble_pda_train_table <- table(ensemble_pda_train_pred, ensemble_y_train)
  ensemble_pda_train_accuracy[i] <- sum(diag(ensemble_pda_train_table)) / sum(ensemble_pda_train_table)
  ensemble_pda_train_accuracy_mean <- mean(ensemble_pda_train_accuracy)
  ensemble_pda_train_mean <- mean(diag(ensemble_pda_train_table)) / mean(ensemble_pda_train_table)
  ensemble_pda_train_sd <- sd(diag(ensemble_pda_train_table)) / sd(ensemble_pda_train_table)
  ensemble_pda_train_sum_diag <- sum(diag(ensemble_pda_train_table))
  ensemble_pda_train_prop <- diag(prop.table(ensemble_pda_train_table, margin = 1))
  ensemble_pda_train_no[i] <- as.numeric(ensemble_pda_train_prop[1])
  ensemble_pda_train_no_mean <- mean(ensemble_pda_train_no)
  ensemble_pda_train_yes[i] <- as.numeric(ensemble_pda_train_prop[2])
  ensemble_pda_train_yes_mean <- mean(ensemble_pda_train_yes)
  
  ensemble_pda_test_pred <- predict(ensemble_pda_train_fit, ensemble_test)
  ensemble_pda_test_table <- table(ensemble_pda_test_pred, ensemble_y_test)
  ensemble_pda_test_accuracy[i] <- sum(diag(ensemble_pda_test_table)) / sum(ensemble_pda_test_table)
  ensemble_pda_test_accuracy_mean <- mean(ensemble_pda_test_accuracy)
  ensemble_pda_test_mean <- mean(diag(ensemble_pda_test_table)) / mean(ensemble_pda_test_table)
  ensemble_pda_test_sd <- sd(diag(ensemble_pda_test_table)) / sd(ensemble_pda_test_table)
  ensemble_pda_test_sum_diag <- sum(diag(ensemble_pda_test_table))
  ensemble_pda_test_prop <- diag(prop.table(ensemble_pda_test_table, margin = 1))
  ensemble_pda_test_no[i] <- as.numeric(ensemble_pda_test_prop[1])
  ensemble_pda_test_no_mean <- mean(ensemble_pda_test_no)
  ensemble_pda_test_yes[i] <- as.numeric(ensemble_pda_test_prop[2])
  ensemble_pda_test_yes_mean <- mean(ensemble_pda_test_yes)
  
  ensemble_pda_validation_pred <- predict(ensemble_pda_train_fit, ensemble_validation)
  ensemble_pda_validation_table <- table(ensemble_pda_validation_pred, ensemble_y_validation)
  ensemble_pda_validation_accuracy[i] <- sum(diag(ensemble_pda_validation_table)) / sum(ensemble_pda_validation_table)
  ensemble_pda_validation_accuracy_mean <- mean(ensemble_pda_validation_accuracy)
  ensemble_pda_validation_mean <- mean(diag(ensemble_pda_validation_table)) / mean(ensemble_pda_validation_table)
  ensemble_pda_validation_sd <- sd(diag(ensemble_pda_validation_table)) / sd(ensemble_pda_validation_table)
  ensemble_pda_validation_sum_diag <- sum(diag(ensemble_pda_validation_table))
  ensemble_pda_validation_prop <- diag(prop.table(ensemble_pda_validation_table, margin = 1))
  ensemble_pda_validation_no[i] <- as.numeric(ensemble_pda_validation_prop[1])
  ensemble_pda_validation_no_mean <- mean(ensemble_pda_validation_no)
  ensemble_pda_validation_yes[i] <- as.numeric(ensemble_pda_validation_prop[2])
  ensemble_pda_validation_yes_mean <- mean(ensemble_pda_validation_yes)
  
  ensemble_pda_no[i] <- mean(c(ensemble_pda_test_no, ensemble_pda_validation_no))
  ensemble_pda_no_mean <- mean(ensemble_pda_no)
  
  ensemble_pda_yes[i] <- mean(c(ensemble_pda_test_yes, ensemble_pda_validation_yes))
  ensemble_pda_yes_mean <- mean(ensemble_pda_yes)
  
  ensemble_pda_holdout[i] <- mean(c(ensemble_pda_test_accuracy_mean, ensemble_pda_validation_accuracy_mean))
  ensemble_pda_holdout_mean <- mean(ensemble_pda_holdout)
  ensemble_pda_overfitting[i] <- ensemble_pda_holdout_mean / ensemble_pda_train_accuracy_mean
  ensemble_pda_overfitting_mean <- mean(ensemble_pda_overfitting)
  ensemble_pda_overfitting_range <- range(ensemble_pda_overfitting)
  
  ensemble_pda_table <- ensemble_pda_test_table + ensemble_pda_validation_table
  
  ensemble_pda_table_sum_diag <- sum(diag(ensemble_pda_table))
  

  #### Ensembles Using Random Forest ####
  ensemble_tune_train_rf <- tune.randomForest(x = ensemble_train, y = ensemble_y_train, data = ensemble1, mtry = 1:(ncol(train)-1))
  
  ensemble_rf_train_pred <- predict(ensemble_tune_train_rf$best.model, ensemble_train)
  ensemble_rf_train_table <- table(ensemble_rf_train_pred, ensemble_y_train)
  ensemble_rf_train_accuracy[i] <- sum(diag(ensemble_rf_train_table)) / sum(ensemble_rf_train_table)
  ensemble_rf_train_accuracy_mean <- mean(ensemble_rf_train_accuracy)
  ensemble_rf_train_mean <- mean(diag(ensemble_rf_train_table)) / mean(ensemble_rf_train_table)
  ensemble_rf_train_sd <- sd(diag(ensemble_rf_train_table)) / sd(ensemble_rf_train_table)
  ensemble_rf_train_sum_diag <- sum(diag(ensemble_rf_train_table))
  ensemble_rf_train_prop <- diag(prop.table(ensemble_rf_train_table, margin = 1))
  ensemble_rf_train_no[i] <- as.numeric(ensemble_rf_train_prop[1])
  ensemble_rf_train_no_mean <- mean(ensemble_rf_train_no)
  ensemble_rf_train_yes[i] <- as.numeric(ensemble_rf_train_prop[2])
  ensemble_rf_train_yes_mean <- mean(ensemble_rf_train_yes)
  
  ensemble_rf_test_pred <- predict(ensemble_tune_train_rf$best.model, ensemble_test)
  ensemble_rf_test_table <- table(ensemble_rf_test_pred, ensemble_y_test)
  ensemble_rf_test_accuracy[i] <- sum(diag(ensemble_rf_test_table)) / sum(ensemble_rf_test_table)
  ensemble_rf_test_accuracy_mean <- mean(ensemble_rf_test_accuracy)
  ensemble_rf_test_mean <- mean(diag(ensemble_rf_test_table)) / mean(ensemble_rf_test_table)
  ensemble_rf_test_sd <- sd(diag(ensemble_rf_test_table)) / sd(ensemble_rf_test_table)
  ensemble_rf_test_sum_diag <- sum(diag(ensemble_rf_test_table))
  ensemble_rf_test_prop <- diag(prop.table(ensemble_rf_test_table, margin = 1))
  ensemble_rf_test_no[i] <- as.numeric(ensemble_rf_test_prop[1])
  ensemble_rf_test_no_mean <- mean(ensemble_rf_test_no)
  ensemble_rf_test_yes[i] <- as.numeric(ensemble_rf_test_prop[2])
  ensemble_rf_test_yes_mean <- mean(ensemble_rf_test_yes)
  
  ensemble_rf_validation_pred <- predict(ensemble_tune_train_rf$best.model, ensemble_validation)
  ensemble_rf_validation_table <- table(ensemble_rf_validation_pred, ensemble_y_validation)
  ensemble_rf_validation_accuracy[i] <- sum(diag(ensemble_rf_validation_table)) / sum(ensemble_rf_validation_table)
  ensemble_rf_validation_accuracy_mean <- mean(ensemble_rf_validation_accuracy)
  ensemble_rf_validation_mean <- mean(diag(ensemble_rf_validation_table)) / mean(ensemble_rf_validation_table)
  ensemble_rf_validation_sd <- sd(diag(ensemble_rf_validation_table)) / sd(ensemble_rf_validation_table)
  ensemble_rf_validation_sum_diag <- sum(diag(ensemble_rf_validation_table))
  ensemble_rf_validation_prop <- diag(prop.table(ensemble_rf_validation_table, margin = 1))
  ensemble_rf_validation_no[i] <- as.numeric(ensemble_rf_validation_prop[1])
  ensemble_rf_validation_no_mean <- mean(ensemble_rf_validation_no)
  ensemble_rf_validation_yes[i] <- as.numeric(ensemble_rf_validation_prop[2])
  ensemble_rf_validation_yes_mean <- mean(ensemble_rf_validation_yes)
  
  ensemble_rf_no[i] <- mean(c(ensemble_rf_test_no, ensemble_rf_validation_no))
  ensemble_rf_no_mean <- mean(ensemble_rf_no)
  
  ensemble_rf_yes[i] <- mean(c(ensemble_rf_test_yes, ensemble_rf_validation_yes))
  ensemble_rf_yes_mean <- mean(ensemble_rf_yes)
  
  ensemble_rf_holdout[i] <- mean(c(ensemble_rf_test_accuracy_mean, ensemble_rf_validation_accuracy_mean))
  ensemble_rf_holdout_mean <- mean(ensemble_rf_holdout)
  ensemble_rf_overfitting[i] <- ensemble_rf_holdout_mean / ensemble_rf_train_accuracy_mean
  ensemble_rf_overfitting_mean <- mean(ensemble_rf_overfitting)
  ensemble_rf_overfitting_range <- range(ensemble_rf_overfitting)
  
  ensemble_rf_table <- ensemble_rf_test_table + ensemble_rf_validation_table
  
  ensemble_rf_table_sum_diag <- sum(diag(ensemble_rf_table))
  

  #### Ensemble Using Ranger ####
  ensemble_ranger_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RangerModel")
  
  ensemble_ranger_train_pred <- predict(ensemble_ranger_train_fit, ensemble_train)
  ensemble_ranger_train_table <- table(ensemble_ranger_train_pred, ensemble_y_train)
  ensemble_ranger_train_accuracy[i] <- sum(diag(ensemble_ranger_train_table)) / sum(ensemble_ranger_train_table)
  ensemble_ranger_train_accuracy_mean <- mean(ensemble_ranger_train_accuracy)
  ensemble_ranger_train_mean <- mean(diag(ensemble_ranger_train_table)) / mean(ensemble_ranger_train_table)
  ensemble_ranger_train_sd <- sd(diag(ensemble_ranger_train_table)) / sd(ensemble_ranger_train_table)
  ensemble_ranger_train_sum_diag <- sum(diag(ensemble_ranger_train_table))
  ensemble_ranger_train_prop <- diag(prop.table(ensemble_ranger_train_table, margin = 1))
  ensemble_ranger_train_no[i] <- as.numeric(ensemble_ranger_train_prop[1])
  ensemble_ranger_train_no_mean <- mean(ensemble_ranger_train_no)
  ensemble_ranger_train_yes[i] <- as.numeric(ensemble_ranger_train_prop[2])
  ensemble_ranger_train_yes_mean <- mean(ensemble_ranger_train_yes)
  
  ensemble_ranger_test_pred <- predict(ensemble_ranger_train_fit, ensemble_test)
  ensemble_ranger_test_table <- table(ensemble_ranger_test_pred, ensemble_y_test)
  ensemble_ranger_test_accuracy[i] <- sum(diag(ensemble_ranger_test_table)) / sum(ensemble_ranger_test_table)
  ensemble_ranger_test_accuracy_mean <- mean(ensemble_ranger_test_accuracy)
  ensemble_ranger_test_mean <- mean(diag(ensemble_ranger_test_table)) / mean(ensemble_ranger_test_table)
  ensemble_ranger_test_sd <- sd(diag(ensemble_ranger_test_table)) / sd(ensemble_ranger_test_table)
  ensemble_ranger_test_sum_diag <- sum(diag(ensemble_ranger_test_table))
  ensemble_ranger_test_prop <- diag(prop.table(ensemble_ranger_test_table, margin = 1))
  ensemble_ranger_test_no[i] <- as.numeric(ensemble_ranger_test_prop[1])
  ensemble_ranger_test_no_mean <- mean(ensemble_ranger_test_no)
  ensemble_ranger_test_yes[i] <- as.numeric(ensemble_ranger_test_prop[2])
  ensemble_ranger_test_yes_mean <- mean(ensemble_ranger_test_yes)
  
  ensemble_ranger_validation_pred <- predict(ensemble_ranger_train_fit, ensemble_validation)
  ensemble_ranger_validation_table <- table(ensemble_ranger_validation_pred, ensemble_y_validation)
  ensemble_ranger_validation_accuracy[i] <- sum(diag(ensemble_ranger_validation_table)) / sum(ensemble_ranger_validation_table)
  ensemble_ranger_validation_accuracy_mean <- mean(ensemble_ranger_validation_accuracy)
  ensemble_ranger_validation_mean <- mean(diag(ensemble_ranger_validation_table)) / mean(ensemble_ranger_validation_table)
  ensemble_ranger_validation_sd <- sd(diag(ensemble_ranger_validation_table)) / sd(ensemble_ranger_validation_table)
  ensemble_ranger_validation_sum_diag <- sum(diag(ensemble_ranger_validation_table))
  ensemble_ranger_validation_prop <- diag(prop.table(ensemble_ranger_validation_table, margin = 1))
  ensemble_ranger_validation_no[i] <- as.numeric(ensemble_ranger_validation_prop[1])
  ensemble_ranger_validation_no_mean <- mean(ensemble_ranger_validation_no)
  ensemble_ranger_validation_yes[i] <- as.numeric(ensemble_ranger_validation_prop[2])
  ensemble_ranger_validation_yes_mean <- mean(ensemble_ranger_validation_yes)
  
  ensemble_ranger_no[i] <- mean(c(ensemble_ranger_test_no, ensemble_ranger_validation_no))
  ensemble_ranger_no_mean <- mean(ensemble_ranger_no)
  
  ensemble_ranger_yes[i] <- mean(c(ensemble_ranger_test_yes, ensemble_ranger_validation_yes))
  ensemble_ranger_yes_mean <- mean(ensemble_ranger_yes)
  
  ensemble_ranger_holdout[i] <- mean(c(ensemble_ranger_test_accuracy_mean, ensemble_ranger_validation_accuracy_mean))
  ensemble_ranger_holdout_mean <- mean(ensemble_ranger_holdout)
  ensemble_ranger_overfitting[i] <- ensemble_ranger_holdout_mean / ensemble_ranger_train_accuracy_mean
  ensemble_ranger_overfitting_mean <- mean(ensemble_ranger_overfitting)
  ensemble_ranger_overfitting_range <- range(ensemble_ranger_overfitting)
  
  ensemble_ranger_table <- ensemble_ranger_test_table + ensemble_ranger_validation_table
  
  ensemble_ranger_table_sum_diag <- sum(diag(ensemble_ranger_table))
  

  #### Ensemble Using Regularized Discrmininat Analysis ####
  ensemble_row_numbers <- as.numeric(row.names(ensemble1))
  ensemble1$y <- df[ensemble_row_numbers,"type"]
  
  ensemble_index <- sample(c(1:3), nrow(ensemble1), replace=TRUE, prob=c(0.6,0.2, 0.2))
  ensemble_train  <- ensemble1[ensemble_index == 1, ]
  ensemble_test   <- ensemble1[ensemble_index ==2, ]
  ensemble_validation <- ensemble1[ensemble_index == 3, ]
  ensemble_y_train <- ensemble_train$y
  ensemble_y_test <- ensemble_test$y
  ensemble_y_validation <- ensemble_validation$y
  
  ensemble_rda_train_fit <- klaR::rda(y ~ ., data = ensemble_train)
  
  ensemble_rda_train_pred <- predict(ensemble_rda_train_fit, ensemble_train)
  ensemble_rda_train_table <- table(ensemble_rda_train_pred$class, ensemble_y_train)
  ensemble_rda_train_accuracy[i] <- sum(diag(ensemble_rda_train_table)) / sum(ensemble_rda_train_table)
  ensemble_rda_train_accuracy_mean <- mean(ensemble_rda_train_accuracy)
  ensemble_rda_train_mean <- mean(diag(ensemble_rda_train_table)) / mean(ensemble_rda_train_table)
  ensemble_rda_train_sd <- sd(diag(ensemble_rda_train_table)) / sd(ensemble_rda_train_table)
  ensemble_rda_train_sum_diag <- sum(diag(ensemble_rda_train_table))
  ensemble_rda_train_prop <- diag(prop.table(ensemble_rda_train_table, margin = 1))
  ensemble_rda_train_no[i] <- as.numeric(ensemble_rda_train_prop[1])
  ensemble_rda_train_no_mean <- mean(ensemble_rda_train_no)
  ensemble_rda_train_yes[i] <- as.numeric(ensemble_rda_train_prop[2])
  
  ensemble_rda_test_pred <- predict(ensemble_rda_train_fit, ensemble_test)
  ensemble_rda_test_table <- table(ensemble_rda_test_pred$class, ensemble_y_test)
  ensemble_rda_test_accuracy[i] <- sum(diag(ensemble_rda_test_table)) / sum(ensemble_rda_test_table)
  ensemble_rda_test_accuracy_mean <- mean(ensemble_rda_test_accuracy)
  ensemble_rda_test_mean <- mean(diag(ensemble_rda_test_table)) / mean(ensemble_rda_test_table)
  ensemble_rda_test_sd <- sd(diag(ensemble_rda_test_table)) / sd(ensemble_rda_test_table)
  ensemble_rda_test_sum_diag <- sum(diag(ensemble_rda_test_table))
  ensemble_rda_test_prop <- diag(prop.table(ensemble_rda_test_table, margin = 1))
  ensemble_rda_test_no[i] <- as.numeric(ensemble_rda_test_prop[1])
  ensemble_rda_test_no_mean <- mean(ensemble_rda_test_no)
  ensemble_rda_test_yes[i] <- as.numeric(ensemble_rda_test_prop[2])
  ensemble_rda_test_yes_mean <- mean(ensemble_rda_test_yes)
  
  ensemble_rda_validation_pred <- predict(ensemble_rda_train_fit, ensemble_validation)
  ensemble_rda_validation_table <- table(ensemble_rda_validation_pred$class, ensemble_y_validation)
  ensemble_rda_validation_accuracy[i] <- sum(diag(ensemble_rda_validation_table)) / sum(ensemble_rda_validation_table)
  ensemble_rda_validation_accuracy_mean <- mean(ensemble_rda_validation_accuracy)
  ensemble_rda_validation_mean <- mean(diag(ensemble_rda_validation_table)) / mean(ensemble_rda_validation_table)
  ensemble_rda_validation_sd <- sd(diag(ensemble_rda_validation_table)) / sd(ensemble_rda_validation_table)
  ensemble_rda_validation_sum_diag <- sum(diag(ensemble_rda_validation_table))
  ensemble_rda_validation_prop <- diag(prop.table(ensemble_rda_validation_table, margin = 1))
  ensemble_rda_validation_no[i] <- as.numeric(ensemble_rda_validation_prop[1])
  ensemble_rda_validation_no_mean <- mean(ensemble_rda_validation_no)
  ensemble_rda_validation_yes[i] <- as.numeric(ensemble_rda_validation_prop[2])
  ensemble_rda_validation_yes_mean <- mean( ensemble_rda_validation_yes)
  
  ensemble_rda_no[i] <- mean(c(ensemble_rda_test_no, ensemble_rda_validation_no))
  ensemble_rda_no_mean <- mean(ensemble_rda_no)
  
  ensemble_rda_yes[i] <- mean(c(ensemble_rda_test_yes, ensemble_rda_validation_yes))
  ensemble_rda_yes_mean <- mean(ensemble_rda_yes)
  
  ensemble_rda_holdout[i] <- mean(c(ensemble_rda_test_accuracy_mean, ensemble_rda_validation_accuracy_mean))
  ensemble_rda_holdout_mean <- mean(ensemble_rda_holdout)
  ensemble_rda_overfitting[i] <- ensemble_rda_holdout_mean/ensemble_rda_train_accuracy_mean
  ensemble_rda_overfitting_mean <- mean(ensemble_rda_overfitting)
  ensemble_rda_overfitting_range <- range(ensemble_rda_overfitting)
  
  ensemble_rda_table <- ensemble_rda_test_table + ensemble_rda_validation_table
  
  ensemble_rda_table_sum_diag <- sum(diag(ensemble_rda_table))


  #### Ensembles Using Rpart ####
  ensemble_rpart_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RPartModel")
  
  ensemble_rpart_train_pred <- predict(ensemble_rpart_train_fit, ensemble_train)
  ensemble_rpart_train_table <- table(ensemble_rpart_train_pred, ensemble_y_train)
  ensemble_rpart_train_accuracy[i] <- sum(diag(ensemble_rpart_train_table)) / sum(ensemble_rpart_train_table)
  ensemble_rpart_train_accuracy_mean <- mean(ensemble_rpart_train_accuracy)
  ensemble_rpart_train_mean <- mean(diag(ensemble_rpart_train_table)) / mean(ensemble_rpart_train_table)
  ensemble_rpart_train_sd <- sd(diag(ensemble_rpart_train_table)) / sd(ensemble_rpart_train_table)
  ensemble_rpart_train_sum_diag <- sum(diag(ensemble_rpart_train_table))
  ensemble_rpart_train_prop <- diag(prop.table(ensemble_rpart_train_table, margin = 1))
  ensemble_rpart_train_no[i] <- as.numeric(ensemble_rpart_train_prop[1])
  ensemble_rpart_train_no_mean <- mean(ensemble_rpart_train_no)
  ensemble_rpart_train_yes[i] <- as.numeric(ensemble_rpart_train_prop[2])
  ensemble_rpart_train_yes_mean <- mean(ensemble_rpart_train_yes)
  
  ensemble_rpart_test_pred <- predict(ensemble_rpart_train_fit, ensemble_test)
  ensemble_rpart_test_table <- table(ensemble_rpart_test_pred, ensemble_y_test)
  ensemble_rpart_test_accuracy[i] <- sum(diag(ensemble_rpart_test_table)) / sum(ensemble_rpart_test_table)
  ensemble_rpart_test_accuracy_mean <- mean(ensemble_rpart_test_accuracy)
  ensemble_rpart_test_mean <- mean(diag(ensemble_rpart_test_table)) / mean(ensemble_rpart_test_table)
  ensemble_rpart_test_sd <- sd(diag(ensemble_rpart_test_table)) / sd(ensemble_rpart_test_table)
  ensemble_rpart_test_sum_diag <- sum(diag(ensemble_rpart_test_table))
  ensemble_rpart_test_prop <- diag(prop.table(ensemble_rpart_test_table, margin = 1))
  ensemble_rpart_test_no[i] <- as.numeric(ensemble_rpart_test_prop[1])
  ensemble_rpart_test_no_mean <- mean(ensemble_rpart_test_no)
  ensemble_rpart_test_yes[i] <- as.numeric(ensemble_rpart_test_prop[2])
  ensemble_rpart_test_yes_mean <- mean(ensemble_rpart_test_yes)
  
  ensemble_rpart_validation_pred <- predict(ensemble_rpart_train_fit, ensemble_validation)
  ensemble_rpart_validation_table <- table(ensemble_rpart_validation_pred, ensemble_y_validation)
  ensemble_rpart_validation_accuracy[i] <- sum(diag(ensemble_rpart_validation_table)) / sum(ensemble_rpart_validation_table)
  ensemble_rpart_validation_accuracy_mean <- mean(ensemble_rpart_validation_accuracy)
  ensemble_rpart_validation_mean <- mean(diag(ensemble_rpart_validation_table)) / mean(ensemble_rpart_validation_table)
  ensemble_rpart_validation_sd <- sd(diag(ensemble_rpart_validation_table)) / sd(ensemble_rpart_validation_table)
  ensemble_rpart_validation_sum_diag <- sum(diag(ensemble_rpart_validation_table))
  ensemble_rpart_validation_prop <- diag(prop.table(ensemble_rpart_validation_table, margin = 1))
  ensemble_rpart_validation_no[i] <- as.numeric(ensemble_rpart_validation_prop[1])
  ensemble_rpart_validation_no_mean <- mean(ensemble_rpart_validation_no)
  ensemble_rpart_validation_yes[i] <- as.numeric(ensemble_rpart_validation_prop[2])
  ensemble_rpart_validation_yes_mean <- mean(ensemble_rpart_validation_yes)
  
  ensemble_rpart_no[i] <- mean(c(ensemble_rpart_test_no, ensemble_rpart_validation_no))
  ensemble_rpart_no_mean <- mean(ensemble_rpart_no)
  
  ensemble_rpart_yes[i] <- mean(c(ensemble_rpart_test_yes, ensemble_rpart_validation_yes))
  ensemble_rpart_yes_mean <- mean(c(ensemble_rpart_yes))
  
  ensemble_rpart_holdout[i] <- mean(c(ensemble_rpart_test_accuracy_mean, ensemble_rpart_validation_accuracy_mean))
  ensemble_rpart_holdout_mean <- mean(ensemble_rpart_holdout)
  ensemble_rpart_overfitting[i] <- ensemble_rpart_holdout_mean / ensemble_rpart_train_accuracy_mean
  ensemble_rpart_overfitting_mean <- mean(ensemble_rpart_overfitting)
  ensemble_rpart_overfitting_range <- range(ensemble_rpart_overfitting)
  
  ensemble_rpart_table <- ensemble_rpart_test_table + ensemble_rpart_validation_table
  
  ensemble_rpart_table_sum_diag <- sum(diag(ensemble_rpart_table))
  
  
  #### Ensembles Using Support Vector Machines (SVM) ####
  ensemble_svm_train_fit <- svm(y ~ ., data = ensemble_train, kernel = "radial", gamma = 1, cost = 1)
  
  ensemble_svm_train_pred <- predict(ensemble_svm_train_fit, ensemble_train)
  ensemble_svm_train_table <- table(ensemble_svm_train_pred, ensemble_y_train)
  ensemble_svm_train_accuracy[i] <- sum(diag(ensemble_svm_train_table)) / sum(ensemble_svm_train_table)
  ensemble_svm_train_accuracy_mean <- mean(ensemble_svm_train_accuracy)
  ensemble_svm_train_mean <- mean(diag(ensemble_svm_train_table)) / mean(ensemble_svm_train_table)
  ensemble_svm_train_sd <- sd(diag(ensemble_svm_train_table)) / sd(ensemble_svm_train_table)
  ensemble_svm_train_sum_diag <- sum(diag(ensemble_svm_train_table))
  ensemble_svm_train_prop <- diag(prop.table(ensemble_svm_train_table, margin = 1))
  ensemble_svm_train_no[i] <- as.numeric(ensemble_svm_train_prop[1])
  ensemble_svm_train_no_mean <- mean(ensemble_svm_train_no)
  ensemble_svm_train_yes[i] <- as.numeric(ensemble_svm_train_prop[2])
  ensemble_svm_train_yes_mean <- mean(ensemble_svm_train_yes)
  
  ensemble_svm_test_pred <- predict(ensemble_svm_train_fit, ensemble_test)
  ensemble_svm_test_table <- table(ensemble_svm_test_pred, ensemble_y_test)
  ensemble_svm_test_accuracy[i] <- sum(diag(ensemble_svm_test_table)) / sum(ensemble_svm_test_table)
  ensemble_svm_test_accuracy_mean <- mean(ensemble_svm_test_accuracy)
  ensemble_svm_test_mean <- mean(diag(ensemble_svm_test_table)) / mean(ensemble_svm_test_table)
  ensemble_svm_test_sd <- sd(diag(ensemble_svm_test_table)) / sd(ensemble_svm_test_table)
  ensemble_svm_test_sum_diag <- sum(diag(ensemble_svm_test_table))
  ensemble_svm_test_prop <- diag(prop.table(ensemble_svm_test_table, margin = 1))
  ensemble_svm_test_no[i] <- as.numeric(ensemble_svm_test_prop[1])
  ensemble_svm_test_no_mean <- mean(ensemble_svm_test_no)
  ensemble_svm_test_yes[i] <- as.numeric(ensemble_svm_test_prop[2])
  ensemble_svm_test_yes_mean <- mean(ensemble_svm_test_yes)
  
  ensemble_svm_validation_pred <- predict(ensemble_svm_train_fit, ensemble_validation)
  ensemble_svm_validation_table <- table(ensemble_svm_validation_pred, ensemble_y_validation)
  ensemble_svm_validation_accuracy[i] <- sum(diag(ensemble_svm_validation_table)) / sum(ensemble_svm_validation_table)
  ensemble_svm_validation_accuracy_mean <- mean(ensemble_svm_validation_accuracy)
  ensemble_svm_validation_mean <- mean(diag(ensemble_svm_validation_table)) / mean(ensemble_svm_validation_table)
  ensemble_svm_validation_sd <- sd(diag(ensemble_svm_validation_table)) / sd(ensemble_svm_validation_table)
  ensemble_svm_validation_sum_diag <- sum(diag(ensemble_svm_validation_table))
  ensemble_svm_validation_prop <- diag(prop.table(ensemble_svm_validation_table, margin = 1))
  ensemble_svm_validation_no[i] <- as.numeric(ensemble_svm_validation_prop[1])
  ensemble_svm_validation_no_mean <- mean(ensemble_svm_validation_no)
  ensemble_svm_validation_yes[i] <- as.numeric(ensemble_svm_validation_prop[2])
  ensemble_svm_validation_yes_mean <- mean(ensemble_svm_validation_yes)
  
  ensemble_svm_no[i] <- mean(c(ensemble_svm_test_no, ensemble_svm_validation_no))
  ensemble_svm_no_mean <- mean(ensemble_svm_no)
  
  ensemble_svm_yes[i] <- mean(c(ensemble_svm_test_yes, ensemble_svm_validation_yes))
  ensemble_svm_yes_mean <- mean(ensemble_svm_yes)
  
  ensemble_svm_holdout[i] <- mean(c(ensemble_svm_test_accuracy_mean, ensemble_svm_validation_accuracy_mean))
  ensemble_svm_holdout_mean <- mean(ensemble_svm_holdout)
  ensemble_svm_overfitting[i] <- ensemble_svm_holdout_mean/ensemble_svm_train_accuracy_mean
  ensemble_svm_overfitting_mean <- mean(ensemble_svm_overfitting)
  ensemble_svm_overfitting_range <- range(ensemble_svm_overfitting)
  
  ensemble_svm_table <- ensemble_svm_test_table + ensemble_svm_validation_table
  
  ensemble_svm_table_sum_diag <- sum(diag(ensemble_svm_table))
  
  
  #### Ensembles Using ensemble.trees ####
  ensemble_tree_train_fit <- tree(y ~ ., data = ensemble_train)
  
  cv_train_ensemble_tree <- cv.tree(object = ensemble_tree_train_fit, FUN = prune.misclass)
  prune_train_ensemble_tree <- prune.misclass(ensemble_tree_train_fit, best = 5)
  ensemble_tree_train_pred <- predict(prune_train_ensemble_tree, ensemble_train, type = "class")
  ensemble_tree_train_table <- table(ensemble_tree_train_pred, ensemble_y_train)
  ensemble_tree_train_accuracy[i] <- sum(diag(ensemble_tree_train_table)) / sum(ensemble_tree_train_table)
  ensemble_tree_train_accuracy_mean <- mean(ensemble_tree_train_accuracy)
  ensemble_tree_train_diag <- sum(diag(ensemble_tree_train_table))
  ensemble_tree_train_mean <- mean(diag(ensemble_tree_train_table)) / mean(ensemble_tree_train_table)
  ensemble_tree_train_sd <- sd(diag(ensemble_tree_train_table)) / sd(ensemble_tree_train_table)
  sum_diag_train_ensemble_tree <- sum(diag(ensemble_tree_train_table))
  ensemble_tree_train_prop <- diag(prop.table(ensemble_tree_train_table, margin = 1))
  ensemble_tree_train_prop <- diag(prop.table(ensemble_tree_train_table, margin = 1))
  ensemble_tree_train_no[i] <- as.numeric(ensemble_tree_train_prop[1])
  ensemble_tree_train_no_mean <- mean(ensemble_tree_train_no)
  ensemble_tree_train_yes[i] <- as.numeric(ensemble_tree_train_prop[2])
  ensemble_tree_train_yes_mean <- mean(ensemble_tree_train_yes)
  
  ensemble_tree_test_pred <- predict(prune_train_ensemble_tree, ensemble_test, type = "class")
  ensemble_tree_test_table <- table(ensemble_tree_test_pred, ensemble_y_test)
  ensemble_tree_test_accuracy[i] <- sum(diag(ensemble_tree_test_table)) / sum(ensemble_tree_test_table)
  ensemble_tree_test_accuracy_mean <- mean(ensemble_tree_test_accuracy)
  ensemble_tree_test_diag <- sum(diag(ensemble_tree_test_table))
  ensemble_tree_test_mean <- mean(diag(ensemble_tree_test_table)) / mean(ensemble_tree_test_table)
  ensemble_tree_test_sd <- sd(diag(ensemble_tree_test_table)) / sd(ensemble_tree_test_table)
  sum_diag_test_ensemble_tree <- sum(diag(ensemble_tree_test_table))
  ensemble_tree_test_prop <- diag(prop.table(ensemble_tree_test_table, margin = 1))
  ensemble_tree_test_prop <- diag(prop.table(ensemble_tree_test_table, margin = 1))
  ensemble_tree_test_no[i] <- as.numeric(ensemble_tree_test_prop[1])
  ensemble_tree_test_no_mean <- mean(ensemble_tree_test_no)
  ensemble_tree_test_yes[i] <- as.numeric(ensemble_tree_test_prop[2])
  ensemble_tree_test_yes_mean <- mean(ensemble_tree_test_yes)
  
  ensemble_tree_validation_pred <- predict(prune_train_ensemble_tree, ensemble_validation, type = "class")
  ensemble_tree_validation_table <- table(ensemble_tree_validation_pred, ensemble_y_validation)
  ensemble_tree_validation_accuracy[i] <- sum(diag(ensemble_tree_validation_table)) / sum(ensemble_tree_validation_table)
  ensemble_tree_validation_accuracy_mean <- mean(ensemble_tree_validation_accuracy)
  ensemble_tree_validation_diag <- sum(diag(ensemble_tree_validation_table))
  ensemble_tree_validation_mean <- mean(diag(ensemble_tree_validation_table)) / mean(ensemble_tree_validation_table)
  ensemble_tree_validation_sd <- sd(diag(ensemble_tree_validation_table)) / sd(ensemble_tree_validation_table)
  sum_diag_validation_ensemble_tree <- sum(diag(ensemble_tree_validation_table))
  ensemble_tree_validation_prop <- diag(prop.table(ensemble_tree_validation_table, margin = 1))
  ensemble_tree_validation_prop <- diag(prop.table(ensemble_tree_validation_table, margin = 1))
  ensemble_tree_validation_no[i] <- as.numeric(ensemble_tree_validation_prop[1])
  ensemble_tree_validation_no_mean <- mean(ensemble_tree_validation_no)
  ensemble_tree_validation_yes[i] <- as.numeric(ensemble_tree_validation_prop[2])
  ensemble_tree_validation_yes_mean <- mean(ensemble_tree_validation_yes)
  
  ensemble_tree_no[i] <- mean(c(ensemble_tree_test_no, ensemble_tree_validation_no))
  ensemble_tree_no_mean <- mean(ensemble_tree_no)
  
  ensemble_tree_yes[i] <- mean(c(ensemble_tree_test_yes, ensemble_tree_validation_yes))
  ensemble_tree_yes_mean <- mean(ensemble_tree_yes)
  
  ensemble_tree_holdout_mean <- mean(c(ensemble_tree_test_accuracy_mean, ensemble_tree_validation_accuracy_mean))
  ensemble_tree_overfitting[i] <- ensemble_tree_holdout_mean / ensemble_tree_train_accuracy_mean
  ensemble_tree_overfitting_mean <- mean(ensemble_tree_overfitting)
  ensemble_tree_overfitting_range <- range(ensemble_tree_overfitting)
  
  ensemble_tree_table <- ensemble_tree_test_table + ensemble_tree_validation_table
  
  ensemble_tree_table_sum_diag <- sum(diag(ensemble_tree_table))

}


#### Barchart of solution methods vs y ####
ensemble1 %>% 
  mutate(across(-y, as.numeric)) %>% 
  pivot_longer(-y, names_to = "var", values_to = "value") %>%
  ggplot(aes(x = y, y = value)) +
  geom_col() +
  facet_wrap(~var, scales = "free") +
  labs(title="Numerical values against type")


#### Summary of the dataset ####
summary(ensemble1)


#### Data dictionary of the ensemble ####
str(ensemble1)


#### Boxplot of ensemble of solution methods vs y ####
ensemble1 %>% 
  mutate(across(-y, as.numeric)) %>% 
  pivot_longer(-y, names_to = "var", values_to = "value") %>%
  ggplot(aes(x = y, y = value)) +
  geom_boxplot() +
  facet_wrap(~var, scales = "free") +
  labs(title="Ensemble boxplot of solution methods vs type")


results <- data_frame(
  Model = c('ADA Bag', 'ADA Boost', 'Bagging', 'Bagged Random Forest', 'C50', 'Earth', 'Flexible Discriminant Analysis',
    'Linear Discriminant Analysis', 'Linear Model', 'Mixed Discriminant Analysis', 'Naive Bayes', 'Quadratic Discriminant Analysis',
    'Partial Least Squares', 'Penalized Discriminant Analysis', 'Random Forest', 'Ranger', 'Regularized Discriminant Analysis',
    'Support Vector Machines', 'Trees', 'Ensemble ADA Bag', 'Ensemble ADA Boost', 'Ensemble Bagging',
    'Ensemble Bagged Random Forest', 'Ensemble C50', 'Ensemble Earth', 'Ensemble Flexible Discriminant Analysis',
    'Ensemble Linear', 'Ensemble Naive Bayes', 'Ensemble Ranger', 'Ensemble Random Forest',
    'Ensemble Regularized Discriminant Analysis', 'Ensemble RPart', 'Ensemble Support Vector Machines', 'Ensemble Trees'),
  
  'Mean_Train_Accuracy' = round(c(adabag_train_accuracy_mean, adaboost_train_accuracy_mean, bagging_train_accuracy_mean, 
    bag_rf_train_accuracy_mean, C50_train_accuracy_mean, earth_train_accuracy_mean, fda_train_accuracy_mean, lda_train_accuracy_mean,
    linear_train_accuracy_mean, mda_train_accuracy_mean, n_bayes_train_accuracy_mean, qda_train_accuracy_mean, pls_train_accuracy_mean,
    pda_train_accuracy_mean, rf_train_accuracy_mean, ranger_train_accuracy_mean, rda_train_accuracy_mean,
    svm_train_accuracy_mean, tree_train_accuracy_mean, ensemble_adabag_train_accuracy_mean, ensemble_adaboost_train_accuracy_mean,
    ensemble_bagging_train_accuracy_mean, ensemble_bag_rf_train_accuracy_mean, ensemble_C50_train_accuracy_mean,
    ensemble_earth_train_accuracy_mean, ensemble_fda_train_accuracy_mean, ensemble_linear_train_accuracy_mean,
    ensemble_n_bayes_train_accuracy_mean, ensemble_ranger_train_accuracy_mean, ensemble_rf_train_accuracy_mean,
    ensemble_rda_train_accuracy_mean, ensemble_rpart_train_accuracy_mean, ensemble_svm_train_accuracy_mean,
    ensemble_tree_train_accuracy_mean),4),
  
  'Mean_Test_Accuracy' = round(c(adabag_test_accuracy_mean, adaboost_test_accuracy_mean, bagging_test_accuracy_mean,
    bag_rf_test_accuracy_mean, C50_test_accuracy_mean, earth_test_accuracy_mean, fda_test_accuracy_mean, lda_test_accuracy_mean,
    linear_test_accuracy_mean, mda_test_accuracy_mean, n_bayes_test_accuracy_mean, qda_test_accuracy_mean, pls_test_accuracy_mean,
    pda_test_accuracy_mean, rf_test_accuracy_mean, ranger_test_accuracy_mean, rda_test_accuracy_mean,
    svm_test_accuracy_mean, tree_test_accuracy_mean, ensemble_adabag_test_accuracy_mean, ensemble_adaboost_test_accuracy_mean,
    ensemble_bagging_test_accuracy_mean, ensemble_bag_rf_test_accuracy_mean, ensemble_C50_test_accuracy_mean,
    ensemble_earth_test_accuracy_mean, ensemble_fda_test_accuracy_mean, ensemble_linear_test_accuracy_mean,
    ensemble_n_bayes_test_accuracy_mean, ensemble_ranger_test_accuracy_mean, ensemble_rf_test_accuracy_mean,
    ensemble_rda_test_accuracy_mean, ensemble_rpart_test_accuracy_mean, ensemble_svm_test_accuracy_mean,
    ensemble_tree_test_accuracy_mean),4),
  
  'Mean_Validation_Accuracy' = round(c(adabag_validation_accuracy_mean, adaboost_validation_accuracy_mean, bagging_validation_accuracy_mean,
    bag_rf_validation_accuracy_mean, C50_validation_accuracy_mean, earth_validation_accuracy_mean, fda_validation_accuracy_mean,
    lda_validation_accuracy_mean, linear_validation_accuracy_mean, mda_validation_accuracy_mean, n_bayes_validation_accuracy_mean,
    qda_validation_accuracy_mean, pls_validation_accuracy_mean, pda_validation_accuracy_mean, rf_validation_accuracy_mean,
    ranger_validation_accuracy_mean, rda_validation_accuracy_mean, svm_validation_accuracy_mean,
    tree_validation_accuracy_mean, ensemble_adabag_validation_accuracy_mean, ensemble_adaboost_validation_accuracy_mean,
    ensemble_bagging_validation_accuracy_mean, ensemble_bag_rf_validation_accuracy_mean, ensemble_C50_validation_accuracy_mean,
    ensemble_earth_validation_accuracy_mean, ensemble_fda_validation_accuracy_mean, ensemble_linear_validation_accuracy_mean,
    ensemble_n_bayes_validation_accuracy_mean, ensemble_ranger_validation_accuracy_mean, ensemble_rf_validation_accuracy_mean,
    ensemble_rda_validation_accuracy_mean, ensemble_rpart_validation_accuracy_mean, ensemble_svm_validation_accuracy_mean,
    ensemble_tree_validation_accuracy_mean),4),
  
  'Mean_Holdout' = round(c(adabag_holdout_mean, adaboost_holdout_mean, bagging_holdout_mean, bag_rf_holdout_mean, C50_holdout_mean, earth_holdout_mean,
    fda_holdout_mean, lda_holdout_mean, linear_holdout_mean, mda_holdout_mean, n_bayes_holdout_mean, qda_holdout_mean, pls_holdout_mean,
    pda_holdout_mean, rf_holdout_mean, ranger_holdout_mean, rda_holdout_mean, svm_holdout_mean, tree_holdout_mean,
    ensemble_adabag_holdout_mean, ensemble_adaboost_holdout_mean, ensemble_bagging_holdout_mean, ensemble_bag_rf_holdout_mean,
    ensemble_C50_holdout_mean, ensemble_earth_holdout_mean, ensemble_fda_holdout_mean, ensemble_linear_holdout_mean, ensemble_n_bayes_holdout_mean,
    ensemble_ranger_holdout_mean, ensemble_rf_holdout_mean, ensemble_rda_holdout_mean, ensemble_rpart_holdout_mean,
    ensemble_svm_holdout_mean, ensemble_tree_holdout_mean),4),
  
  'Mean_overfitting' = round(c(adabag_overfitting_mean, adaboost_overfitting_mean, bagging_overfitting_mean, bag_rf_overfitting_mean,
    C50_overfitting_mean, earth_overfitting_mean, fda_overfitting_mean, lda_overfitting_mean, linear_overfitting_mean, mda_overfitting_mean,
    n_bayes_overfitting_mean, qda_overfitting_mean, pls_overfitting_mean, pda_overfitting_mean, rf_overfitting_mean, ranger_overfitting_mean,
    rda_overfitting_mean, svm_overfitting_mean, tree_overfitting_mean, ensemble_adabag_overfitting_mean,
    ensemble_adaboost_overfitting_mean, ensemble_bagging_overfitting_mean, ensemble_bag_rf_overfitting_mean, ensemble_C50_overfitting_mean,
    ensemble_earth_overfitting_mean, ensemble_fda_overfitting_mean, ensemble_linear_overfitting_mean, ensemble_n_bayes_overfitting_mean,
    ensemble_ranger_overfitting_mean, ensemble_rf_overfitting_mean, ensemble_rda_overfitting_mean, ensemble_rpart_overfitting_mean,
    ensemble_svm_overfitting_mean, ensemble_tree_overfitting_mean),4),
  
  'Min_overfitting' = round(c(adabag_overfitting_range[1], adaboost_overfitting_range[1], bagging_overfitting_range[1], bag_rf_overfitting_range[1],
    C50_overfitting_range[1], earth_overfitting_range[1], fda_overfitting_range[1], lda_overfitting_range[1], linear_overfitting_range[1],
    mda_overfitting_range[1], n_bayes_overfitting_range[1], qda_overfitting_range[1], pls_overfitting_range[1], pda_overfitting_range[1],
    rf_overfitting_range[1], ranger_overfitting_range[1], rda_overfitting_range[1], svm_overfitting_range[1],
    tree_overfitting_range[1], ensemble_adabag_overfitting_range[1], ensemble_adaboost_overfitting_range[1],
    ensemble_bagging_overfitting_range[1], ensemble_bag_rf_overfitting_range[1], ensemble_C50_overfitting_range[1],
    ensemble_earth_overfitting_range[1], ensemble_fda_overfitting_range[1], ensemble_linear_overfitting_range[1],
    ensemble_n_bayes_overfitting_range[1], ensemble_ranger_overfitting_range[1], ensemble_rf_overfitting_range[1],
    ensemble_rda_overfitting_range[1], ensemble_rpart_overfitting_range[1], ensemble_svm_overfitting_range[1],
    ensemble_tree_overfitting_range[1]),4),
  
  'Max_overfitting' = round(c(adabag_overfitting_range[2], adaboost_overfitting_range[2], bagging_overfitting_range[2],
    bag_rf_overfitting_range[2], C50_overfitting_range[2], earth_overfitting_range[2], fda_overfitting_range[2], lda_overfitting_range[2],
    linear_overfitting_range[2], mda_overfitting_range[2], n_bayes_overfitting_range[2], qda_overfitting_range[2], pls_overfitting_range[2],
    pda_overfitting_range[2], rf_overfitting_range[2], ranger_overfitting_range[2], rda_overfitting_range[2],
    svm_overfitting_range[2], tree_overfitting_range[2], ensemble_adabag_overfitting_range[2], ensemble_adaboost_overfitting_range[2],
    ensemble_bagging_overfitting_range[2], ensemble_bag_rf_overfitting_range[2], ensemble_C50_overfitting_range[2], ensemble_earth_overfitting_range[2],
    ensemble_fda_overfitting_range[2], ensemble_linear_overfitting_range[2], ensemble_n_bayes_overfitting_range[2], ensemble_ranger_overfitting_range[2],
    ensemble_rf_overfitting_range[2], ensemble_rda_overfitting_range[2], ensemble_rpart_overfitting_range[2],
    ensemble_svm_overfitting_range[2], ensemble_tree_overfitting_range[2]),4),
  
  'Sum_of_Diagonal' = round(c(adabag_table_sum_diag, adaboost_table_sum_diag, bagging_table_sum_diag, bag_rf_table_sum_diag, C50_table_sum_diag,
    earth_table_sum_diag, fda_table_sum_diag, lda_table_sum_diag, linear_table_sum_diag, mda_table_sum_diag, n_bayes_table_sum_diag,
    qda_table_sum_diag, pls_table_sum_diag, pda_table_sum_diag, rf_table_sum_diag, ranger_table_sum_diag, rda_table_sum_diag,
    svm_table_sum_diag, tree_table_sum_diag, ensemble_adabag_table_sum_diag, ensemble_adaboost_table_sum_diag,
    ensemble_bagging_table_sum_diag, ensemble_bag_rf_table_sum_diag, ensemble_C50_table_sum_diag, ensemble_earth_table_sum_diag,
    ensemble_fda_table_sum_diag, ensemble_linear_table_sum_diag, ensemble_n_bayes_table_sum_diag, ensemble_ranger_table_sum_diag,
    ensemble_rf_table_sum_diag, ensemble_rda_table_sum_diag, ensemble_rpart_table_sum_diag, ensemble_svm_table_sum_diag, ensemble_tree_table_sum_diag),4),
  
  'Final_fit_train' = c('adabag_fit_train', 'adaboost_fit_train', 'bagging_fit_train', 'bag_rf_fit_train', 'C50_fit_train', 'earth_fit_train', 'fda_fit_train', 'lda_fit_train',
    'linear_fit_train', 'mda_fit_train', 'n_bayes_fit_train', 'qda_fit_train', 'pls_fit_train', 'pda_fit_train', 'rf_fit_train', 'ranger_fit_train', 'rda_fit_train',
    'svm_fit_train', 'tree_fit_train', 'ensemble_adabag_fit_train', 'ensemble_adaboost_fit_train', 'ensemble_bagging_fit_train', 'ensemble_bag_rf_fit_train',
    'ensemble_C50_fit_train', 'ensemble_earth_fit_train', 'ensemble_fda_fit_train', 'ensemble_linear_fit_train', 'ensemble_n_bayes_fit_train', 'ensemble_ranger_fit_train',
    'ensemble_rf_fit_train', 'ensemble_rda_fit_train', 'ensemble_rpart_fit_train', 'ensemble_svm_fit_train', 'ensemble_tree_fit_train'),
  
  'Summary_Table' = c('adabag_table', 'adaboost_table', 'bagging_table', 'bag_rf_table', 'C50_table', 'earth_table', 'fda_table', 'lda_table',
    'linear_table', 'mda_table', 'n_bayes_table', 'qda_table', 'pls_table', 'pda_table', 'rf_table', 'ranger_table', 'rda_table', 
    'svm_table', 'tree_table', 'ensemble_adabag_table', 'ensemble_adaboost_table', 'ensemble_bagging_table', 'ensemble_bag_rf_table',
    'ensemble_C50_table', 'ensemble_earth_table', 'ensemble_fda_table', 'ensemble_linear_table', 'ensemble_n_bayes_table', 'ensemble_ranger_table',
    'ensemble_rf_table', 'ensemble_rda_table', 'ensemble_rpart_table', 'ensemble_svm_table', 'ensemble_tree_table')
)

results <- results %>% arrange(desc(Mean_Holdout))
reactable(results, searchable = TRUE, pagination = FALSE, wrap = TRUE, fullWidth = TRUE, filterable = TRUE, bordered = TRUE, striped = TRUE,
    highlight = TRUE, rownames = TRUE)

all_tables <- list(adabag_table, adaboost_table, bagging_table, bag_rf_table, C50_table, earth_table, fda_table, lda_table, linear_table,
                   mda_table, n_bayes_table, qda_table, pls_table, pda_table, rf_table, ranger_table, rda_table, svm_table, tree_table,
                   ensemble_adabag_table, ensemble_adaboost_table, ensemble_bagging_table, ensemble_bag_rf_table, ensemble_C50_table, ensemble_earth_table,
                   ensemble_fda_table, ensemble_linear_table, ensemble_n_bayes_table, ensemble_ranger_table, ensemble_rf_table,
                   ensemble_rda_table, ensemble_rpart_table, ensemble_svm_table, ensemble_tree_table)

all_tables
