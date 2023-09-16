# 1. Describe the problem, data, and goal(s). Literature review if appropriate.
# 2. Load all packages at once in alphabetical order (lines 11 - 22)
# 3. Input all the data (lines 25 - 37)
# 4. Full exploratory data analysis (lines 40 - 138)
# 5. Initialize values to 0
# 6. Set up loops for data resampling, Data splitting train (60%) test (20%) and validation (20%) (lines 141-144)
# 7. Fit on training data, predict on test and validation, calculate RMSE and overfitting for all 20 models and 18 ensembles (lines 146 - 202)
# 8. Summary data visualizations
# 9. Summary results
# 10. Strongest evidence based recommendations, suggestions for future research

#### <------------------- 1. Describe the problem, data, and goal(s). Literature review if appropriate. ----------------------------->####



#### <----------------------------------- 2. Load all packages at once in alphabetical order ---------------------------------------->####

library(arm) # For Bayes Generalized Linear Models
library(brnn) # For Bayes RNN
library(corrplot) # to plot the correlation matrix
library(Cubist) # for cubist analysis
library(earth) # for 'earth' analysis - Multivariate Analysis of Regression Splines (MARS)
library(e1071) # used for tuning multiple models
library(gam) # for smoothing splines analysis
library(gbm) # for gradient boosted models
library(glmnet) # for Ridge, Elastic and Lasso regression
library(gridExtra) # for plotting the visualizations at the end of the report
library(ipred) # for improved predictors, used for Bagging
library(kableExtra) # for printing to the Viewer window
library(leaps) # to do dimension reduction of the ensemble data set
library(MASS) # for the Boston data set
library(Metrics) # to quickly calculate Root Mean Squared Error (RMSE)
library(PerformanceAnalytics) # For the correlation chart
library(pls) # for Partial Least Squares and Principle Components Analysis
library(randomForest) # for regular and boosted random forest models
library(reactable) # to create the final table
library(reactablefmtr) # to add title and other aspects of the finished reports
library(robust) # for robust models
library(rpart) # for rpart (also known as cart)
library(tidyverse) # My favorite way to do data science :)
library(tree) # to do regression trees
library(xgboost) # doing XGBoost analysis


#### <------------------------------------------------ 3. Input all the data -------------------------------------------------------->####

df <- MASS::Boston # we will use the first row as new data to test predictions
df <- df %>% relocate(medv, .after = last_col()) # Moves the target column to the last column on the right


## Set baseline RMSE and Standard Deviation (SD) based on the full data set
actual_RMSE <- Metrics::rmse(actual = df$medv, predicted = df$medv)
actual_mean <- round(mean(df$medv), 4)
actual_sd <- round(sd(df$medv), 4)


##### <------------------------------------------- 4. Full exploratory data analysis ------------------------------------------------>####

## Correlation data and plots ##
df1 <- df %>% purrr::keep(is.numeric)
M1 = cor(df1)
title = "Correlation plot of the numerical data"
corrplot::corrplot(M1, method = 'number', title = title, mar=c(0, 0, 1, 0)) # http://stackoverflow.com/a/14754408/54964)
corrplot::corrplot(M1, method = 'circle', title = title, mar=c(0, 0, 1, 0)) # http://stackoverflow.com/a/14754408/54964)


## Summary of the numeric data ##
knitr::kable(summary(df1), caption = 'Summary of the data') %>% kableExtra::kable_material()


## Data dictionary ##
summary(df)


## Print correlation table of the data ##
knitr::kable((M1), caption = 'Correlation Matrix') %>% kableExtra::kable_material()


# Pairwise scatterplot
pairs(df, main = "Matrix of scatterplots of the numerical data")


## Boxplots of the numeric data ##
df1 %>%
  gather(key = "var", value = "value") %>%
  ggplot(aes(x = '', y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~ var, scales = "free") +
  theme_bw() +
  labs(title = "Boxplots of the numeric data")
# Thanks to https://rstudio-pubs-static.s3.amazonaws.com/388596_e21196f1adf04e0ea7cd68edd9eba966.html


## Correlation charts and numbers for the numeric data ##
PerformanceAnalytics::chart.Correlation(df1)
mtext("Correlation chart from Performance Analytics package for the numerical data", side=3, line=3)
# https://community.rstudio.com/t/add-title-to-correlation-chart/63354


## Create histograms of the numerical columns ##
df1 <- df %>% dplyr::select_if(is.numeric)
ggplot(gather(df1, cols, value), aes(x = value)) +
  geom_histogram(bins = round(nrow(df1)/10)) + 
  facet_wrap(.~cols, scales = "free") +
  labs(title = "Histograms of each numeric column. Each bar = 10 rows of data")


#### <----------------------------------------------------- 5. Initialize values to 0 ----------------------------------------------->####

bag_rf_duration <- 0
bag_rf_end <- 0
bag_rf_overfitting <- 0
bag_rf_overfitting_mean <- 0
bag_rf_sd <- 0
bag_rf_start <- 0
bag_rf_test_predict_value <- 0
bag_rf_test_RMSE <- 0
bag_rf_test_RMSE_mean <- 0
bag_rf_train_predict_value <- 0
bag_rf_train_predict_value <- 0
bag_rf_train_RMSE <- 0
bag_rf_train_RMSE_mean <- 0
bag_rf_validation_predict_value <- 0
bag_rf_validation_RMSE <- 0
bag_rf_validation_RMSE_mean <- 0
y_hat_bag_rf <- 0

bagging_duration <- 0
bagging_end <- 0
bagging_overfitting <- 0
bagging_overfitting_mean <- 0
bagging_sd <- 0
bagging_start <- 0
bagging_test_predict_value <- 0
bagging_test_RMSE <- 0
bagging_test_RMSE_mean <- 0
bagging_train_predict_value <- 0
bagging_train_predict_value <- 0
bagging_train_RMSE <- 0
bagging_train_RMSE_mean <- 0
bagging_validation_predict_value <- 0
bagging_validation_RMSE <- 0
bagging_validation_RMSE_mean <- 0
y_hat_bagging <- 0

bayesglm_duration <- 0
bayesglm_end <- 0
bayesglm_overfitting <- 0
bayesglm_overfitting_mean <- 0
bayesglm_sd <- 0
bayesglm_start <- 0
bayesglm_test_predict_value <- 0
bayesglm_test_RMSE <- 0
bayesglm_test_RMSE_mean <- 0
bayesglm_train_predict_value <- 0
bayesglm_train_predict_value <- 0
bayesglm_train_RMSE <- 0
bayesglm_train_RMSE_mean <- 0
bayesglm_validation_predict_value <- 0
bayesglm_validation_RMSE <- 0
bayesglm_validation_RMSE_mean <- 0
y_hat_bayesglm <- 0

bayesrnn_duration <- 0
bayesrnn_end <- 0
bayesrnn_overfitting <- 0
bayesrnn_overfitting_mean <- 0
bayesrnn_sd <- 0
bayesrnn_start <- 0
bayesrnn_test_predict_value <- 0
bayesrnn_test_RMSE <- 0
bayesrnn_test_RMSE_mean <- 0
bayesrnn_train_predict_value <- 0
bayesrnn_train_predict_value <- 0
bayesrnn_train_RMSE <- 0
bayesrnn_train_RMSE_mean <- 0
bayesrnn_validation_predict_value <- 0
bayesrnn_validation_RMSE <- 0
bayesrnn_validation_RMSE_mean <- 0
y_hat_bayesrnn <- 0

boost_rf_duration <- 0
boost_rf_end <- 0
boost_rf_overfitting <- 0
boost_rf_overfitting_mean <- 0
boost_rf_sd <- 0
boost_rf_start <- 0
boost_rf_test_predict_value <- 0
boost_rf_test_RMSE <- 0
boost_rf_test_RMSE_mean <- 0
boost_rf_train_predict_value <- 0
boost_rf_train_predict_value <- 0
boost_rf_train_RMSE <- 0
boost_rf_train_RMSE_mean <- 0
boost_rf_validation_predict_value <- 0
boost_rf_validation_RMSE <- 0
boost_rf_validation_RMSE_mean <- 0
y_hat_boost_rf <- 0

cubist_duration <- 0
cubist_end <- 0
cubist_overfitting <- 0
cubist_overfitting_mean <- 0
cubist_sd <- 0
cubist_start <- 0
cubist_test_predict_value <- 0
cubist_test_RMSE <- 0
cubist_test_RMSE_mean <- 0
cubist_train_predict_value <- 0
cubist_train_predict_value <- 0
cubist_train_RMSE <- 0
cubist_train_RMSE_mean <- 0
cubist_validation_predict_value <- 0
cubist_validation_RMSE <- 0
cubist_validation_RMSE_mean <- 0
y_hat_cubist <- 0

earth_duration <- 0
earth_end <- 0
earth_overfitting <- 0
earth_overfitting_mean <- 0
earth_sd <- 0
earth_start <- 0
earth_test_predict_value <- 0
earth_test_RMSE <- 0
earth_test_RMSE_mean <- 0
earth_train_predict_value <- 0
earth_train_predict_value <- 0
earth_train_RMSE <- 0
earth_train_RMSE_mean <- 0
earth_validation_predict_value <- 0
earth_validation_RMSE <- 0
earth_validation_RMSE_mean <- 0
y_hat_earth <- 0

gam_duration <- 0
gam_end <- 0
gam_overfitting <- 0
gam_overfitting_mean <- 0
gam_sd <- 0
gam_start <- 0
gam_test_predict_value <- 0
gam_test_RMSE <- 0
gam_test_RMSE_mean <- 0
gam_train_predict_value <- 0
gam_train_predict_value <- 0
gam_train_RMSE <- 0
gam_train_RMSE_mean <- 0
gam_validation_predict_value <- 0
gam_validation_RMSE <- 0
gam_validation_RMSE_mean <- 0
y_hat_gam <- 0

gb_duration <- 0
gb_end <- 0
gb_overfitting <- 0
gb_overfitting_mean <- 0
gb_sd <- 0
gb_start <- 0
gb_test_predict_value <- 0
gb_test_RMSE <- 0
gb_test_RMSE_mean <- 0
gb_train_predict_value <- 0
gb_train_predict_value <- 0
gb_train_RMSE <- 0
gb_train_RMSE_mean <- 0
gb_validation_predict_value <- 0
gb_validation_RMSE <- 0
gb_validation_RMSE_mean <- 0
y_hat_gb <- 0

knn_duration <- 0
knn_end <- 0
knn_overfitting <- 0
knn_overfitting_mean <- 0
knn_sd <- 0
knn_start <- 0
knn_test_predict_value <- 0
knn_test_RMSE <- 0
knn_test_RMSE_mean <- 0
knn_train_predict_value <- 0
knn_train_predict_value <- 0
knn_train_RMSE <- 0
knn_train_RMSE_mean <- 0
knn_validation_predict_value <- 0
knn_validation_RMSE <- 0
knn_validation_RMSE_mean <- 0
y_hat_knn <- 0

linear_duration <- 0
linear_end <- 0
linear_overfitting <- 0
linear_overfitting_mean <- 0
linear_sd <- 0
linear_start <- 0
linear_test_predict_value <- 0
linear_test_RMSE <- 0
linear_test_RMSE_mean <- 0
linear_train_predict_value <- 0
linear_train_predict_value <- 0
linear_train_RMSE <- 0
linear_train_RMSE_mean <- 0
linear_validation_predict_value <- 0
linear_validation_RMSE <- 0
linear_validation_RMSE_mean <- 0
y_hat_linear <- 0

lqs_duration <- 0
lqs_end <- 0
lqs_overfitting <- 0
lqs_overfitting_mean <- 0
lqs_sd <- 0
lqs_start <- 0
lqs_test_predict_value <- 0
lqs_test_RMSE <- 0
lqs_test_RMSE_mean <- 0
lqs_train_predict_value <- 0
lqs_train_predict_value <- 0
lqs_train_RMSE <- 0
lqs_train_RMSE_mean <- 0
lqs_validation_predict_value <- 0
lqs_validation_RMSE <- 0
lqs_validation_RMSE_mean <- 0
y_hat_lqs <- 0

pls_duration <- 0
pls_end <- 0
pls_overfitting <- 0
pls_overfitting_mean <- 0
pls_sd <- 0
pls_start <- 0
pls_test_predict_value <- 0
pls_test_RMSE <- 0
pls_test_RMSE_mean <- 0
pls_train_predict_value <- 0
pls_train_predict_value <- 0
pls_train_RMSE <- 0
pls_train_RMSE_mean <- 0
pls_validation_predict_value <- 0
pls_validation_RMSE <- 0
pls_validation_RMSE_mean <- 0
y_hat_pls <- 0

pcr_duration <- 0
pcr_end <- 0
pcr_overfitting <- 0
pcr_overfitting_mean <- 0
pcr_sd <- 0
pcr_start <- 0
pcr_test_predict_value <- 0
pcr_test_RMSE <- 0
pcr_test_RMSE_mean <- 0
pcr_train_predict_value <- 0
pcr_train_predict_value <- 0
pcr_train_RMSE <- 0
pcr_train_RMSE_mean <- 0
pcr_validation_predict_value <- 0
pcr_validation_RMSE <- 0
pcr_validation_RMSE_mean <- 0
y_hat_pcr <- 0

rf_duration <- 0
rf_end <- 0
rf_overfitting <- 0
rf_overfitting_mean <- 0
rf_sd <- 0
rf_start <- 0
rf_test_predict_value <- 0
rf_test_RMSE <- 0
rf_test_RMSE_mean <- 0
rf_train_predict_value <- 0
rf_train_predict_value <- 0
rf_train_RMSE <- 0
rf_train_RMSE_mean <- 0
rf_validation_predict_value <- 0
rf_validation_RMSE <- 0
rf_validation_RMSE_mean <- 0
y_hat_rf <- 0

robust_duration <- 0
robust_end <- 0
robust_overfitting <- 0
robust_overfitting_mean <- 0
robust_sd <- 0
robust_start <- 0
robust_test_predict_value <- 0
robust_test_RMSE <- 0
robust_test_RMSE_mean <- 0
robust_train_predict_value <- 0
robust_train_predict_value <- 0
robust_train_RMSE <- 0
robust_train_RMSE_mean <- 0
robust_validation_predict_value <- 0
robust_validation_RMSE <- 0
robust_validation_RMSE_mean <- 0
y_hat_robust <- 0

rpart_duration <- 0
rpart_end <- 0
rpart_overfitting <- 0
rpart_overfitting_mean <- 0
rpart_sd <- 0
rpart_start <- 0
rpart_test_predict_value <- 0
rpart_test_RMSE <- 0
rpart_test_RMSE_mean <- 0
rpart_train_predict_value <- 0
rpart_train_predict_value <- 0
rpart_train_RMSE <- 0
rpart_train_RMSE_mean <- 0
rpart_validation_predict_value <- 0
rpart_validation_RMSE <- 0
rpart_validation_RMSE_mean <- 0
y_hat_rpart <- 0

svm_duration <- 0
svm_end <- 0
svm_overfitting <- 0
svm_overfitting_mean <- 0
svm_sd <- 0
svm_start <- 0
svm_test_predict_value <- 0
svm_test_RMSE <- 0
svm_test_RMSE_mean <- 0
svm_train_predict_value <- 0
svm_train_predict_value <- 0
svm_train_RMSE <- 0
svm_train_RMSE_mean <- 0
svm_validation_predict_value <- 0
svm_validation_RMSE <- 0
svm_validation_RMSE_mean <- 0
y_hat_svm <- 0

tree_duration <- 0
tree_end <- 0
tree_overfitting <- 0
tree_overfitting_mean <- 0
tree_sd <- 0
tree_start <- 0
tree_test_predict_value <- 0
tree_test_RMSE <- 0
tree_test_RMSE_mean <- 0
tree_train_predict_value <- 0
tree_train_predict_value <- 0
tree_train_RMSE <- 0
tree_train_RMSE_mean <- 0
tree_validation_predict_value <- 0
tree_validation_RMSE <- 0
tree_validation_RMSE_mean <- 0
y_hat_tree <- 0

xgb_duration <- 0
xgb_end <- 0
xgb_holdout_RMSE_mean <- 0
xgb_overfitting <- 0
xgb_overfitting_mean <- 0
xgb_sd <- 0
xgb_start <- 0
xgb_test_RMSE <- 0
xgb_test_RMSE_mean <- 0
xgb_train_RMSE <- 0
xgb_train_RMSE_mean <- 0
xgb_validation_RMSE <- 0
xgb_validation_RMSE_mean <- 0
y_hat_xgb <- 0


## Initialize Ensemble Variables to 0 ##
ensemble_bag_rf_duration <- 0
ensemble_bag_rf_end <- 0
ensemble_bag_rf_overfitting <- 0
ensemble_bag_rf_overfitting_mean <- 0
ensemble_bag_rf_sd <- 0
ensemble_bag_rf_start <- 0
ensemble_bag_rf_test_predict_value <- 0
ensemble_bag_rf_test_RMSE <- 0
ensemble_bag_rf_test_RMSE_mean <- 0
ensemble_bag_rf_train_RMSE <- 0
ensemble_bag_rf_train_RMSE_mean <- 0
ensemble_bag_rf_validation_RMSE <- 0
ensemble_bag_rf_validation_RMSE_mean <- 0
ensemble_y_hat_bag_rf <- 0

ensemble_bagging_duration <- 0
ensemble_bagging_end <- 0
ensemble_bagging_overfitting <- 0
ensemble_bagging_overfitting_mean <- 0
ensemble_bagging_sd <- 0
ensemble_bagging_start <- 0
ensemble_bagging_test_predict_value <- 0
ensemble_bagging_test_RMSE <- 0
ensemble_bagging_test_RMSE_mean <- 0
ensemble_bagging_train_RMSE <- 0
ensemble_bagging_train_RMSE_mean <- 0
ensemble_bagging_validation_RMSE <- 0
ensemble_bagging_validation_RMSE_mean <- 0
ensemble_y_hat_bagging <- 0

ensemble_bayesglm_duration <- 0
ensemble_bayesglm_end <- 0
ensemble_bayesglm_overfitting <- 0
ensemble_bayesglm_overfitting_mean <- 0
ensemble_bayesglm_sd <- 0
ensemble_bayesglm_start <- 0
ensemble_bayesglm_test_predict_value <- 0
ensemble_bayesglm_test_RMSE <- 0
ensemble_bayesglm_test_RMSE_mean <- 0
ensemble_bayesglm_train_RMSE <- 0
ensemble_bayesglm_train_RMSE_mean <- 0
ensemble_bayesglm_validation_RMSE <- 0
ensemble_bayesglm_validation_RMSE_mean <- 0
ensemble_y_hat_bag_rf <- 0

ensemble_bayesrnn_duration <- 0
ensemble_bayesrnn_end <- 0
ensemble_bayesrnn_overfitting <- 0
ensemble_bayesrnn_overfitting_mean <- 0
ensemble_bayesrnn_sd <- 0
ensemble_bayesrnn_start <- 0
ensemble_bayesrnn_test_predict_value <- 0
ensemble_bayesrnn_test_RMSE <- 0
ensemble_bayesrnn_test_RMSE_mean <- 0
ensemble_bayesrnn_train_RMSE <- 0
ensemble_bayesrnn_train_RMSE_mean <- 0
ensemble_bayesrnn_validation_RMSE <- 0
ensemble_bayesrnn_validation_RMSE_mean <- 0
ensemble_y_hat_bayesrnn <- 0

ensemble_boost_rf_duration <- 0
ensemble_boost_rf_end <- 0
ensemble_boost_rf_overfitting <- 0
ensemble_boost_rf_overfitting_mean <- 0
ensemble_boost_rf_sd <- 0
ensemble_boost_rf_start <- 0
ensemble_boost_rf_test_predict_value <- 0
ensemble_boost_rf_test_RMSE <- 0
ensemble_boost_rf_test_RMSE_mean <- 0
ensemble_boost_rf_train_RMSE <- 0
ensemble_boost_rf_train_RMSE_mean <- 0
ensemble_boost_rf_validation_RMSE <- 0
ensemble_boost_rf_validation_RMSE_mean <- 0
ensemble_y_hat_boost_rf <- 0

ensemble_cubist_duration <- 0
ensemble_cubist_end <- 0
ensemble_cubist_overfitting <- 0
ensemble_cubist_overfitting_mean <- 0
ensemble_cubist_sd <- 0
ensemble_cubist_start <- 0
ensemble_cubist_test_predict_value <- 0
ensemble_cubist_test_RMSE <- 0
ensemble_cubist_test_RMSE_mean <- 0
ensemble_cubist_train_RMSE <- 0
ensemble_cubist_train_RMSE_mean <- 0
ensemble_cubist_validation_RMSE <- 0
ensemble_cubist_validation_RMSE_mean <- 0
ensemble_y_hat_cubist <- 0

ensemble_earth_duration <- 0
ensemble_earth_end <- 0
ensemble_earth_overfitting <- 0
ensemble_earth_overfitting_mean <- 0
ensemble_earth_sd <- 0
ensemble_earth_start <- 0
ensemble_earth_test_predict_value <- 0
ensemble_earth_test_RMSE <- 0
ensemble_earth_test_RMSE_mean <- 0
ensemble_earth_train_RMSE <- 0
ensemble_earth_train_RMSE_mean <- 0
ensemble_earth_validation_RMSE <- 0
ensemble_earth_validation_RMSE_mean <- 0
ensemble_y_hat_cubist <- 0

ensemble_gam_duration <- 0
ensemble_gam_end <- 0
ensemble_gam_overfitting <- 0
ensemble_gam_overfitting_mean <- 0
ensemble_gam_sd <- 0
ensemble_gam_start <- 0
ensemble_gam_test_predict_value <- 0
ensemble_gam_test_RMSE <- 0
ensemble_gam_test_RMSE_mean <- 0
ensemble_gam_train_RMSE <- 0
ensemble_gam_train_RMSE_mean <- 0
ensemble_gam_validation_RMSE <- 0
ensemble_gam_validation_RMSE_mean <- 0
ensemble_y_hat_gam <- 0

ensemble_gb_duration <- 0
ensemble_gb_end <- 0
ensemble_gb_overfitting <- 0
ensemble_gb_overfitting_mean <- 0
ensemble_gb_sd <- 0
ensemble_gb_start <- 0
ensemble_gb_test_predict_value <- 0
ensemble_gb_test_RMSE <- 0
ensemble_gb_test_RMSE_mean <- 0
ensemble_gb_train_RMSE <- 0
ensemble_gb_train_RMSE_mean <- 0
ensemble_gb_validation_RMSE <- 0
ensemble_gb_validation_RMSE_mean <- 0
ensemble_y_hat_gam <- 0

ensemble_knn_duration <- 0
ensemble_knn_end <- 0
ensemble_knn_overfitting <- 0
ensemble_knn_overfitting_mean <- 0
ensemble_knn_sd <- 0
ensemble_knn_start <- 0
ensemble_knn_test_predict_value <- 0
ensemble_knn_test_RMSE <- 0
ensemble_knn_test_RMSE_mean <- 0
ensemble_knn_train_RMSE <- 0
ensemble_knn_train_RMSE_mean <- 0
ensemble_knn_validation_RMSE <- 0
ensemble_knn_validation_RMSE_mean <- 0
ensemble_y_hat_knn <- 0

ensemble_linear_duration <- 0
ensemble_linear_end <- 0
ensemble_linear_overfitting <- 0
ensemble_linear_overfitting_mean <- 0
ensemble_linear_sd <- 0
ensemble_linear_start <- 0
ensemble_linear_test_predict_value <- 0
ensemble_linear_test_RMSE <- 0
ensemble_linear_test_RMSE_mean <- 0
ensemble_linear_train_RMSE <- 0
ensemble_linear_train_RMSE_mean <- 0
ensemble_linear_validation_RMSE <- 0
ensemble_linear_validation_RMSE_mean <- 0
ensemble_y_hat_linear <- 0

ensemble_pls_duration <- 0
ensemble_pls_end <- 0
ensemble_pls_overfitting <- 0
ensemble_pls_overfitting_mean <- 0
ensemble_pls_sd <- 0
ensemble_pls_start <- 0
ensemble_pls_test_predict_value <- 0
ensemble_pls_test_RMSE <- 0
ensemble_pls_test_RMSE_mean <- 0
ensemble_pls_train_RMSE <- 0
ensemble_pls_train_RMSE_mean <- 0
ensemble_pls_validation_RMSE <- 0
ensemble_pls_validation_RMSE_mean <- 0
ensemble_y_hat_pls <- 0

ensemble_pcr_duration <- 0
ensemble_pcr_end <- 0
ensemble_pcr_overfitting <- 0
ensemble_pcr_overfitting_mean <- 0
ensemble_pcr_start <- 0
ensemble_pcr_test_predict_value <- 0
ensemble_pcr_test_RMSE <- 0
ensemble_pcr_test_RMSE_mean <- 0
ensemble_pcr_train_RMSE <- 0
ensemble_pcr_train_RMSE_mean <- 0
ensemble_pcr_validation_RMSE <- 0
ensemble_pcr_validation_RMSE_mean <- 0
ensemble_y_hat_pcr <- 0
ensemble_y_hat_pls <- 0

ensemble_rpart_duration <- 0
ensemble_rpart_end <- 0
ensemble_rpart_overfitting <- 0
ensemble_rpart_overfitting_mean <- 0
ensemble_rpart_sd <- 0
ensemble_rpart_start <- 0
ensemble_rpart_test_predict_value <- 0
ensemble_rpart_test_RMSE <- 0
ensemble_rpart_test_RMSE_mean <- 0
ensemble_rpart_train_RMSE <- 0
ensemble_rpart_train_RMSE_mean <- 0
ensemble_rpart_validation_RMSE <- 0
ensemble_rpart_validation_RMSE_mean <- 0
ensemble_y_hat_rpart <- 0

ensemble_rf_duration <- 0
ensemble_rf_end <- 0
ensemble_rf_overfitting <- 0
ensemble_rf_overfitting_mean <- 0
ensemble_rf_start <- 0
ensemble_rf_test_predict_value <- 0
ensemble_rf_test_RMSE <- 0
ensemble_rf_test_RMSE_mean <- 0
ensemble_rf_train_RMSE <- 0
ensemble_rf_train_RMSE_mean <- 0
ensemble_rf_validation_RMSE <- 0
ensemble_rf_validation_RMSE_mean <- 0
ensemble_y_hat_rf <- 0
ensemble_y_hat_rpart <- 0

ensemble_svm_duration <- 0
ensemble_svm_end <- 0
ensemble_svm_overfitting <- 0
ensemble_svm_start <- 0
ensemble_svm_test_predict_value <- 0
ensemble_svm_test_RMSE <- 0
ensemble_svm_test_RMSE_mean <- 0
ensemble_svm_train_RMSE <- 0
ensemble_svm_train_RMSE_mean <- 0
ensemble_svm_validation_RMSE <- 0
ensemble_svm_validation_RMSE_mean <- 0
ensemble_y_hat_rf <- 0
ensemble_y_hat_rpart <- 0
ensemble_y_hat_svm <- 0

ensemble_tree_duration <- 0
ensemble_tree_end <- 0
ensemble_tree_overfitting <- 0
ensemble_tree_overfitting_mean <- 0
ensemble_tree_start <- 0
ensemble_tree_test_predict_value <- 0
ensemble_tree_test_RMSE <- 0
ensemble_tree_test_RMSE_mean <- 0
ensemble_tree_train_RMSE <- 0
ensemble_tree_train_RMSE_mean <- 0
ensemble_tree_validation_RMSE <- 0
ensemble_tree_validation_RMSE_mean <- 0
ensemble_y_hat_tree <- 0

ensemble_xgb_duration <- 0
ensemble_xgb_end <- 0
ensemble_xgb_overfitting <- 0
ensemble_xgb_overfitting_mean <- 0
ensemble_xgb_sd <- 0
ensemble_xgb_start <- 0
ensemble_xgb_test_RMSE <- 0
ensemble_xgb_test_RMSE_mean <- 0
ensemble_xgb_train_RMSE <- 0
ensemble_xgb_train_RMSE_mean <- 0
ensemble_xgb_validation_RMSE <- 0
ensemble_xgb_validation_RMSE_mean <- 0
ensemble_y_hat_xgb <- 0


#### <----- 6. Set up loops for data resampling, Data splitting train (60%) test (20%) and validation (20%) (lines 141-144) --------->####

for (i in 1:5) {

idx <- sample(seq(1, 3), size = nrow(df), replace = TRUE, prob = c(.6, .2, .2))
train <- df[idx == 1,]
test <- df[idx == 2,]
validation <- df[idx == 3,]


#### <-- 7. Fit on training data, predict on test and validation, calculate RMSE and overfitting for all 20 models and 18 ensembles-->####


####  Model #1 Bagged Random Forest tuned ####
bag_rf_start <- Sys.time()
bag_rf_train_fit <- e1071::tune.randomForest(x = train, y = train$medv, mtry = ncol(train)-1)
bag_rf_train_RMSE_mean <- Metrics::rmse(actual = train$medv, predicted = predict(object = bag_rf_train_fit$best.model,
  newdata = train))
bag_rf_test_RMSE_mean <- Metrics::rmse(actual = test$medv, predicted = predict(object = bag_rf_train_fit$best.model,
  newdata = test))
bag_rf_validation_RMSE_mean <- Metrics::rmse(actual = validation$medv, predicted = predict(object = bag_rf_train_fit$best.model,
  newdata = validation))
bag_rf_holdout_RMSE_mean <- mean(c(bag_rf_test_RMSE_mean, bag_rf_validation_RMSE_mean))
bag_rf_holdout_RMSE_sd_mean <- sd(c(bag_rf_test_RMSE_mean, bag_rf_validation_RMSE_mean))
bag_rf_train_predict_value <- as.numeric(predict(object = bag_rf_train_fit$best.model, newdata = train))
bag_rf_test_predict_value <- as.numeric(predict(object = bag_rf_train_fit$best.model, newdata = test))
bag_rf_validation_predict_value <- as.numeric(predict(object = bag_rf_train_fit$best.model, newdata = validation))
bag_rf_predict_value_mean <- mean(c(bag_rf_test_predict_value, bag_rf_validation_predict_value))
bag_rf_sd[i] <- sd(c(bag_rf_test_predict_value, bag_rf_validation_predict_value))
bag_rf_sd_mean <- mean(bag_rf_sd)
bag_rf_overfitting[i] <- bag_rf_holdout_RMSE_mean/bag_rf_train_RMSE_mean
bag_rf_overfitting_mean <- mean(bag_rf_overfitting)
bag_rf_overfitting_range <- range(bag_rf_overfitting)
y_hat_bag_rf <- c(bag_rf_test_predict_value, bag_rf_validation_predict_value)
bag_rf_end <- Sys.time()
bag_rf_duration <- bag_rf_end - bag_rf_start
bag_rf_duration_mean <- mean(bag_rf_duration)


####  Model #2 Bagging ####
bagging_start <- Sys.time()
bagging_train_fit <- ipred::bagging(formula = medv ~ ., data = train)
bagging_train_RMSE_mean <- Metrics::rmse(actual = train$medv, predicted = predict(object = bagging_train_fit, newdata = train))
bagging_test_RMSE_mean <- Metrics::rmse(actual = test$medv, predicted = predict(object = bagging_train_fit, newdata = test))
bagging_validation_RMSE_mean <- Metrics::rmse(actual = validation$medv, predicted = predict(object = bagging_train_fit, newdata = validation))
bagging_holdout_RMSE_mean <- mean(c(bagging_test_RMSE_mean, bagging_validation_RMSE_mean))
bagging_holdout_RMSE_sd_mean <- sd(c(bagging_test_RMSE_mean, bagging_validation_RMSE_mean))
bagging_train_predict_value <- as.numeric(predict(object = bagging_train_fit, newdata = train))
bagging_test_predict_value <- as.numeric(predict(object = bagging_train_fit, newdata = test))
bagging_validation_predict_value <- as.numeric(predict(object = bagging_train_fit, newdata = validation))
bagging_predict_value_mean <- mean(c(bagging_test_predict_value, bagging_validation_predict_value))
bagging_sd[i] <- sd(c(bagging_test_predict_value, bagging_validation_predict_value))
bagging_sd_mean <- mean(bagging_sd)
bagging_overfitting[i] <- bagging_holdout_RMSE_mean/bagging_train_RMSE_mean
bagging_overfitting_mean <- mean(bagging_overfitting)
bagging_overfitting_range <- range(bagging_overfitting)
y_hat_bagging <- c(bagging_test_predict_value, bagging_validation_predict_value)
bagging_end <- Sys.time()
bagging_duration <- bagging_end - bagging_start
bagging_duration_mean <- mean(bagging_duration)


####  Model #3 Bayes Generalized Linear Model (GLM) ####
bayesglm_start <- Sys.time()
bayesglm_train_fit <- arm::bayesglm(medv ~ ., data = train, family = gaussian(link = "identity"))
bayesglm_train_RMSE_mean <- Metrics::rmse(actual = train$medv, predicted = predict(object = bayesglm_train_fit, newdata = train))
bayesglm_test_RMSE_mean <- Metrics::rmse(actual = test$medv, predicted = predict(object = bayesglm_train_fit, newdata = test))
bayesglm_validation_RMSE_mean <- Metrics::rmse(actual = validation$medv, predicted = predict(object = bayesglm_train_fit, newdata = validation))
bayesglm_holdout_RMSE_mean <- mean(c(bayesglm_test_RMSE_mean, bayesglm_validation_RMSE_mean))
bayesglm_holdout_RMSE_sd_mean <- sd(c(bayesglm_test_RMSE_mean, bayesglm_validation_RMSE_mean))
bayesglm_train_predict_value <- as.numeric(predict(object = bayesglm_train_fit, newdata = train))
bayesglm_test_predict_value <- as.numeric(predict(object = bayesglm_train_fit, newdata = test))
bayesglm_validation_predict_value <- as.numeric(predict(object = bayesglm_train_fit, newdata = validation))
bayesglm_predict_value_mean <- mean(c(bayesglm_test_predict_value, bayesglm_validation_predict_value))
bayesglm_sd[i] <- sd(c(bayesglm_test_predict_value, bayesglm_validation_predict_value))
bayesglm_sd_mean <- mean(bayesglm_sd)
bayesglm_overfitting[i] <- bayesglm_holdout_RMSE_mean/bayesglm_train_RMSE_mean
bayesglm_overfitting_mean <- mean(bayesglm_overfitting)
bayesglm_overfitting_range <- range(bayesglm_overfitting)
y_hat_bayesglm <- c(bayesglm_test_predict_value, bayesglm_validation_predict_value)
bayesglm_end <- Sys.time()
bayesglm_duration <- bayesglm_end - bayesglm_start
bayesglm_duration_mean <- mean(bayesglm_duration)


####  Model #4 Bayes RNN: Bayes Regularization for feed forward neural networks ####
bayesrnn_start <- Sys.time()
bayesrnn_train_fit <- brnn::brnn(x = as.matrix(train), y = train$medv, neurons = 10, verbose = TRUE)
bayesrnn_train_RMSE_mean <- Metrics::rmse(actual = train$medv, predicted = predict(object = bayesrnn_train_fit, newdata = train))
bayesrnn_test_RMSE_mean <- Metrics::rmse(actual = test$medv, predicted = predict(object = bayesrnn_train_fit, newdata = test))
bayesrnn_validation_RMSE_mean <- Metrics::rmse(actual = validation$medv, predicted = predict(object = bayesrnn_train_fit, newdata = validation))
bayesrnn_holdout_RMSE_mean <- mean(c(bayesrnn_test_RMSE_mean, bayesrnn_validation_RMSE_mean))
bayesrnn_holdout_RMSE_sd_mean <- sd(c(bayesrnn_test_RMSE_mean, bayesrnn_validation_RMSE_mean))
bayesrnn_train_predict_value <- as.numeric(predict(object = bayesrnn_train_fit, newdata = train))
bayesrnn_test_predict_value <- as.numeric(predict(object = bayesrnn_train_fit, newdata = test))
bayesrnn_validation_predict_value <- as.numeric(predict(object = bayesrnn_train_fit, newdata = validation))
bayesrnn_predict_value_mean <- mean(c(bayesrnn_test_predict_value, bayesrnn_validation_predict_value))
bayesrnn_sd_mean <- sd(c(bayesrnn_test_predict_value, bayesrnn_validation_predict_value))
bayesrnn_overfitting[i] <-  bayesrnn_holdout_RMSE_mean/bayesrnn_train_RMSE_mean
bayesrnn_overfitting_mean <- mean(bayesrnn_overfitting)
bayesrnn_overfitting_range <- range(bayesrnn_overfitting)
y_hat_bayesrnn <- c(bayesrnn_test_predict_value, bayesrnn_validation_predict_value)
bayesrnn_end <- Sys.time()
bayesrnn_duration <- bayesrnn_end - bayesrnn_start
bayesrnn_duration_mean <- mean(bayesrnn_duration)

####  Model #5 Boosted Random Forest tuned ####
boost_rf_start <- Sys.time()
boost_rf_train_fit <- e1071::tune.randomForest(x = train, y = train$medv, mtry = ncol(train)-1)
boost_rf_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = boost_rf_train_fit$best.model,
  newdata = train))
boost_rf_train_RMSE_mean <- mean(boost_rf_train_RMSE)
boost_rf_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = boost_rf_train_fit$best.model,
  newdata = test))
boost_rf_test_RMSE_mean <- mean(boost_rf_test_RMSE)
boost_rf_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = boost_rf_train_fit$best.model,
  newdata = validation))
boost_rf_validation_RMSE_mean <- mean(boost_rf_validation_RMSE)
boost_rf_holdout_RMSE_mean <- mean(c(boost_rf_test_RMSE_mean, boost_rf_validation_RMSE_mean))
boost_rf_holdout_RMSE_sd_mean <- sd(c(boost_rf_test_RMSE_mean, boost_rf_validation_RMSE_mean))
boost_rf_train_predict_value <- as.numeric(predict(object = boost_rf_train_fit$best.model, newdata = train))
boost_rf_test_predict_value <- as.numeric(predict(object = boost_rf_train_fit$best.model, newdata = test))
boost_rf_validation_predict_value <- as.numeric(predict(object = boost_rf_train_fit$best.model, newdata = validation))
boost_rf_predict_value_mean <- mean(c(boost_rf_test_predict_value, boost_rf_validation_predict_value))
boost_rf_sd[i] <- sd(c(boost_rf_test_predict_value, boost_rf_validation_predict_value))
boost_rf_sd_mean <- mean(boost_rf_sd)
boost_rf_overfitting[i] <- boost_rf_holdout_RMSE_mean/boost_rf_train_RMSE_mean
boost_rf_overfitting_mean <- mean(boost_rf_overfitting)
boost_rf_overfitting_range <- range(boost_rf_overfitting)
y_hat_boost_rf <- c(boost_rf_test_predict_value, boost_rf_validation_predict_value)
boost_rf_end <- Sys.time()
boost_rf_duration[i] <- boost_rf_end - boost_rf_start
boost_rf_duration_mean <- mean(boost_rf_duration)


####  Model #6 Cubist ####
cubist_start <- Sys.time()
cubist_train_fit <- Cubist::cubist(x = train[,1:ncol(train)-1], y = train$medv)
cubist_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = cubist_train_fit, newdata = train))
cubist_train_RMSE_mean <- mean(cubist_train_RMSE)
cubist_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = cubist_train_fit, newdata = test))
cubist_test_RMSE_mean <- mean(cubist_test_RMSE)
cubist_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = cubist_train_fit, newdata = validation))
cubist_validation_RMSE_mean <- mean(cubist_validation_RMSE)
cubist_holdout_RMSE_mean <- mean(c(cubist_test_RMSE_mean, cubist_validation_RMSE_mean))
cubist_holdout_RMSE_sd_mean <- sd(c(cubist_test_RMSE_mean, cubist_validation_RMSE_mean))
cubist_train_predict_value <- as.numeric(predict(object = cubist_train_fit, newdata = train))
cubist_test_predict_value <- as.numeric(predict(object = cubist_train_fit, newdata = test))
cubist_validation_predict_value <- as.numeric(predict(object = cubist_train_fit, newdata = validation))
cubist_predict_value_mean <- mean(c(cubist_test_predict_value, cubist_validation_predict_value))
cubist_sd[i] <- sd(c(cubist_test_predict_value, cubist_validation_predict_value))
cubist_sd_mean <- mean(cubist_sd)
cubist_overfitting[i] <- cubist_holdout_RMSE_mean/cubist_train_RMSE_mean
cubist_overfitting_mean <- mean(cubist_overfitting)
cubist_overfitting_range <- range(cubist_overfitting)
y_hat_cubist <- c(cubist_test_predict_value, cubist_validation_predict_value)
cubist_end <- Sys.time()
cubist_duration[i] <- cubist_end - cubist_start
cubist_duration_mean <- mean(cubist_duration)


####  Model #7 Earth (Multivariate Adaptive Regression Splines, MARS) ####
earth_start <- Sys.time()
earth_train_fit <- earth::earth(medv ~ ., data = train)
earth_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = earth_train_fit, newdata = train))
earth_train_RMSE_mean <- mean(earth_train_RMSE)
earth_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = earth_train_fit, newdata = test))
earth_test_RMSE_mean <- mean(earth_test_RMSE)
earth_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = earth_train_fit, newdata = validation))
earth_validation_RMSE_mean <- mean(earth_validation_RMSE)
earth_holdout_RMSE_mean <- mean(c(earth_test_RMSE_mean, earth_validation_RMSE_mean))
earth_holdout_RMSE_sd_mean <- sd(c(earth_test_RMSE_mean, earth_validation_RMSE_mean))
earth_train_predict_value <- as.numeric(predict(object = earth_train_fit, newdata = train))
earth_test_predict_value <- as.numeric(predict(object = earth_train_fit, newdata = test))
earth_validation_predict_value <- as.numeric(predict(object = earth_train_fit, newdata = validation))
earth_predict_value_mean <- mean(c(earth_test_predict_value, earth_validation_predict_value))
earth_sd[i] <- sd(c(earth_test_predict_value, earth_validation_predict_value))
earth_sd_mean <- mean(earth_sd)
earth_overfitting[i] <- earth_holdout_RMSE_mean/earth_train_RMSE_mean
earth_overfitting_mean <- mean(earth_overfitting)
earth_overfitting_range <- range(earth_overfitting)
y_hat_earth <- c(earth_test_predict_value, earth_validation_predict_value)
earth_end <- Sys.time()
earth_duration[i] <- earth_end - earth_start
earth_duration_mean <- mean(earth_duration)


#### Model #8 GAM (Generalized Additive Models) with Smoothing Splines ####
gam_start <- Sys.time()
n_unique_vals <- map_dbl(df, n_distinct)

# Names of columns with >= 4 unique vals
keep <- names(n_unique_vals)[n_unique_vals >= 4]

gam_data <- df %>%
  dplyr::select(all_of(keep))

# Model data
train1 <- train %>%
  dplyr::select(all_of(keep))

test1 <- test %>%
  dplyr::select(all_of(keep))

validation1 <- validation %>%
  dplyr::select(all_of(keep))

names_df <- names(gam_data[,1:ncol(gam_data)-1])
f2 <- as.formula(paste0('medv ~', paste0('s(', names_df, ')', collapse = '+')))
gam_train_fit <- gam(f2, data = train1)
gam_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = gam_train_fit, newdata = train))
gam_train_RMSE_mean <- mean(gam_train_RMSE)
gam_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = gam_train_fit, newdata = test))
gam_test_RMSE_mean <- mean(gam_test_RMSE)
gam_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = gam_train_fit, newdata = validation))
gam_validation_RMSE_mean <- mean(gam_validation_RMSE)
gam_holdout_RMSE_mean <- mean(c(gam_test_RMSE_mean, gam_validation_RMSE_mean))
gam_holdout_RMSE_sd_mean <- sd(c(gam_test_RMSE_mean, gam_validation_RMSE_mean))
gam_train_predict_value <- as.numeric(predict(object = gam_train_fit, newdata = train))
gam_test_predict_value <- as.numeric(predict(object = gam_train_fit, newdata = test))
gam_validation_predict_value <- as.numeric(predict(object = gam_train_fit, newdata = validation))
gam_predict_value_mean <- mean(c(gam_test_predict_value, gam_validation_predict_value))
gam_sd[i] <- sd(c(gam_test_predict_value, gam_validation_predict_value))
gam_sd_mean <- mean(gam_sd)
gam_overfitting[i] <- gam_holdout_RMSE_mean/gam_train_RMSE_mean
gam_overfitting_mean <- mean(gam_overfitting)
gam_overfitting_range <- range(gam_overfitting)
y_hat_gam <- c(gam_test_predict_value, gam_validation_predict_value)
gam_end <- Sys.time()
gam_duration[i] <- gam_end - gam_start
gam_duration_mean <- mean(gam_duration)

####  Model #9 Gradient Boosted ####
gb_start <- Sys.time()
gb_train_fit <- gbm::gbm(train$medv ~ ., data = train, distribution = "gaussian",n.trees = 100, shrinkage = 0.1, interaction.depth = 10)
gb_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = gb_train_fit, newdata = train))
gb_train_RMSE_mean <- mean(gb_train_RMSE)
gb_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = gb_train_fit, newdata = test))
gb_test_RMSE_mean <- mean(gb_test_RMSE)
gb_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = gb_train_fit, newdata = validation))
gb_validation_RMSE_mean <- mean(gb_validation_RMSE)
gb_holdout_RMSE_mean <- mean(c(gb_test_RMSE_mean, gb_validation_RMSE_mean))
gb_holdout_RMSE_sd_mean <- sd(c(gb_test_RMSE_mean, gb_validation_RMSE_mean))
gb_train_predict_value <- as.numeric(predict(object = gb_train_fit, newdata = train))
gb_test_predict_value <- as.numeric(predict(object = gb_train_fit, newdata = test))
gb_validation_predict_value <- as.numeric(predict(object = gb_train_fit, newdata = validation))
gb_predict_value_mean <- mean(c(gb_test_predict_value, gb_validation_predict_value))
gb_sd[i] <- sd(c(gb_test_predict_value, gb_validation_predict_value))
gb_sd_mean <- mean(gb_sd)
gb_overfitting[i] <- gb_holdout_RMSE_mean/gb_train_RMSE_mean
gb_overfitting_mean <- mean(gb_overfitting)
gb_overfitting_range <- range(gb_overfitting)
y_hat_gb <- c(gb_test_predict_value, gb_validation_predict_value)
gb_end <- Sys.time()
gb_duration[i] <- gb_end - gb_start
gb_duration_mean <- mean(gb_duration)


####  Model #10 K-Nearest Neighbors ####
knn_start <- Sys.time()
knn_train_fit <- e1071::tune.gknn(x = train[,1:ncol(train)-1], y = train$medv, scale = TRUE, k = c(1:25))
knn_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = knn_train_fit$best.model,
  newdata = train[,1:ncol(train)-1], k = knn_train_fit$best_model$k))
knn_train_RMSE_mean <- mean(knn_train_RMSE)
knn_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = knn_train_fit$best.model,
  k = knn_train_fit$best_model$k, newdata = test[,1:ncol(test)-1]))
knn_test_RMSE_mean <- mean(knn_test_RMSE)
knn_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = knn_train_fit$best.model,
  newdata = validation[,1:ncol(validation) -1], k = knn_train_fit$best_model$k))
knn_validation_RMSE_mean <- mean(knn_validation_RMSE)
knn_holdout_RMSE_mean <- mean(c(knn_test_RMSE_mean, knn_validation_RMSE_mean))
knn_holdout_RMSE_sd_mean <- sd(c(knn_test_RMSE_mean, knn_validation_RMSE_mean))
knn_train_predict_value <- as.numeric(predict(object = knn_train_fit$best.model, newdata = train[,1:ncol(train)-1],
  k = knn_train_fit$best_model$k))
knn_test_predict_value <- as.numeric(predict(object = knn_train_fit$best.model, newdata = test[,1:ncol(test)-1],
  k = knn_train_fit$best_model$k))
knn_validation_predict_value <- as.numeric(predict(object = knn_train_fit$best.model, newdata = validation[,1:ncol(test)-1],
  k = knn_train_fit$best_model$k))
knn_predict_value <- mean(c(knn_test_predict_value, knn_validation_predict_value))
knn_predict_value_mean <- mean(c(knn_test_predict_value, knn_validation_predict_value))
knn_sd[i] <- sd(c(knn_test_predict_value, knn_validation_predict_value))
knn_sd_mean <- mean(knn_sd)
knn_overfitting[i] <- knn_holdout_RMSE_mean/knn_train_RMSE_mean
knn_overfitting_mean <- mean(knn_overfitting)
knn_overfitting_range <- range(knn_overfitting)
y_hat_knn <- c(knn_test_predict_value, knn_validation_predict_value)
knn_end <- Sys.time()
knn_duration[i] <- knn_end - knn_start
knn_duration_mean <- mean(knn_duration)


####  Model 11 Linear ####
linear_start <- Sys.time()
linear_train_fit <- e1071::tune.rpart(formula = medv ~ ., data = train)
linear_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = linear_train_fit$best.model, newdata = train))
linear_train_RMSE_mean <- mean(linear_train_RMSE)
linear_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = linear_train_fit$best.model, newdata = test))
linear_test_RMSE_mean <- mean(linear_test_RMSE)
linear_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = linear_train_fit$best.model, newdata = validation))
linear_validation_RMSE_mean <- mean(linear_validation_RMSE)
linear_holdout_RMSE_mean <- mean(c(linear_test_RMSE_mean, linear_validation_RMSE_mean))
linear_holdout_RMSE_sd_mean <- sd(c(linear_test_RMSE_mean, linear_validation_RMSE_mean))
linear_train_predict_value <- as.numeric(predict(object = linear_train_fit$best.model, newdata = train))
linear_test_predict_value <- as.numeric(predict(object = linear_train_fit$best.model, newdata = test))
linear_validation_predict_value <- as.numeric(predict(object = linear_train_fit$best.model, newdata = validation))
linear_predict_value_mean <- mean(c(linear_test_predict_value, linear_validation_predict_value))
linear_sd[i] <- sd(c(linear_test_predict_value, linear_validation_predict_value))
linear_sd_mean <- mean(linear_sd)
linear_overfitting[i] <- linear_holdout_RMSE_mean/linear_train_RMSE_mean
linear_overfitting_mean <- mean(linear_overfitting)
linear_overfitting_range <- range(linear_overfitting)
y_hat_linear <- c(linear_test_predict_value, linear_validation_predict_value)
linear_end <- Sys.time()
linear_duration[i] <- linear_end - linear_start
linear_duration_mean <- mean(linear_duration)


####  Model 12 LQS ####
lqs_start <- Sys.time()
lqs_train_fit <- MASS::lqs(train$medv ~ ., data = train)
lqs_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = lqs_train_fit, newdata = train))
lqs_train_RMSE_mean <- mean(lqs_train_RMSE)
lqs_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = lqs_train_fit, newdata = test))
lqs_test_RMSE_mean <- mean(lqs_test_RMSE)
lqs_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = lqs_train_fit, newdata = validation))
lqs_validation_RMSE_mean <- mean(lqs_validation_RMSE)
lqs_holdout_RMSE_mean <- mean(c(lqs_test_RMSE_mean, lqs_validation_RMSE_mean))
lqs_holdout_RMSE_sd_mean <- sd(c(lqs_test_RMSE_mean, lqs_validation_RMSE_mean))
lqs_train_predict_value <- as.numeric(predict(object = lqs_train_fit, newdata = train))
lqs_test_predict_value <- as.numeric(predict(object = lqs_train_fit, newdata = test))
lqs_validation_predict_value <- as.numeric(predict(object = lqs_train_fit, newdata = validation))
lqs_predict_value_mean <- mean(c(lqs_test_predict_value, lqs_validation_predict_value))
lqs_sd[i] <- sd(c(lqs_test_predict_value, lqs_validation_predict_value))
lqs_sd_mean <- mean(lqs_sd)
lqs_overfitting[i] <- lqs_holdout_RMSE_mean/lqs_train_RMSE_mean
lqs_overfitting_mean <- mean(lqs_overfitting)
lqs_overfitting_range <- range(lqs_overfitting)
y_hat_lqs <- c(lqs_test_predict_value, lqs_validation_predict_value)
lqs_end <- Sys.time()
lqs_duration[i] <- lqs_end - lqs_start
lqs_duration_mean <- mean(lqs_duration)


#### 7 Model 13 Partial Least Squares ####
pls_start <- Sys.time()
pls_train_fit <- pls::plsr(train$medv ~ ., data = train)
pls_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = pls_train_fit, newdata = train))
pls_train_RMSE_mean <- mean(pls_train_RMSE)
pls_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = pls_train_fit, newdata = test))
pls_test_RMSE_mean <- mean(pls_test_RMSE)
pls_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = pls_train_fit, newdata = validation))
pls_validation_RMSE_mean <- mean(pls_validation_RMSE)
pls_holdout_RMSE_mean <- mean(c(pls_test_RMSE_mean, pls_validation_RMSE_mean))
pls_holdout_RMSE_sd_mean <- sd(c(pls_test_RMSE_mean, pls_validation_RMSE_mean))
pls_train_predict_value <- predict(object = pls_train_fit, newdata = train)
pls_test_predict_value <- predict(object = pls_train_fit, newdata = test)
pls_validation_predict_value <- predict(object = pls_train_fit, newdata = validation )
pls_predict_value_mean <- mean(c(pls_test_predict_value, pls_validation_predict_value))
pls_sd[i] <- sd(c(pls_test_predict_value, pls_validation_predict_value))
pls_sd_mean <- mean(pls_sd)
pls_overfitting[i] <- pls_holdout_RMSE_mean/pls_train_RMSE_mean
pls_overfitting_mean <- mean(pls_overfitting)
pls_overfitting_range <- range(pls_overfitting)
y_hat_pls <- c(pls_test_predict_value[,,1], pls_validation_predict_value[,,1])
pls_end <- Sys.time()
pls_duration[i] <- pls_end - pls_start
pls_duration_mean <- mean(pls_duration)


####  Model 14 Principle Components Analysis ####
pcr_start <- Sys.time()
pcr_train_fit <- pls::pcr(train$medv ~ ., data = train)
pcr_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = pcr_train_fit, newdata = train))
pcr_train_RMSE_mean <- mean(pcr_train_RMSE)
pcr_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = pcr_train_fit, newdata = test))
pcr_test_RMSE_mean <- mean(pcr_test_RMSE)
pcr_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = pcr_train_fit, newdata = validation))
pcr_validation_RMSE_mean <- mean(pcr_validation_RMSE)
pcr_holdout_RMSE_mean <- mean(c(pcr_test_RMSE_mean, pcr_validation_RMSE_mean))
pcr_holdout_RMSE_sd_mean <- sd(c(pcr_test_RMSE_mean, pcr_validation_RMSE_mean))
pcr_train_predict_value <- predict(object = pcr_train_fit, newdata = train)
pcr_test_predict_value <- predict(object = pcr_train_fit, newdata = test)
pcr_validation_predict_value <- predict(object = pcr_train_fit, newdata = validation )
pcr_predict_value_mean <- mean(c(pcr_test_predict_value, pcr_validation_predict_value))
pcr_sd[i] <- sd(c(pcr_test_predict_value, pcr_validation_predict_value))
pcr_sd_mean <- mean(pcr_sd)
pcr_overfitting[i] <- pcr_holdout_RMSE_mean/pcr_train_RMSE_mean
pcr_overfitting_mean <- mean(pcr_overfitting)
pcr_overfitting_range <- range(pcr_overfitting)
y_hat_pcr <- c(pcr_test_predict_value[,,1], pcr_validation_predict_value[,,1])
pcr_end <- Sys.time()
pcr_duration[i] <- pcr_end - pcr_start
pcr_duration_mean <- mean(pcr_duration)


####  Model 15 Random Forest tuned ####
rf_start <- Sys.time()
rf_train_fit <- tune.randomForest(x = train, y = train$medv, data = train)
rf_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = rf_train_fit$best.model, newdata = train))
rf_train_RMSE_mean <- mean(rf_train_RMSE)
rf_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = rf_train_fit$best.model, newdata = test))
rf_test_RMSE_mean <- mean(rf_test_RMSE)
rf_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = rf_train_fit$best.model, newdata = validation))
rf_validation_RMSE_mean <- mean(rf_validation_RMSE)
rf_holdout_RMSE_mean <- mean(c(rf_test_RMSE_mean, rf_validation_RMSE_mean))
rf_holdout_RMSE_sd_mean <- sd(c(rf_test_RMSE_mean, rf_validation_RMSE_mean))
rf_train_predict_value <- predict(object = rf_train_fit$best.model, newdata = train)
rf_test_predict_value <- predict(object = rf_train_fit$best.model, newdata = test)
rf_validation_predict_value <- predict(object = rf_train_fit$best.model, newdata = validation )
rf_predict_value_mean <- mean(c(rf_test_predict_value, rf_validation_predict_value))
rf_sd[i] <- sd(c(rf_test_predict_value, rf_validation_predict_value))
rf_sd_mean <- mean(rf_sd)
rf_overfitting[i] <- rf_holdout_RMSE_mean/rf_train_RMSE_mean
rf_overfitting_mean <- mean(rf_overfitting)
rf_overfitting_range <- range(rf_overfitting)
y_hat_rf <- c(rf_test_predict_value, rf_validation_predict_value)
rf_end <- Sys.time()
rf_duration[i] <- rf_end - rf_start
rf_duration_mean <- mean(rf_duration)


####  Model 16 Robust Regression ####
robust_start <- Sys.time()
robust_train_fit <- MASS::rlm(x = train[,1:ncol(df)-1], y = train$medv)
robust_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = robust_train_fit$fitted.values)
robust_train_RMSE_mean <- mean(robust_train_RMSE)
robust_test_RMSE[i] <- Metrics::rmse(actual = test$medv, predicted = predict(object = MASS::rlm(medv ~ .,data = train), newdata = test))
robust_test_RMSE_mean <- mean(robust_test_RMSE)
robust_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = MASS::rlm(medv ~ .,data = train),
  newdata = validation))
robust_validation_RMSE_mean <- mean(robust_validation_RMSE)
robust_holdout_RMSE_mean <- mean(c(robust_test_RMSE_mean, robust_validation_RMSE_mean))
robust_holdout_RMSE_sd_mean <- sd(c(robust_test_RMSE_mean, robust_validation_RMSE_mean))
robust_train_predict_value <- as.numeric(predict(object = MASS::rlm(medv ~ .,data = train), newdata = train))
robust_test_predict_value <- as.numeric(predict(object = MASS::rlm(medv ~ .,data = train), newdata = test))
robust_validation_predict_value <- as.numeric(predict(object = MASS::rlm(medv ~ .,data = train), newdata = validation))
robust_predict_value_mean <- mean(c(robust_test_predict_value, robust_validation_predict_value))
robust_sd[i] <- sd(c(robust_test_predict_value, robust_validation_predict_value))
robust_sd_mean <- mean(robust_sd)
robust_overfitting[i] <- robust_holdout_RMSE_mean/robust_train_RMSE_mean
robust_overfitting_mean <- mean(robust_overfitting)
robust_overfitting_range <- range(robust_overfitting)
y_hat_robust <- c(robust_test_predict_value, robust_validation_predict_value)
robust_end <- Sys.time()
robust_duration[i] <- robust_end - robust_start
robust_duration_mean <- mean(robust_duration)


####  Model 17 Rpart (also known as cart) ####
rpart_start <- Sys.time()
rpart_train_fit <- rpart::rpart(train$medv ~ ., data = train)
rpart_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = rpart_train_fit, newdata = train))
rpart_train_RMSE_mean <- mean(rpart_train_RMSE)
rpart_test_RMSE[i] <-  Metrics::rmse(actual = test$medv, predicted = predict(object = rpart_train_fit, newdata = test))
rpart_test_RMSE_mean <- mean(rpart_test_RMSE)
rpart_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = rpart_train_fit, newdata = validation))
rpart_validation_RMSE_mean <- mean(rpart_validation_RMSE)
rpart_holdout_RMSE_mean <- mean(c(rpart_test_RMSE_mean, rpart_validation_RMSE_mean))
rpart_holdout_RMSE_sd_mean <- sd(c(rpart_test_RMSE_mean, rpart_validation_RMSE_mean))
rpart_train_predict_value <- as.numeric(predict(object = rpart::rpart(medv ~ .,data = train), newdata = train))
rpart_test_predict_value <- as.numeric(predict(object = rpart::rpart(medv ~ .,data = train), newdata = test))
rpart_validation_predict_value <- as.numeric(predict(object = rpart::rpart(medv ~ .,data = train), newdata = validation))
rpart_predict_value_mean <- mean(c(rpart_test_predict_value, rpart_validation_predict_value))
rpart_sd[i] <- sd(rpart_test_predict_value)
rpart_sd_mean <- mean(rpart_sd)
rpart_overfitting[i] <- rpart_holdout_RMSE_mean/rpart_train_RMSE_mean
rpart_overfitting_mean <- mean(rpart_overfitting)
rpart_overfitting_range <- range(rpart_overfitting)
y_hat_rpart <- c(rpart_test_predict_value, rpart_validation_predict_value)
rpart_end <- Sys.time()
rpart_duration[i] <- rpart_end - rpart_start
rpart_duration_mean <- mean(rpart_duration)


####  Model 18 Support Vector Machines tuned ####
svm_start <- Sys.time()
svm_train_fit <- e1071::tune.svm(x = train, y = train$medv, data = train)
svm_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = svm_train_fit$best.model, newdata = train))
svm_train_RMSE_mean <- mean(svm_train_RMSE)
svm_test_RMSE[i] <-  Metrics::rmse(actual = test$medv, predicted = predict(object = svm_train_fit$best.model, newdata = test))
svm_test_RMSE_mean <- mean(svm_test_RMSE)
svm_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = svm_train_fit$best.model, newdata = validation))
svm_validation_RMSE_mean <- mean(svm_validation_RMSE)
svm_holdout_RMSE_mean <- mean(c(svm_test_RMSE_mean, svm_validation_RMSE_mean))
svm_holdout_RMSE_sd_mean <- sd(c(svm_validation_RMSE_mean))
svm_train_predict_value <- as.numeric(predict(object = svm_train_fit$best.model, newdata = train))
svm_test_predict_value <- as.numeric(predict(object = svm_train_fit$best.model, newdata = test))
svm_validation_predict_value <- as.numeric(predict(object = svm_train_fit$best.model, newdata = validation))
svm_predict_value_mean <- mean(c(svm_test_predict_value, svm_validation_predict_value))
svm_sd[i] <- sd(c(svm_test_predict_value, svm_validation_predict_value))
svm_sd_mean <- mean(svm_sd)
svm_overfitting[i] <- svm_holdout_RMSE_mean/svm_train_RMSE_mean
svm_overfitting_mean <- mean(svm_overfitting)
svm_overfitting_range <- range(svm_overfitting)
y_hat_svm <- c(svm_test_predict_value, svm_validation_predict_value)
svm_end <- Sys.time()
svm_duration[i] <- svm_end - svm_start
svm_duration_mean <- mean(svm_duration)


####  Model 19 Trees ####
tree_start <- Sys.time()
tree_train_fit <- tree::tree(train$medv ~ ., data = train)
tree_train_fit_cv <- cv.tree(tree_train_fit)
size <- tree_train_fit_cv$size[which.min(tree_train_fit_cv$dev)]
tree_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = tree_train_fit, newdata = train))
tree_train_RMSE_mean <- mean(tree_train_RMSE)
tree_test_RMSE[i] <-  Metrics::rmse(actual = test$medv, predicted = predict(object = tree_train_fit, newdata = test))
tree_test_RMSE_mean <- mean(tree_test_RMSE)
tree_validation_RMSE[i] = Metrics::rmse(actual = validation$medv, predicted = predict(object = tree_train_fit, newdata = validation))
tree_validation_RMSE_mean <- mean(tree_validation_RMSE)
tree_holdout_RMSE_mean <- mean(c(tree_test_RMSE_mean, tree_validation_RMSE_mean))
tree_holdout_RMSE_sd_mean <- sd(c(tree_test_RMSE_mean, tree_validation_RMSE_mean))
tree_train_predict_value <- as.numeric(predict(object = tree::tree(medv ~ .,data = train), newdata = train))
tree_test_predict_value <- as.numeric(predict(object = tree::tree(medv ~ .,data = train), newdata = test))
tree_validation_predict_value <- as.numeric(predict(object = tree::tree(medv ~ .,data = train), newdata = validation))
tree_predict_value_mean <- mean(c(tree_test_predict_value, tree_validation_predict_value))
tree_sd[i] <- sd(tree_test_predict_value)
tree_sd_mean <- mean(tree_sd)
tree_overfitting[i] <- tree_holdout_RMSE_mean/tree_train_RMSE_mean
tree_overfitting_mean <- mean(tree_overfitting)
tree_overfitting_range <- range(tree_overfitting)
y_hat_tree <- c(tree_test_predict_value, tree_validation_predict_value)
tree_end <- Sys.time()
tree_duration[i] <- tree_end - tree_start
tree_duration_mean <- mean(tree_duration)


####  Model 20 XGBoost ####
xgb_start <- Sys.time()
train_x = data.matrix(train[, -ncol(train)])
train_y = train[,ncol(train)]

#define predictor and response variables in test set
test_x = data.matrix(test[, -ncol(test)])
test_y = test[, ncol(test)]

#define predictor and response variables in validation set
validation_x = data.matrix(validation[, -ncol(validation)])
validation_y = validation[, ncol(validation)]

#define final train, test and validation sets
xgb_train = xgboost::xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgboost::xgb.DMatrix(data = test_x, label = test_y)
xgb_validation = xgboost::xgb.DMatrix(data = validation_x, label = validation_y)

#define watchlist
watchlist = list(train = xgb_train, validation=xgb_validation)
watchlist_test <- list(train = xgb_train, test = xgb_test)
watchlist_validation <- list(train = xgb_train, validation = xgb_validation)

#fit XGBoost model and display training and validation data at each round

xgb_model = xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist_test, nrounds = 70)
xgb_model_validation = xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist_validation, nrounds = 70)

xgboost_min <- which.min(xgb_model$evaluation_log$validation_rmse)
xgboost_validation.min <- which.min(xgb_model$evaluation_log$validation_rmse)

xgb_train_RMSE[i] <- Metrics::rmse(actual = train$medv, predicted = predict(object = xgb_model, newdata = train_x))
xgb_train_RMSE_mean <- mean(xgb_train_RMSE)
xgb_test_RMSE <- Metrics::rmse(actual = test$medv, predicted = predict(object = xgb_model, newdata = test_x))
xgb_test_RMSE_mean <- mean(xgb_test_RMSE)
xgb_validation_RMSE <- round(Metrics::rmse(actual = validation$medv, predicted = predict(object = xgb_model, newdata = validation_x)), 4)
xgb_validation_RMSE_mean <- mean(xgb_validation_RMSE)

xgb_holdout_RMSE_mean <- mean(c(xgb_test_RMSE_mean, xgb_validation_RMSE_mean))
xgb_holdout_RMSE_sd_mean <- sd(c(xgb_test_RMSE_mean, xgb_validation_RMSE_mean))

y_hat_xgb <- c(predict(object = xgb_model, newdata = test_x), predict(object = xgb_model, newdata = validation_x))
xgb_mean <-  mean(y_hat_xgb)
xgb_sd <-  sd(y_hat_xgb)
xgb_overfitting[i] <- xgb_holdout_RMSE_mean/xgb_train_RMSE_mean
xgb_overfitting_mean <- mean(xgb_overfitting)
xgb_overfitting_range <- range(xgb_overfitting)
xgb_end <- Sys.time()
xgb_duration[i] <- xgb_end - xgb_start
xgb_duration_mean <- mean(xgb_duration)


###################################################################################################################

########################################     Start Weighted ENSEMBLES Here      ############################################

###################################################################################################################


#### Build the Ensemble ####
ensemble <- data.frame('BagRF' = y_hat_bag_rf* 1/bag_rf_holdout_RMSE_mean, 'BayesGLM' = y_hat_bayesglm* 1/bayesglm_holdout_RMSE_mean,
  'BayesRNN'= y_hat_bayesrnn*1/bayesrnn_holdout_RMSE_mean, 'BoostRF' = y_hat_boost_rf*1/boost_rf_holdout_RMSE_mean,
  'Cubist' = y_hat_cubist*1/cubist_holdout_RMSE_mean, 'Earth' = y_hat_earth*1/earth_holdout_RMSE_mean,
  'GAM_(with_splines)' = y_hat_gam*1/gam_holdout_RMSE_mean,'Gradient Boosted' = y_hat_gb*1/gb_holdout_RMSE_mean,
  'Bagging' = y_hat_bagging*1/bagging_holdout_RMSE_mean, 'KNN' = y_hat_knn*1/knn_holdout_RMSE_mean,
  'Linear' = y_hat_linear*1/linear_holdout_RMSE_mean, 'LQS' = y_hat_lqs*1/lqs_holdout_RMSE_mean,
  'PCR' = y_hat_pcr*1/pcr_holdout_RMSE_mean,'PLS' = y_hat_pls*1/pls_holdout_RMSE_mean,
  'RandomForest' = y_hat_rf*1/rf_holdout_RMSE_mean, 'Rpart' = y_hat_rpart*1/rpart_holdout_RMSE_mean,
  'SVM' = y_hat_svm*1/svm_holdout_RMSE_mean, 'Tree' = y_hat_tree*1/tree_holdout_RMSE_mean)
  
  ensemble$Row_mean <- rowMeans(ensemble)
  ensemble$y_ensemble = c(test$medv, validation$medv)
  y_ensemble = c(test$medv, validation$medv)
  
  #### Split the ensemble data into train (60%), test (20%) and validation (20%) ####
  ensemble_idx <- sample(seq(1, 3), size = nrow(ensemble), replace = TRUE, prob = c(.6, .2, .2))
  ensemble_train <- ensemble[ensemble_idx == 1,]
  ensemble_test <- ensemble[ensemble_idx == 2,]
  ensemble_validation <- ensemble[ensemble_idx == 3,]
  
  #### Model 21: Ensemble Using Bagged Random Forest tuned ####
  ensemble_bag_rf_start <- Sys.time()
  ensemble_bag_rf_train_fit <- e1071::tune.randomForest(x = ensemble_train, y = ensemble_train$y_ensemble, mtry = ncol(ensemble_train)-1)
  ensemble_bag_rf_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_bag_rf_train_fit$best.model,
    newdata = ensemble_train))
  ensemble_bag_rf_train_RMSE_mean <- mean(ensemble_bag_rf_train_RMSE)
  ensemble_bag_rf_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_bag_rf_train_fit$best.model,
    newdata = ensemble_test))
  ensemble_bag_rf_test_RMSE_mean <- mean(ensemble_bag_rf_test_RMSE)
  ensemble_bag_rf_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble,
    predicted = predict(object = ensemble_bag_rf_train_fit$best.model,
    newdata = ensemble_validation))
  ensemble_bag_rf_validation_RMSE_mean <- mean(ensemble_bag_rf_validation_RMSE)
  ensemble_bag_rf_holdout_RMSE_mean <- mean(c(ensemble_bag_rf_test_RMSE_mean, ensemble_bag_rf_validation_RMSE_mean))
  ensemble_bag_rf_holdout_RMSE_sd_mean <- sd(c(ensemble_bag_rf_test_RMSE_mean, ensemble_bag_rf_validation_RMSE_mean))
  ensemble_bag_rf_train_predict_value <- as.numeric(predict(object = ensemble_bag_rf_train_fit$best.model, newdata = ensemble_train))
  ensemble_bag_rf_test_predict_value <- as.numeric(predict(object = ensemble_bag_rf_train_fit$best.model, newdata = ensemble_test))
  ensemble_bag_rf_validation_predict_value <- as.numeric(predict(object = ensemble_bag_rf_train_fit$best.model, newdata = ensemble_validation))
  ensemble_bag_rf_predict_value_mean <- mean(c(ensemble_bag_rf_test_predict_value, ensemble_bag_rf_validation_predict_value))
  ensemble_bag_rf_sd[i] <- sd(c(ensemble_bag_rf_test_predict_value, ensemble_bag_rf_validation_predict_value))
  ensemble_bag_rf_sd_mean <- mean(ensemble_bag_rf_sd)
  ensemble_bag_rf_overfitting[i] <- ensemble_bag_rf_holdout_RMSE_mean/ensemble_bag_rf_train_RMSE_mean
  ensemble_bag_rf_overfitting_mean <- mean(ensemble_bag_rf_overfitting)
  ensemble_bag_rf_overfitting_range <- range(ensemble_bag_rf_overfitting)
  ensemble_y_hat_bag_rf <- c(ensemble_bag_rf_test_predict_value, ensemble_bag_rf_validation_predict_value)
  ensemble_bag_rf_end <- Sys.time()
  ensemble_bag_rf_duration[i] <- ensemble_bag_rf_end - ensemble_bag_rf_start
  ensemble_bag_rf_duration_mean <- mean(ensemble_bag_rf_duration)
  
  
  #### Model 22: Ensemble Using Bagged Random Forest tuned ####
  ensemble_bagging_start <- Sys.time()
  ensemble_bagging_train_fit <- ipred::bagging(formula = y_ensemble ~ ., data = ensemble_train)
  ensemble_bagging_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_bagging_train_fit,
    newdata = ensemble_train))
  ensemble_bagging_train_RMSE_mean <- mean(ensemble_bagging_train_RMSE)
  ensemble_bagging_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_bagging_train_fit,
    newdata = ensemble_test))
  ensemble_bagging_test_RMSE_mean <- mean(ensemble_bagging_test_RMSE)
  ensemble_bagging_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_bagging_train_fit,
    newdata = ensemble_validation))
  ensemble_bagging_validation_RMSE_mean <- mean(ensemble_bagging_validation_RMSE)
  ensemble_bagging_holdout_RMSE_mean <- mean(c(ensemble_bagging_test_RMSE_mean, ensemble_bagging_validation_RMSE_mean))
  ensemble_bagging_holdout_RMSE_sd_mean <- sd(c(ensemble_bagging_test_RMSE_mean, ensemble_bagging_validation_RMSE_mean))
  ensemble_bagging_train_predict_value <- as.numeric(predict(object = ensemble_bagging_train_fit, newdata = ensemble_train))
  ensemble_bagging_test_predict_value <- as.numeric(predict(object = ensemble_bagging_train_fit, newdata = ensemble_test))
  ensemble_bagging_validation_predict_value <- as.numeric(predict(object = ensemble_bagging_train_fit, newdata = ensemble_validation))
  ensemble_bagging_predict_value_mean <- mean(c(ensemble_bagging_test_predict_value, ensemble_bagging_validation_predict_value))
  ensemble_bagging_sd[i] <- sd(c(ensemble_bagging_test_predict_value, ensemble_bagging_validation_predict_value))
  ensemble_bagging_sd_mean <- mean(ensemble_bagging_sd)
  ensemble_bagging_overfitting[i] <- ensemble_bagging_holdout_RMSE_mean/ensemble_bagging_train_RMSE_mean
  ensemble_bagging_overfitting_mean <- mean(ensemble_bagging_overfitting)
  ensemble_bagging_overfitting_range <- range(ensemble_bagging_overfitting)
  ensemble_y_hat_bagging <- c(ensemble_bagging_test_predict_value, ensemble_bagging_validation_predict_value)
  ensemble_bagging_end <- Sys.time()
  ensemble_bagging_duration[i] <- ensemble_bagging_end - ensemble_bagging_start
  ensemble_bagging_duration_mean <- mean(ensemble_bagging_duration)
  
  
  #### Model 23: Ensemble Using BayesGLM ####
  ensemble_bayesglm_start <- Sys.time()
  ensemble_bayesglm_train_fit <- arm::bayesglm(y_ensemble ~ ., data = ensemble_train, family = gaussian(link = "identity"))
  ensemble_bayesglm_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_bayesglm_train_fit,
    newdata = ensemble_train))
  ensemble_bayesglm_train_RMSE_mean <- mean(ensemble_bayesglm_train_RMSE)
  ensemble_bayesglm_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_bayesglm_train_fit,
    newdata = ensemble_test))
  ensemble_bayesglm_test_RMSE_mean <- mean(ensemble_bayesglm_test_RMSE)
  ensemble_bayesglm_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_bayesglm_train_fit,
    newdata = ensemble_validation))
  ensemble_bayesglm_validation_RMSE_mean <- mean(ensemble_bayesglm_validation_RMSE)
  ensemble_bayesglm_holdout_RMSE_mean <- mean(c(ensemble_bayesglm_test_RMSE_mean, ensemble_bayesglm_validation_RMSE_mean))
  ensemble_bayesglm_holdout_RMSE_sd_mean <- sd(c(ensemble_bayesglm_test_RMSE_mean, ensemble_bayesglm_validation_RMSE_mean))
  ensemble_bayesglm_train_predict_value <- as.numeric(predict(object = ensemble_bayesglm_train_fit, newdata = ensemble_train))
  ensemble_bayesglm_test_predict_value <- as.numeric(predict(object = ensemble_bayesglm_train_fit, newdata = ensemble_test))
  ensemble_bayesglm_validation_predict_value <- as.numeric(predict(object = ensemble_bayesglm_train_fit, newdata = ensemble_validation))
  ensemble_bayesglm_predict_value_mean <- mean(c(ensemble_bayesglm_test_predict_value, ensemble_bayesglm_validation_predict_value))
  ensemble_bayesglm_sd[i] <- sd(c(ensemble_bayesglm_test_predict_value, ensemble_bayesglm_validation_predict_value))
  ensemble_bayesglm_sd_mean <- mean(ensemble_bayesglm_sd)
  ensemble_bayesglm_overfitting[i] <- ensemble_bayesglm_holdout_RMSE_mean/ensemble_bayesglm_train_RMSE_mean
  ensemble_bayesglm_overfitting_mean <- mean(ensemble_bayesglm_overfitting)
  ensemble_bayesglm_overfitting_range <- range(ensemble_bayesglm_overfitting)
  ensemble_y_hat_bayesglm <- c(ensemble_bayesglm_test_predict_value, ensemble_bayesglm_validation_predict_value)
  ensemble_bayesglm_end <- Sys.time()
  ensemble_bayesglm_duration[i] <- ensemble_bayesglm_end - ensemble_bayesglm_start
  ensemble_bayesglm_duration_mean <- mean(ensemble_bayesglm_duration)
  
  
  #### Model 24: Ensemble Using Bayes RNN ####
  ensemble_bayesrnn_start <- Sys.time()
  ensemble_bayesrnn_train_fit <- brnn::brnn(x = as.matrix(ensemble_train), y = ensemble_train$y_ensemble, neurons = 10, verbose = TRUE)
  ensemble_bayesrnn_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_bayesrnn_train_fit,
    newdata = ensemble_train))
  ensemble_bayesrnn_train_RMSE_mean <- mean(ensemble_bayesrnn_train_RMSE)
  ensemble_bayesrnn_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_bayesrnn_train_fit,
    newdata = ensemble_test))
  ensemble_bayesrnn_test_RMSE_mean <- mean(ensemble_bayesrnn_test_RMSE)
  ensemble_bayesrnn_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_bayesrnn_train_fit,
    newdata = ensemble_validation))
  ensemble_bayesrnn_validation_RMSE_mean <- mean(ensemble_bayesrnn_validation_RMSE)
  ensemble_bayesrnn_holdout_RMSE_mean <- mean(c(ensemble_bayesrnn_test_RMSE_mean, ensemble_bayesrnn_validation_RMSE_mean))
  ensemble_bayesrnn_holdout_RMSE_sd_mean <- sd(c(ensemble_bayesrnn_test_RMSE_mean, ensemble_bayesrnn_validation_RMSE_mean))
  ensemble_bayesrnn_train_predict_value <- as.numeric(predict(object = ensemble_bayesrnn_train_fit, newdata = ensemble_train))
  ensemble_bayesrnn_test_predict_value <- as.numeric(predict(object = ensemble_bayesrnn_train_fit, newdata = ensemble_test))
  ensemble_bayesrnn_validation_predict_value <- as.numeric(predict(object = ensemble_bayesrnn_train_fit, newdata = ensemble_validation))
  ensemble_bayesrnn_predict_value_mean <- mean(c(ensemble_bayesrnn_test_predict_value, ensemble_bayesrnn_validation_predict_value))
  ensemble_bayesrnn_sd[i] <- sd(c(ensemble_bayesrnn_test_predict_value, ensemble_bayesrnn_validation_predict_value))
  ensemble_bayesrnn_sd_mean <- mean(ensemble_bayesrnn_sd)
  ensemble_bayesrnn_overfitting[i] <- ensemble_bayesrnn_holdout_RMSE_mean/ensemble_bayesrnn_train_RMSE_mean
  ensemble_bayesrnn_overfitting_mean <- mean(ensemble_bayesrnn_overfitting)
  ensemble_bayesrnn_overfitting_range <- range(ensemble_bayesrnn_overfitting)
  ensemble_y_hat_bayesrnn <- c(ensemble_bayesrnn_test_predict_value, ensemble_bayesrnn_validation_predict_value)
  ensemble_bayesrnn_end <- Sys.time()
  ensemble_bayesrnn_duration[i] <- ensemble_bayesrnn_end - ensemble_bayesrnn_start
  ensemble_bayesrnn_duration_mean <- mean(ensemble_bayesrnn_duration)
  
  
  #### Model 25: Ensemble Using Boosted Random Forest tuned ####
  ensemble_boost_rf_start <- Sys.time()
  ensemble_boost_rf_train_fit <- e1071::tune.randomForest(x = ensemble_train, y = ensemble_train$y_ensemble, mtry = ncol(ensemble_train)-1)
  ensemble_boost_rf_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble,
    predicted = predict(object = ensemble_boost_rf_train_fit$best.model,
    newdata = ensemble_train))
  ensemble_boost_rf_train_RMSE_mean <- mean(ensemble_boost_rf_train_RMSE)
  ensemble_boost_rf_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble,
    predicted = predict(object = ensemble_boost_rf_train_fit$best.model,
    newdata = ensemble_test))
  ensemble_boost_rf_test_RMSE_mean <- mean(ensemble_boost_rf_test_RMSE)
  ensemble_boost_rf_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble,
    predicted = predict(object = ensemble_boost_rf_train_fit$best.model,
    newdata = ensemble_validation))
  ensemble_boost_rf_validation_RMSE_mean <- mean(ensemble_boost_rf_validation_RMSE)
  ensemble_boost_rf_holdout_RMSE_mean <- mean(c(ensemble_boost_rf_test_RMSE_mean, ensemble_boost_rf_validation_RMSE_mean))
  ensemble_boost_rf_holdout_RMSE_sd_mean <- sd(c(ensemble_boost_rf_test_RMSE_mean, ensemble_boost_rf_validation_RMSE_mean))
  ensemble_boost_rf_train_predict_value <- as.numeric(predict(object = ensemble_boost_rf_train_fit$best.model, newdata = ensemble_train))
  ensemble_boost_rf_test_predict_value <- as.numeric(predict(object = ensemble_boost_rf_train_fit$best.model, newdata = ensemble_test))
  ensemble_boost_rf_validation_predict_value <- as.numeric(predict(object = ensemble_boost_rf_train_fit$best.model, newdata = ensemble_validation))
  ensemble_boost_rf_predict_value_mean <- mean(c(ensemble_boost_rf_test_predict_value, ensemble_boost_rf_validation_predict_value))
  ensemble_boost_rf_sd[i] <- sd(c(ensemble_boost_rf_test_predict_value, ensemble_boost_rf_validation_predict_value))
  ensemble_boost_rf_sd_mean <- mean(ensemble_boost_rf_sd)
  ensemble_boost_rf_overfitting[i] <- ensemble_boost_rf_holdout_RMSE_mean/ensemble_boost_rf_train_RMSE_mean
  ensemble_boost_rf_overfitting_mean <- mean(ensemble_boost_rf_overfitting)
  ensemble_boost_rf_overfitting_range <- range(ensemble_boost_rf_overfitting)
  ensemble_y_hat_boost_rf <- c(ensemble_boost_rf_test_predict_value, ensemble_boost_rf_validation_predict_value)
  ensemble_boost_rf_end <- Sys.time()
  ensemble_boost_rf_duration[i] <- ensemble_boost_rf_end - ensemble_boost_rf_start
  ensemble_boost_rf_duration_mean <- mean(ensemble_boost_rf_duration)
  
  
  #### Model 26: Ensemble Using Cubist ####
  ensemble_cubist_start <- Sys.time()
  ensemble_cubist_train_fit <- Cubist::cubist(x = ensemble_train[,1:ncol(ensemble_train)-1], y = ensemble_train$y_ensemble)
  ensemble_cubist_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_cubist_train_fit,
    newdata = ensemble_train))
  ensemble_cubist_train_RMSE_mean <- mean(ensemble_cubist_train_RMSE)
  ensemble_cubist_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_cubist_train_fit,
    newdata = ensemble_test))
  ensemble_cubist_test_RMSE_mean <- mean(ensemble_cubist_test_RMSE)
  ensemble_cubist_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_cubist_train_fit,
    newdata = ensemble_validation))
  ensemble_cubist_validation_RMSE_mean <- mean(ensemble_cubist_validation_RMSE)
  ensemble_cubist_holdout_RMSE_mean <- mean(c(ensemble_cubist_test_RMSE_mean, ensemble_cubist_validation_RMSE_mean))
  ensemble_cubist_holdout_RMSE_sd_mean <- sd(c(ensemble_cubist_test_RMSE_mean, ensemble_cubist_validation_RMSE_mean))
  ensemble_cubist_train_predict_value <- as.numeric(predict(object = ensemble_cubist_train_fit, newdata = ensemble_train))
  ensemble_cubist_test_predict_value <- as.numeric(predict(object = ensemble_cubist_train_fit, newdata = ensemble_test))
  ensemble_cubist_validation_predict_value <- as.numeric(predict(object = ensemble_cubist_train_fit, newdata = ensemble_validation))
  ensemble_cubist_predict_value_mean <- mean(c(ensemble_cubist_test_predict_value, ensemble_cubist_validation_predict_value))
  ensemble_cubist_sd[i] <- sd(c(ensemble_cubist_test_predict_value, ensemble_cubist_validation_predict_value))
  ensemble_cubist_sd_mean <- mean(ensemble_cubist_sd)
  ensemble_cubist_overfitting[i] <- ensemble_cubist_holdout_RMSE_mean/ensemble_cubist_train_RMSE_mean
  ensemble_cubist_overfitting_mean <- mean(ensemble_cubist_overfitting)
  ensemble_cubist_overfitting_range <- range(ensemble_cubist_overfitting)
  ensemble_y_hat_cubist <- c(ensemble_cubist_test_predict_value, ensemble_cubist_validation_predict_value)
  ensemble_cubist_end <- Sys.time()
  ensemble_cubist_duration[i] <- ensemble_cubist_end - ensemble_cubist_start
  ensemble_cubist_duration_mean <- mean(ensemble_cubist_duration)
  
  
  #### Model 27: Ensembles Using Earth ####
  ensemble_earth_start <- Sys.time()
  ensemble_earth_train_fit <- earth::earth(y_ensemble ~ ., data = ensemble_train)
  ensemble_earth_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_earth_train_fit,
    newdata = ensemble_train))
  ensemble_earth_train_RMSE_mean <- mean(ensemble_earth_train_RMSE)
  ensemble_earth_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_earth_train_fit,
    newdata = ensemble_test))
  ensemble_earth_test_RMSE_mean <- mean(ensemble_earth_test_RMSE)
  ensemble_earth_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_earth_train_fit,
    newdata = ensemble_validation))
  ensemble_earth_validation_RMSE_mean <- mean(ensemble_earth_validation_RMSE)
  ensemble_earth_holdout_RMSE_mean <- mean(c(ensemble_earth_test_RMSE_mean, ensemble_earth_validation_RMSE_mean))
  ensemble_earth_holdout_RMSE_sd_mean <- sd(c(ensemble_earth_test_RMSE_mean, ensemble_earth_validation_RMSE_mean))
  ensemble_earth_train_predict_value <- as.numeric(predict(object = ensemble_earth_train_fit, newdata = ensemble_train))
  ensemble_earth_test_predict_value <- as.numeric(predict(object = ensemble_earth_train_fit, newdata = ensemble_test))
  ensemble_earth_validation_predict_value <- as.numeric(predict(object = ensemble_earth_train_fit, newdata = ensemble_validation))
  ensemble_earth_predict_value_mean <- mean(c(ensemble_earth_test_predict_value, ensemble_earth_validation_predict_value))
  ensemble_earth_sd[i] <- sd(c(ensemble_earth_test_predict_value, ensemble_earth_validation_predict_value))
  ensemble_earth_sd_mean <- mean(ensemble_earth_sd)
  ensemble_earth_overfitting[i] <- ensemble_earth_holdout_RMSE_mean/ensemble_earth_train_RMSE_mean
  ensemble_earth_overfitting_mean <- mean(ensemble_earth_overfitting)
  ensemble_earth_overfitting_range <- range(ensemble_earth_overfitting)
  ensemble_y_hat_earth <- c(ensemble_earth_test_predict_value, ensemble_earth_validation_predict_value)
  ensemble_earth_end <- Sys.time()
  ensemble_earth_duration[i] <- ensemble_earth_end - ensemble_earth_start
  ensemble_earth_duration_mean <- mean(ensemble_earth_duration)
  
  
  #### Model 28: Ensembles Using GAM + Splines ####
  ensemble_gam_start <- Sys.time()
  ensemble_n_unique_vals <- map_dbl(ensemble, n_distinct)
  
  # # Names of columns with >= 4 unique vals
  ensemble_keep <- names(ensemble_n_unique_vals)[ensemble_n_unique_vals >= 4]
  
  ensemble_gam_data <- ensemble %>%
    dplyr::select(all_of(ensemble_keep))
  
  # Model data
  ensemble_train1 <- ensemble_train %>%
    dplyr::select(all_of(ensemble_keep))
  
  ensemble_test1 <- ensemble_test %>%
    dplyr::select(all_of(ensemble_keep))
  
  ensemble_validation1 <- ensemble_validation %>%
    dplyr::select(all_of(ensemble_keep))
  
  ensemble_names_df <- names(ensemble_gam_data[,1:ncol(ensemble_gam_data)-1])
  ensemble_f2 <- as.formula(paste0('y_ensemble ~', paste0('s(', ensemble_names_df, ')', collapse = '+')))
  ensemble_gam_train_fit <- gam::gam(ensemble_f2, data = ensemble_train1)
  ensemble_gam_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_gam_train_fit,
    newdata = ensemble_train))
  ensemble_gam_train_RMSE_mean <- mean(ensemble_gam_train_RMSE)
  ensemble_gam_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_gam_train_fit,
    newdata = ensemble_test))
  ensemble_gam_test_RMSE_mean <- mean(ensemble_gam_test_RMSE)
  ensemble_gam_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_gam_train_fit,
    newdata = ensemble_validation))
  ensemble_gam_validation_RMSE_mean <- mean(ensemble_gam_validation_RMSE)
  ensemble_gam_holdout_RMSE_mean <- mean(c(ensemble_gam_test_RMSE_mean, ensemble_gam_validation_RMSE_mean))
  ensemble_gam_holdout_RMSE_sd_mean <- sd(c(ensemble_gam_test_RMSE_mean, ensemble_gam_validation_RMSE_mean))
  ensemble_gam_train_predict_value <- as.numeric(predict(object = ensemble_gam_train_fit, newdata = ensemble_train))
  ensemble_gam_test_predict_value <- as.numeric(predict(object = ensemble_gam_train_fit, newdata = ensemble_test))
  ensemble_gam_validation_predict_value <- as.numeric(predict(object = ensemble_gam_train_fit, newdata = ensemble_validation))
  ensemble_gam_predict_value_mean <- mean(c(ensemble_gam_test_predict_value, ensemble_gam_validation_predict_value))
  ensemble_gam_sd[i] <- sd(c(ensemble_gam_test_predict_value, ensemble_gam_validation_predict_value))
  ensemble_gam_sd_mean <- mean(ensemble_gam_sd)
  ensemble_gam_overfitting[i] <- ensemble_gam_holdout_RMSE_mean/ensemble_gam_train_RMSE_mean
  ensemble_gam_overfitting_mean <- mean(ensemble_gam_overfitting)
  ensemble_gam_overfitting_range <- range(ensemble_gam_overfitting)
  ensemble_y_hat_gam <- c(ensemble_gam_test_predict_value, ensemble_gam_validation_predict_value)
  ensemble_gam_end <- Sys.time()
  ensemble_gam_duration[i] <- ensemble_gam_end - ensemble_gam_start
  ensemble_gam_duration_mean <- mean(ensemble_gam_duration)
  
  
  #### Model 29: Ensemble Gradient Boosted ####
  ensemble_gb_start <- Sys.time()
  ensemble_gb_train_fit <- gbm::gbm(ensemble_train$y_ensemble ~ ., data = ensemble_train, distribution = "gaussian",n.trees = 100,
    shrinkage = 0.1, interaction.depth = 10)
  ensemble_gb_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_gb_train_fit,
    newdata = ensemble_train))
  ensemble_gb_train_RMSE_mean <- mean(ensemble_gb_train_RMSE)
  ensemble_gb_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_gb_train_fit,
    newdata = ensemble_test))
  ensemble_gb_test_RMSE_mean <- mean(ensemble_gb_test_RMSE)
  ensemble_gb_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_gb_train_fit,
    newdata = ensemble_validation))
  ensemble_gb_validation_RMSE_mean <- mean(ensemble_gb_validation_RMSE)
  ensemble_gb_holdout_RMSE_mean <- mean(c(ensemble_gb_test_RMSE_mean, ensemble_gb_validation_RMSE_mean))
  ensemble_gb_holdout_RMSE_sd_mean <- sd(c(ensemble_gb_test_RMSE_mean, ensemble_gb_validation_RMSE_mean))
  ensemble_gb_train_predict_value <- as.numeric(predict(object = ensemble_gb_train_fit, newdata = ensemble_train))
  ensemble_gb_test_predict_value <- as.numeric(predict(object = ensemble_gb_train_fit, newdata = ensemble_test))
  ensemble_gb_validation_predict_value <- as.numeric(predict(object = ensemble_gb_train_fit, newdata = ensemble_validation))
  ensemble_gb_predict_value_mean <- mean(c(ensemble_gb_test_predict_value, ensemble_gb_validation_predict_value))
  ensemble_gb_sd[i] <- sd(c(ensemble_gb_test_predict_value, ensemble_gb_validation_predict_value))
  ensemble_gb_sd_mean <- mean(ensemble_gb_sd)
  ensemble_gb_overfitting[i] <- ensemble_gb_holdout_RMSE_mean/ensemble_gb_train_RMSE_mean
  ensemble_gb_overfitting_mean <- mean(ensemble_gb_overfitting)
  ensemble_gb_overfitting_range <- range(ensemble_gb_overfitting)
  ensemble_y_hat_gb <- c(ensemble_gb_test_predict_value, ensemble_gb_validation_predict_value)
  ensemble_gb_end <- Sys.time()
  ensemble_gb_duration[i] <- ensemble_gb_end - ensemble_gb_start
  ensemble_gb_duration_mean <- mean(ensemble_gb_duration)
  
  
  #### Model 30: Ensemble Using K-Nearest Neighbors ####
  ensemble_knn_start <- Sys.time()
  ensemble_knn_train_fit <- e1071::tune.gknn(x = ensemble_train[,1:ncol(ensemble_train)-1], y = ensemble_train$y_ensemble, scale = TRUE, k = c(1:25))
  ensemble_knn_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_knn_train_fit$best.model,
    newdata = ensemble_train[,1:ncol(ensemble_train)-1], k = ensemble_knn_train_fit$best.model$k))
  ensemble_knn_train_RMSE_mean <- mean(ensemble_knn_train_RMSE)
  ensemble_knn_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_knn_train_fit$best.model,
    k = ensemble_knn_train_fit$best.model$k, newdata = ensemble_test[,1:ncol(ensemble_test)-1]))
  ensemble_knn_test_RMSE_mean <- mean(ensemble_knn_test_RMSE)
  ensemble_knn_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_knn_train_fit$best.model,
    newdata = ensemble_validation[,1:ncol(ensemble_validation) -1], k = ensemble_knn_train_fit$best.model$k))
  ensemble_knn_validation_RMSE_mean <- mean(ensemble_knn_validation_RMSE)
  ensemble_knn_holdout_RMSE_mean <- mean(c(ensemble_knn_test_RMSE_mean, ensemble_knn_validation_RMSE_mean))
  ensemble_knn_holdout_RMSE_sd_mean <- sd(c(ensemble_knn_test_RMSE_mean, ensemble_knn_validation_RMSE_mean))
  ensemble_knn_train_predict_value <- as.numeric(predict(object = ensemble_knn_train_fit$best.model,
    newdata = ensemble_train[,1:ncol(ensemble_train)-1],
    k = which_min(ensemble_knn_RMSE_train_best)))
  ensemble_knn_test_predict_value <- as.numeric(predict(object = ensemble_knn_train_fit$best.model,
    newdata = ensemble_test[,1:ncol(ensemble_test)-1],
    k = which_min(ensemble_knn_RMSE_test_best)))
  ensemble_knn_validation_predict_value <- as.numeric(predict(object = ensemble_knn_train_fit$best.model,
    newdata = ensemble_validation[,1:ncol(ensemble_test)-1],
    k = which_min(ensemble_knn_RMSE_validation_best)))
  ensemble_knn_predict_value_mean <- mean(c(ensemble_knn_test_predict_value, ensemble_knn_validation_predict_value))
  ensemble_knn_sd[i] <- sd(ensemble_knn_test_predict_value)
  ensemble_knn_sd_mean <- mean(ensemble_knn_sd)
  ensemble_knn_overfitting[i] <- ensemble_knn_holdout_RMSE_mean/ensemble_knn_train_RMSE_mean
  ensemble_knn_overfitting_mean <- mean(ensemble_knn_overfitting)
  ensemble_knn_overfitting_range <- range(ensemble_knn_overfitting)
  ensemble_y_hat_knn <- c(ensemble_knn_test_predict_value, ensemble_knn_validation_predict_value)
  ensemble_knn_end <- Sys.time()
  ensemble_knn_duration[i] <- ensemble_knn_end - ensemble_knn_start
  ensemble_knn_duration_mean <- mean(ensemble_knn_duration)
  
  
  #### Model 31: Ensembles Using Linear tuned ####
  ensemble_linear_start <- Sys.time()
  ensemble_linear_train_fit <- e1071::tune.rpart(formula = y_ensemble ~ ., data = ensemble_train)
  ensemble_linear_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble,
    predicted = predict(object = ensemble_linear_train_fit$best.model,
    newdata = ensemble_train))
  ensemble_linear_train_RMSE_mean <- mean(ensemble_linear_train_RMSE)
  ensemble_linear_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble,
    predicted = predict(object = ensemble_linear_train_fit$best.model,
    newdata = ensemble_test))
  ensemble_linear_test_RMSE_mean <- mean(ensemble_linear_test_RMSE)
  ensemble_linear_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble,
    predicted = predict(object = ensemble_linear_train_fit$best.model,
    newdata = ensemble_validation))
  ensemble_linear_validation_RMSE_mean <- mean(ensemble_linear_validation_RMSE)
  ensemble_linear_holdout_RMSE_mean <- mean(c(ensemble_linear_test_RMSE_mean, ensemble_linear_validation_RMSE_mean))
  ensemble_linear_holdout_RMSE_sd_mean <- sd(c(ensemble_linear_test_RMSE_mean, ensemble_linear_validation_RMSE_mean))
  ensemble_linear_train_predict_value <- as.numeric(predict(object = ensemble_linear_train_fit$best.model, newdata = ensemble_train))
  ensemble_linear_test_predict_value <- as.numeric(predict(object = ensemble_linear_train_fit$best.model, newdata = ensemble_test))
  ensemble_linear_validation_predict_value <- as.numeric(predict(object = ensemble_linear_train_fit$best.model, newdata = ensemble_validation))
  ensemble_linear_predict_value_mean <- mean(c(ensemble_linear_test_predict_value, ensemble_linear_validation_predict_value))
  ensemble_linear_sd[i] <- sd(c(ensemble_linear_test_predict_value, ensemble_linear_validation_predict_value))
  ensemble_linear_sd_mean <- mean(ensemble_linear_sd)
  ensemble_linear_overfitting[i] <- ensemble_linear_holdout_RMSE_mean/ensemble_linear_train_RMSE_mean
  ensemble_linear_overfitting_mean <- mean(ensemble_linear_overfitting)
  ensemble_linear_overfitting_range <- range(ensemble_linear_overfitting)
  ensemble_y_hat_linear <- c(ensemble_linear_test_predict_value, ensemble_linear_validation_predict_value)
  ensemble_linear_end <- Sys.time()
  ensemble_linear_duration[i] <- ensemble_linear_end - ensemble_linear_start
  ensemble_linear_duration_mean <- mean(ensemble_linear_duration)
  
  
  #### Model 32: Ensembles Using Partial Least Squares Regression (PLS) ####
  ensemble_pls_start <- Sys.time()
  ensemble_pls_train_fit <- pls::plsr(ensemble_train$y_ensemble ~ ., data = ensemble_train)
  ensemble_pls_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_pls_train_fit,
    newdata = ensemble_train))
  ensemble_pls_train_RMSE_mean <- mean(ensemble_pls_train_RMSE)
  ensemble_pls_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_pls_train_fit,
    newdata = ensemble_test))
  ensemble_pls_test_RMSE_mean <- mean(ensemble_pls_test_RMSE)
  ensemble_pls_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_pls_train_fit,
    newdata = ensemble_validation))
  ensemble_pls_validation_RMSE_mean <- mean(ensemble_pls_validation_RMSE)
  ensemble_pls_holdout_RMSE_mean <- mean(c(ensemble_pls_test_RMSE_mean, ensemble_pls_validation_RMSE_mean))
  ensemble_pls_holdout_RMSE_sd_mean <- sd(c(ensemble_pls_test_RMSE_mean, ensemble_pls_validation_RMSE_mean))
  ensemble_pls_train_predict_value <- predict(object = ensemble_pls_train_fit, newdata = ensemble_train)
  ensemble_pls_test_predict_value <- predict(object = ensemble_pls_train_fit, newdata = ensemble_test)
  ensemble_pls_validation_predict_value <- predict(object = ensemble_pls_train_fit, newdata = ensemble_validation)
  ensemble_pls_predict_value_mean <- mean(c(ensemble_pls_test_predict_value, ensemble_pls_validation_predict_value))
  ensemble_pls_sd[i] <- sd(ensemble_pls_test_predict_value)
  ensemble_pls_sd_mean <- mean(ensemble_pls_sd)
  ensemble_pls_overfitting[i] <- ensemble_pls_holdout_RMSE_mean/ensemble_pls_train_RMSE_mean
  ensemble_pls_overfitting_mean <- mean(ensemble_pls_overfitting)
  ensemble_pls_overfitting_range <- range(ensemble_pls_overfitting)
  ensemble_y_hat_pls <- c(ensemble_pls_test_predict_value[,,1], ensemble_pls_validation_predict_value[,,1])
  ensemble_pls_end <- Sys.time()
  ensemble_pls_duration[i] <- ensemble_pls_end - ensemble_pls_start
  ensemble_pls_duration_mean <- mean(ensemble_pls_duration)
  
  
  #### Model 33: Ensembles Using Principle Components Analysis ####
  ensemble_pcr_start <- Sys.time()
  ensemble_pcr_train_fit <- pls::pcr(ensemble_train$y_ensemble ~ ., data = ensemble_train)
  ensemble_pcr_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_pcr_train_fit,
    newdata = ensemble_train))
  ensemble_pcr_train_RMSE_mean <- mean(ensemble_pcr_train_RMSE)
  ensemble_pcr_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_pcr_train_fit,
    newdata = ensemble_test))
  ensemble_pcr_test_RMSE_mean <- mean(ensemble_pcr_test_RMSE)
  ensemble_pcr_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_pcr_train_fit,
    newdata = ensemble_validation))
  ensemble_pcr_validation_RMSE_mean <- mean(ensemble_pcr_validation_RMSE)
  ensemble_pcr_holdout_RMSE_mean <- mean(c(ensemble_pcr_test_RMSE_mean, ensemble_pcr_validation_RMSE_mean))
  ensemble_pcr_holdout_RMSE_sd_mean <- sd(c(ensemble_pcr_test_RMSE_mean, ensemble_pcr_validation_RMSE_mean))
  ensemble_pcr_train_predict_value <- predict(object = pls::pcr(y_ensemble ~ ., data = ensemble_train), newdata = ensemble_train)
  ensemble_pcr_test_predict_value <- predict(object = pls::pcr(y_ensemble ~ ., data = ensemble_train), newdata = ensemble_test)
  ensemble_pcr_validation_predict_value <- predict(object = pls::pcr(y_ensemble ~ ., data = ensemble_train), newdata = ensemble_validation)
  ensemble_pcr_predict_value_mean <- mean(c(ensemble_pcr_test_predict_value, ensemble_pcr_validation_predict_value))
  ensemble_pcr_sd_mean <- sd(ensemble_pcr_test_predict_value)
  ensemble_pcr_overfitting[i] <- ensemble_pcr_holdout_RMSE_mean/ensemble_pcr_train_RMSE_mean
  ensemble_pcr_overfitting_mean <- mean(ensemble_pcr_overfitting)
  ensemble_pcr_overfitting_range <- range(ensemble_pcr_overfitting)
  ensemble_y_hat_pcr <- c(ensemble_pcr_test_predict_value[,,1], ensemble_pcr_validation_predict_value[,,1])
  ensemble_pcr_end <- Sys.time()
  ensemble_pcr_duration[i] <- ensemble_pcr_end - ensemble_pcr_start
  ensemble_pcr_duration_mean <- mean(ensemble_pcr_duration)
  
  
  #### Model 34: Ensembles Using Random Forest tuned ####
  ensemble_rf_start <- Sys.time()
  ensemble_rf_train_fit <- tune.randomForest(x = ensemble_train, y = ensemble_train$y_ensemble, data = ensemble_train)
  ensemble_rf_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble,
    predicted = predict(object = ensemble_rf_train_fit$best.model,
    newdata = ensemble_train))
  ensemble_rf_train_RMSE_mean <- mean(ensemble_rf_train_RMSE)
  ensemble_rf_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble,
    predicted = predict(object = ensemble_rf_train_fit$best.model,
    newdata = ensemble_test))
  ensemble_rf_test_RMSE_mean <- mean(ensemble_rf_test_RMSE)
  ensemble_rf_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble,
    predicted = predict(object = ensemble_rf_train_fit$best.model,
    newdata = ensemble_validation))
  ensemble_rf_validation_RMSE_mean <- mean(ensemble_rf_validation_RMSE)
  ensemble_rf_holdout_RMSE_mean <- mean(c(ensemble_rf_test_RMSE_mean, ensemble_rf_validation_RMSE_mean))
  ensemble_rf_holdout_RMSE_sd_mean <- sd(c(ensemble_rf_test_RMSE_mean, ensemble_rf_validation_RMSE_mean))
  ensemble_rf_train_predict_value <- predict(object = ensemble_rf_train_fit$best.model, newdata = ensemble_train)
  ensemble_rf_test_predict_value <- predict(object = ensemble_rf_train_fit$best.model, newdata = ensemble_test)
  ensemble_rf_validation_predict_value <- predict(object = ensemble_rf_train_fit$best.model, newdata = ensemble_validation)
  ensemble_rf_predict_value_mean <- mean(c(ensemble_rf_test_predict_value, ensemble_rf_validation_predict_value))
  ensemble_rf_sd_mean <- sd(c(ensemble_rf_test_predict_value, ensemble_rf_validation_predict_value))
  ensemble_rf_overfitting[i] <- ensemble_rf_holdout_RMSE_mean/ensemble_rf_train_RMSE_mean
  ensemble_rf_overfitting_mean <- mean(ensemble_rf_overfitting)
  ensemble_rf_overfitting_range <- range(ensemble_rf_overfitting)
  ensemble_y_hat_rf <- c(ensemble_rf_test_predict_value, ensemble_rf_validation_predict_value)
  ensemble_rf_end <- Sys.time()
  ensemble_rf_duration[i] <- ensemble_rf_end - ensemble_rf_start
  ensemble_rf_duration_mean <- mean(ensemble_rf_duration)
  
  #### Model #35: Ensembles Using Rpart ####
  ensemble_rpart_start <- Sys.time()
  ensemble_rpart_train_fit <- rpart::rpart(ensemble_train$y_ensemble ~ ., data = ensemble_train)
  ensemble_rpart_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_rpart_train_fit,
    newdata = ensemble_train))
  ensemble_rpart_train_RMSE_mean <- mean(ensemble_rpart_train_RMSE)
  ensemble_rpart_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_rpart_train_fit,
    newdata = ensemble_test))
  ensemble_rpart_test_RMSE_mean <- mean(ensemble_rpart_test_RMSE)
  ensemble_rpart_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_rpart_train_fit,
    newdata = ensemble_validation))
  ensemble_rpart_validation_RMSE_mean <- mean(ensemble_rpart_validation_RMSE)
  ensemble_rpart_holdout_RMSE_mean <- mean(c(ensemble_rpart_test_RMSE_mean, ensemble_rpart_validation_RMSE_mean))
  ensemble_rpart_holdout_RMSE_sd_mean <- sd(c(ensemble_rpart_test_RMSE_mean, ensemble_rpart_validation_RMSE_mean))
  ensemble_rpart_train_predict_value <- as.numeric(predict(object = rpart::rpart(y_ensemble ~ .,data = ensemble_train),
    newdata = ensemble_train))
  ensemble_rpart_test_predict_value <- as.numeric(predict(object = rpart::rpart(y_ensemble ~ .,data = ensemble_train),
    newdata = ensemble_test))
  ensemble_rpart_validation_predict_value <- as.numeric(predict(object = rpart::rpart(y_ensemble ~ .,data = ensemble_train),
    newdata = ensemble_validation))
  ensemble_rpart_predict_value_mean <- mean(c(ensemble_rpart_test_predict_value, ensemble_rpart_validation_predict_value))
  ensemble_rpart_sd[i] <- sd(c(ensemble_rpart_test_predict_value, ensemble_rpart_validation_predict_value))
  ensemble_rpart_sd_mean <- mean(ensemble_rpart_sd)
  ensemble_rpart_overfitting[i] <- ensemble_rpart_holdout_RMSE_mean/ensemble_rpart_train_RMSE_mean
  ensemble_rpart_overfitting_mean <- mean(ensemble_rpart_overfitting)
  ensemble_rpart_overfitting_range <- range(ensemble_rpart_overfitting)
  ensemble_y_hat_rpart <- c(ensemble_rpart_test_predict_value, ensemble_rpart_validation_predict_value)
  ensemble_rpart_end <- Sys.time()
  ensemble_rpart_duration[i] <- ensemble_rpart_end - ensemble_rpart_start
  ensemble_rpart_duration_mean <- mean(ensemble_rpart_duration)
  
  
  #### Model #36: Ensemble Using Support Vector Machines (SVM) tuned ####
  ensemble_svm_start <- Sys.time()
  ensemble_svm_train_fit <- e1071::tune.svm(x = ensemble_train, y = ensemble_train$y_ensemble, data = ensemble_train)
  ensemble_svm_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble,
    predicted = predict(object = ensemble_svm_train_fit$best.model,
    newdata = ensemble_train))
  ensemble_svm_train_RMSE_mean <- mean(ensemble_svm_train_RMSE)
  ensemble_svm_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble,
    predicted = predict(object = ensemble_svm_train_fit$best.model,
    newdata = ensemble_test))
  ensemble_svm_test_RMSE_mean <- mean(ensemble_svm_test_RMSE)
  ensemble_svm_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble,
    predicted = predict(object = ensemble_svm_train_fit$best.model,
    newdata = ensemble_test))
  ensemble_svm_test_RMSE_mean <- mean(ensemble_svm_test_RMSE)
  ensemble_svm_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble,
    predicted = predict(object = ensemble_svm_train_fit$best.model,
    newdata = ensemble_validation))
  ensemble_svm_validation_RMSE_mean <- mean(ensemble_svm_validation_RMSE)
  ensemble_svm_holdout_RMSE_mean <- mean(c(ensemble_svm_test_RMSE_mean, ensemble_svm_validation_RMSE_mean))
  ensemble_svm_holdout_RMSE_sd_mean <- sd(c(ensemble_svm_test_RMSE_mean, ensemble_svm_validation_RMSE_mean))
  ensemble_svm_train_predict_value <- predict(object = ensemble_svm_train_fit$best.model, newdata = ensemble_train)
  ensemble_svm_test_predict_value <- predict(object = ensemble_svm_train_fit$best.model, newdata = ensemble_test)
  ensemble_svm_validation_predict_value <- predict(object = ensemble_svm_train_fit$best.model, newdata = ensemble_validation)
  ensemble_svm_predict_value_mean <- mean(c(ensemble_svm_test_predict_value, ensemble_svm_validation_predict_value))
  ensemble_svm_sd_mean <- sd(c(ensemble_svm_test_predict_value, ensemble_svm_validation_predict_value))
  ensemble_svm_overfitting[i] <- ensemble_svm_holdout_RMSE_mean/ensemble_svm_train_RMSE_mean
  ensemble_svm_overfitting_mean <- mean(ensemble_svm_overfitting)
  ensemble_svm_overfitting_range <- range(ensemble_svm_overfitting)
  ensemble_y_hat_svm <- c(ensemble_svm_test_predict_value, ensemble_svm_validation_predict_value)
  ensemble_svm_end <- Sys.time()
  ensemble_svm_duration[i] <- ensemble_svm_end - ensemble_svm_start
  ensemble_svm_duration_mean <- mean(ensemble_svm_duration)
  
  
  #### Model 37: Ensemble Using Trees ####
  ensemble_tree_start <- Sys.time()
  ensemble_tree_train_fit <- tree::tree(ensemble_train$y_ensemble ~ ., data = ensemble_train)
  ensemble_tree_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_tree_train_fit,
    newdata = ensemble_train))
  ensemble_tree_train_RMSE_mean <- mean(ensemble_tree_train_RMSE)
  ensemble_tree_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_tree_train_fit,
    newdata = ensemble_test))
  ensemble_tree_test_RMSE_mean <- mean(ensemble_tree_test_RMSE)
  ensemble_tree_validation_RMSE[i] = Metrics::rmse(actual = ensemble_validation$y_ensemble, predicted = predict(object = ensemble_tree_train_fit,
    newdata = ensemble_validation))
  ensemble_tree_validation_RMSE_mean <- mean(ensemble_tree_validation_RMSE)
  ensemble_tree_holdout_RMSE_mean <- mean(c(ensemble_tree_test_RMSE_mean, ensemble_tree_validation_RMSE_mean))
  ensemble_tree_holdout_RMSE_sd_mean <- sd(c(ensemble_tree_test_RMSE_mean, ensemble_tree_validation_RMSE_mean))
  ensemble_tree_train_predict_value <- predict(object = ensemble_tree_train_fit, newdata = ensemble_train)
  ensemble_tree_test_predict_value <- predict(object = ensemble_tree_train_fit, newdata = ensemble_test)
  ensemble_tree_validation_predict_value <- predict(object = ensemble_tree_train_fit, newdata = ensemble_validation)
  ensemble_tree_predict_value_mean <- mean(c(ensemble_tree_test_predict_value, ensemble_tree_validation_predict_value))
  ensemble_tree_sd_mean <- sd(ensemble_tree_test_predict_value)
  ensemble_tree_overfitting[i] <- ensemble_tree_holdout_RMSE_mean/ensemble_tree_train_RMSE_mean
  ensemble_tree_overfitting_mean <- mean(ensemble_tree_overfitting)
  ensemble_tree_overfitting_range <- range(ensemble_tree_overfitting)
  ensemble_y_hat_tree <- c(ensemble_tree_test_predict_value, ensemble_tree_validation_predict_value)
  ensemble_tree_end <- Sys.time()
  ensemble_tree_duration[i] <- ensemble_tree_end - ensemble_tree_start
  ensemble_tree_duration_mean <- mean(ensemble_tree_duration)
  
  
  #### Model 38: Ensembles Using XGBoost ####
  ensemble_xgb_start <- Sys.time()
  train_x = data.matrix(ensemble_train[, -ncol(ensemble_train)])
  train_y = ensemble_train[,ncol(ensemble_train)]
  
  #define predictor and response variables in test set
  test_x = data.matrix(ensemble_test[, -ncol(ensemble_test)])
  test_y = data.matrix(ensemble_test[, ncol(ensemble_test)])
  
  #define predictor and response variables in validation set
  validation_x = data.matrix(ensemble_validation[, -ncol(ensemble_validation)])
  validation_y = ensemble_validation[, ncol(ensemble_validation)]
  
  #define final train, test and validationing sets
  xgb_train = xgboost::xgb.DMatrix(data = train_x, label = train_y)
  xgb_test <- xgboost::xgb.DMatrix(data = test_x, label = test_y)
  xgb_validation = xgboost::xgb.DMatrix(data = validation_x, label = validation_y)
  
  #define watchlist
  watchlist = list(train = xgb_train, validation = xgb_validation)
  watchlist_test <- list(train = xgb_train, test = xgb_test)
  watchlist_validation <- list(train = xgb_train, validation = xgb_validation)
  
  #fit XGBoost model and display training and validation data at each round
  
  xgb_model_train = xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist, nrounds = 70)
  xgb_model_test = xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist_test, nrounds = 70)
  xgb_model_validation = xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist_validation, nrounds = 70)
  
  xgboost_min <- which.min(xgb_model$evaluation_log$validation_rmse)
  xgboost_validation_min <- which.min(xgb_model$evaluation_log$validation_rmse)
  
  ensemble_xgb_train_RMSE[i] <- xgb_model_train$evaluation_log$train_rmse[length(xgb_model_train$evaluation_log$train_rmse)]
  ensemble_xgb_train_RMSE_mean <- mean(ensemble_xgb_train_RMSE)
  
  ensemble_xgb_test_RMSE[i] <- xgb_model_test$evaluation_log$test_rmse[length(xgb_model_train$evaluation_log$train_rmse)]
  ensemble_xgb_test_RMSE_mean <- mean(ensemble_xgb_test_RMSE)
  
  ensemble_xgb_validation_RMSE[i] <- xgb_model_validation$evaluation_log$validation_rmse[length(xgb_model_validation$evaluation_log$validation_rmse)]
  ensemble_xgb_validation_RMSE_mean <- mean(ensemble_xgb_validation_RMSE)
  
  ensemble_xgb_holdout_RMSE_mean <- mean(c(xgb_test_RMSE_mean, xgb_validation_RMSE_mean))
  ensemble_xgb_holdout_RMSE_sd_mean <- mean(sd(c(xgb_test_RMSE_mean, xgb_validation_RMSE_mean)))
  
  ensemble_y_hat_xgb <- c(predict(object = xgb_model_test, newdata = test_x), predict(object = xgb_model_validation, newdata = validation_x))
  ensemble_xgb_mean <- round(mean(ensemble_y_hat_xgb), 4)
  ensemble_xgb_sd <- round(sd(ensemble_y_hat_xgb), 4)
  
  ensemble_xgb_overfitting[i] <- ensemble_xgb_holdout_RMSE_mean/ensemble_xgb_train_RMSE_mean
  ensemble_xgb_overfitting_mean <- mean(ensemble_xgb_overfitting)
  ensemble_xgb_overfitting_range <- range(ensemble_xgb_overfitting)
  ensemble_xgb_end <- Sys.time()
  ensemble_xgb_duration[i] <- ensemble_xgb_end - ensemble_xgb_start
  ensemble_xgb_duration_mean <- mean(ensemble_xgb_duration)
  
} # end of loops for simulations from current data

#### <-----------------------------------------  8. Summary data visualizations ----------------------------------------------------> ####

#### Bagged random forest visualizations ####
bag_rf_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_bag_rf),
  residuals = c(test$medv, validation$medv) - y_hat_bag_rf)

bag_rf_pred_vs_actual <- ggplot(bag_rf_df, mapping = aes(x = as.numeric(y_hat_bag_rf), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Bagged Random Forest model: Predicted vs actual", x = "Predicted", y = "Actual")

bag_rf_resid_vs_actual <- ggplot(bag_rf_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Bagged Random Forest model: Residuals", x = "Residuals", y = "Actual")

bag_rf_qq <- ggplot(bag_rf_df, aes(sample = as.numeric(y_hat_bag_rf))) + stat_qq() +
  labs(title = "Bagged Random forest model: Q-Q plot") +
  stat_qq_line(color = "red")

bag_rf_hist_residuals <- ggplot(bag_rf_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Bagged Random Forest model: Histogram of residuals, each bar = 10 rows of data_")

#### Bayes Generalized Linear Models visualizations ####
bag_rf_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_bag_rf),
  residuals = c(test$medv, validation$medv) - y_hat_bag_rf)

bayesglm_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_bayesglm),
  residuals = c(test$medv, validation$medv) - y_hat_bayesglm)

bayesglm_pred_vs_actual <- ggplot(bayesglm_df, mapping = aes(x = as.numeric(y_hat_bayesglm), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "BayesGLM model: Predicted vs actual", x = "Predicted", y = "Actual")

bayesglm_resid_vs_actual <- ggplot(bayesglm_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "BayesGLM model: Residuals", x = "Residuals", y = "Actual")

bayesglm_qq <- ggplot(bayesglm_df, aes(sample = as.numeric(y_hat_bayesglm))) + stat_qq() +
  labs(title = "BayesGLM model: Q-Q plot") +
  stat_qq_line(color = "red")

bayesglm_hist_residuals <- ggplot(bayesglm_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "BayesGLM model: Histotram of residuals, each bar = 10 rows of data_")

#### Bayes RNN visualizations ##
bayesrnn_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_bayesrnn),
  residuals = c(test$medv, validation$medv) - y_hat_bayesrnn)

bayesrnn_pred_vs_actual <- ggplot(bayesrnn_df, mapping = aes(x = as.numeric(y_hat_bayesrnn), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "BayesRNN model: Predicted vs actual", x = "Predicted", y = "Actual")

bayesrnn_resid_vs_actual <- ggplot(bayesrnn_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "BayesRNN Forest model: Residuals", x = "Residuals", y = "Actual")

bayesrnn_qq <- ggplot(bayesrnn_df, aes(sample = as.numeric(y_hat_bayesrnn))) + stat_qq() +
  labs(title = "BayesRNN model: Q-Q plot") +
  stat_qq_line(color = "red")

bayesrnn_hist_residuals <- ggplot(bayesrnn_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "BayesRNN model: Histotram of residuals, each bar = 10 rows of data_")


### Boosted random forest visualizations ####
boost_rf_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_boost_rf),
  residuals = c(test$medv, validation$medv) - y_hat_boost_rf)

boost_rf_pred_vs_actual <- ggplot(boost_rf_df, mapping = aes(x = as.numeric(y_hat_boost_rf), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Boosted Random Forest model: Predicted vs actual", x = "Predicted", y = "Actual")

boost_rf_resid_vs_actual <- ggplot(boost_rf_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Boosted Random Forest model: Residuals", x = "Residuals", y = "Actual")

boost_rf_qq <- ggplot(boost_rf_df, aes(sample = as.numeric(y_hat_boost_rf))) + stat_qq() +
  labs(title = "Boosted Random forest model: Q-Q plot") +
  stat_qq_line(color = "red")

boost_rf_hist_residuals <- ggplot(boost_rf_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Boosted Random Forest model: Histotram of residuals, each bar = 10 rows of data_")


#### Cubist visualizations ####
cubist_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_cubist),
  residuals = c(test$medv, validation$medv) - y_hat_cubist)

cubist_pred_vs_actual <- ggplot(cubist_df, mapping = aes(x = as.numeric(y_hat_cubist), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Cubist model: Predicted vs actual", x = "Predicted", y = "Actual")

cubist_resid_vs_actual <- ggplot(cubist_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Cubist model: Residuals", x = "Residuals", y = "Actual")

cubist_qq <- ggplot(cubist_df, aes(sample = as.numeric(y_hat_cubist))) + stat_qq() +
  labs(title = "Cubist model: Q-Q plot") +
  stat_qq_line(color = "red")

cubist_hist_residuals <- ggplot(cubist_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Cubist model: Histotram of residuals, each bar = 10 rows of data_")


####  Earth visualizations ####
earth_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_earth),
  residuals = c(test$medv, validation$medv) - y_hat_earth)

earth_pred_vs_actual <- ggplot(earth_df, mapping = aes(x = as.numeric(y_hat_earth), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Earth model: Predicted vs actual", x = "Predicted", y = "Actual")

earth_resid_vs_actual <- ggplot(earth_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Earth model: Residuals", x = "Residuals", y = "Actual")

earth_qq <- ggplot(earth_df, aes(sample = as.numeric(y_hat_earth))) + stat_qq() +
  labs(title = "Earth model: Q-Q plot") +
  stat_qq_line(color = "red")

earth_hist_residuals <- ggplot(earth_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Earth model: Histotram of residuals, each bar = 10 rows of data_")

####  Gradient Boosted visualizations ####
gb_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_gb),
  residuals = c(test$medv, validation$medv) - y_hat_gb)

gb_pred_vs_actual <- ggplot(gb_df, mapping = aes(x = as.numeric(y_hat_gb), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Gradient Boosted Forest model: Predicted vs actual", x = "Predicted", y = "Actual")

gb_resid_vs_actual <- ggplot(gb_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Gradient Boosted Forest model: Residuals", x = "Residuals", y = "Actual")

gb_qq <- ggplot(gb_df, aes(sample = as.numeric(y_hat_gb))) + stat_qq() +
  labs(title = "Gradient Boosted forest model: Q-Q plot") +
  stat_qq_line(color = "red")

gb_hist_residuals <- ggplot(gb_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Gradient Boosted Forest model: Histogram of residuals, each bar = 10 rows of data_")


#### Generalized Additive Models using splines data visualizations ####
gam_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_gam),
  residuals = c(test$medv, validation$medv) - y_hat_gam)

gam_pred_vs_actual <- ggplot(gam_df, mapping = aes(x = as.numeric(y_hat_gam), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Generalized Additive model using smoothing splines: Predicted vs actual", x = "Predicted", y = "Actual")

gam_resid_vs_actual <- ggplot(gam_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Generalized Additive model using smoothing splines: Residuals", x = "Residuals", y = "Actual")

gam_qq <- ggplot(gam_df, aes(sample = as.numeric(y_hat_gam))) + stat_qq() +
  labs(title = "Generalized Additive model using smoothing splines: Q-Q plot") +
  stat_qq_line(color = "red")

gam_hist_residuals <- ggplot(gam_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Generalized Additve model using smoothing splines: Histotram of residuals, each bar = 10 rows of data_")


#### bagging data visualizations ####
bagging_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_bagging),
  residuals = c(test$medv, validation$medv) - y_hat_bagging)

bagging_pred_vs_actual <- ggplot(bagging_df, mapping = aes(x = as.numeric(y_hat_bagging), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "bagging model: Predicted vs actual", x = "Predicted", y = "Actual")

bagging_resid_vs_actual <- ggplot(bagging_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "bagging model: Residuals", x = "Residuals", y = "Actual")

bagging_qq <- ggplot(bagging_df, aes(sample = as.numeric(y_hat_bagging))) + stat_qq() +
  labs(title = "bagging model: Q-Q plot") +
  stat_qq_line(color = "red")

bagging_hist_residuals <- ggplot(bagging_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "bagging model: Histogram of residuals, each bar = 10 rows of data_")


#### K-Nearest Neighboars visualizations ####
knn_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_knn), residuals = c(test$medv, validation$medv) - y_hat_knn)

knn_pred_vs_actual <- ggplot(knn_df, mapping = aes(x = as.numeric(y_hat_knn), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "K-Nearst Neighbors model: Predicted vs actual", x = "Predicted", y = "Actual")

knn_resid_vs_actual <- ggplot(knn_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "K-Nearst Neighbors model: Residuals", x = "Residuals", y = "Actual")

knn_qq <- ggplot(knn_df, aes(sample = as.numeric(y_hat_knn))) + stat_qq() +
  labs(title = "K-Nearst Neighbors model: Q-Q plot") +
  stat_qq_line(color = "red")

knn_hist_residuals <- ggplot(knn_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "K-Nearst Neighbors model: Histogram of residuals, each bar = 10 rows of data_")

####  Linear visualizations ####
linear_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_linear),
  residuals = c(test$medv, validation$medv) - y_hat_linear)

linear_pred_vs_actual <- ggplot(linear_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "linear model: Predicted vs actual", x = "Predicted", y = "Actual")

linear_resid_vs_actual <- ggplot(linear_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "linear model: Residuals", x = "Residuals", y = "Actual")

linear_qq <- ggplot(linear_df, aes(sample = as.numeric(y_hat_linear))) + stat_qq() +
  labs(title = "linear model: Q-Q plot") +
  stat_qq_line(color = "red")

linear_hist_residuals <- ggplot(linear_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "linear model: Histogram of residuals, each bar = 10 rows of data_")

####  lqs visualizations ####
lqs_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_lqs), residuals = c(test$medv, validation$medv) - y_hat_lqs)

lqs_pred_vs_actual <- ggplot(lqs_df, mapping = aes(x = as.numeric(y_hat_lqs), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "lqs model: Predicted vs actual", x = "Predicted", y = "Actual")

lqs_resid_vs_actual <- ggplot(lqs_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "lqs model: Residuals", x = "Residuals", y = "Actual")

lqs_qq <- ggplot(lqs_df, aes(sample = as.numeric(y_hat_lqs))) + stat_qq() +
  labs(title = "lqs model: Q-Q plot") +
  stat_qq_line(color = "red")

lqs_hist_residuals <- ggplot(lqs_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "lqs model: Histogram of residuals, each bar = 10 rows of data_")


####  Partial Least Squares visualizartions ####
pls_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_pls), residuals = c(test$medv, validation$medv) - y_hat_pls)

pls_pred_vs_actual <- ggplot(pls_df, mapping = aes(x = as.numeric(y_hat_pls), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Partial Least Squares model: Predicted vs actual", x = "Predicted", y = "Actual")

pls_resid_vs_actual <- ggplot(pls_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Partial Least Squares model: Residuals", x = "Residuals", y = "Actual")

pls_qq <- ggplot(pls_df, aes(sample = as.numeric(y_hat_pls))) + stat_qq() +
  labs(title = "Partial Least Squares model: Q-Q plot") +
  stat_qq_line(color = "red")

pls_hist_residuals <- ggplot(pls_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Partial Least Squares model: Histogram of residuals, each bar = 10 rows of data_")


####  Principle Components Regression visualizations ####
pcr_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_pcr), residuals = c(test$medv, validation$medv) - y_hat_pcr)

pcr_pred_vs_actual <- ggplot(pcr_df, mapping = aes(x = as.numeric(y_hat_pcr), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Principal Components Regression model: Predicted vs actual", x = "Predicted", y = "Actual")

pcr_resid_vs_actual <- ggplot(pcr_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Principal Components Regression model: Residuals", x = "Residuals", y = "Actual")

pcr_qq <- ggplot(pcr_df, aes(sample = as.numeric(y_hat_pcr))) + stat_qq() +
  labs(title = "Principal Components Regression model: Q-Q plot") +
  stat_qq_line(color = "red")

pcr_hist_residuals <- ggplot(pcr_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Principal Components Regression model: Histogram of residuals, each bar = 10 rows of data_")


####  Random Forest visualizations ####
rf_df <- data.frame(actual =c(test$medv, validation$medv), predicted = as.numeric(y_hat_rf), residuals = c(test$medv, validation$medv) - y_hat_rf)

rf_pred_vs_actual <- ggplot(rf_df, mapping = aes(x = as.numeric(y_hat_rf), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Random Forest model: Predicted vs actual", x = "Predicted", y = "Actual")

rf_resid_vs_actual <- ggplot(rf_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Random Forest model: Residuals vs_ Actuals", x = "Residuals", y = "Actual")

rf_qq <- ggplot(rf_df, aes(sample = as.numeric(y_hat_rf))) + stat_qq() +
  labs(title = "Random Forest model: Q-Q plot") +
  stat_qq_line(color = "red")

rf_hist_residuals <- ggplot(rf_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Random Forest model: Histogram of residuals, each bar = 10 rows of data_")


####  Robust Regression visualizations ####
robust_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_robust),
  residuals = c(test$medv, validation$medv) - y_hat_robust)

robust_pred_vs_actual <- ggplot(robust_df, mapping = aes(x = as.numeric(y_hat_robust), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Robust model: Predicted vs actual", x = "Predicted", y = "Actual")

robust_resid_vs_actual <- ggplot(robust_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Robust model: Residuals", x = "Residuals", y = "Actual")

robust_qq <- ggplot(robust_df, aes(sample = as.numeric(y_hat_robust))) + stat_qq() +
  labs(title = "Robust model: Q-Q plot") +
  stat_qq_line(color = "red")

robust_hist_residuals <- ggplot(robust_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Robust model: Histogram of residuals, each bar = 10 rows of data_")


#### Rpart (also known as Cart) visualizations ####
rpart_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_rpart),
  residuals = c(test$medv, validation$medv) - y_hat_rpart)

rpart_pred_vs_actual <- ggplot(rpart_df, mapping = aes(x = as.numeric(y_hat_rpart), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Rpart model: Predicted vs actual", x = "Predicted", y = "Actual")

rpart_resid_vs_actual <- ggplot(rpart_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Rpart model: Residuals", x = "Residuals", y = "Actual")

rpart_qq <- ggplot(rpart_df, aes(sample = as.numeric(y_hat_rpart))) + stat_qq() +
  labs(title = "Rpart model: Q-Q plot") +
  stat_qq_line(color = "red")

rpart_hist_residuals <- ggplot(rpart_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Rpart model: Histotram of residuals, each bar = 10 rows of data_")


####  Support Vector Machines visualizations ####
svm_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_svm), residuals = c(test$medv, validation$medv) - y_hat_svm)

svm_pred_vs_actual <- ggplot(svm_df, mapping = aes(x = as.numeric(y_hat_svm), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Support Vector Machines model: Predicted vs actual", x = "Predicted", y = "Actual")

svm_resid_vs_actual <- ggplot(svm_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Support Vector Machines model: Residuals", x = "Residuals", y = "Actual")

svm_qq <- ggplot(svm_df, aes(sample = as.numeric(y_hat_svm))) + stat_qq() +
  labs(title = "Support Vector Machines model: Q-Q plot") +
  stat_qq_line(color = "red")

svm_hist_residuals <- ggplot(svm_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Suppoer Vector Machines model: Histogram of residuals, each bar = 10 rows of data_")


####  Trees visualizations ####
tree_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_tree), residuals = c(test$medv, validation$medv) - y_hat_tree)

tree_pred_vs_actual <- ggplot(tree_df, mapping = aes(x = as.numeric(y_hat_tree), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Tree model: Predicted vs actual", x = "Predicted", y = "Actual")

tree_resid_vs_actual <- ggplot(tree_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Tree model: Residuals", x = "Residuals", y = "Actual")

tree_qq <- ggplot(tree_df, aes(sample = as.numeric(y_hat_tree))) + stat_qq() +
  labs(title = "Tree model: Q-Q plot") +
  stat_qq_line(color = "red")

tree_hist_residuals <- ggplot(tree_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Tree model: Histogram of residuals, each bar = 10 rows of data_")


#### XGBoost visualizations ####
xgb_df <- data.frame(actual = c(test$medv, validation$medv), predicted = as.numeric(y_hat_xgb), residuals = c(test$medv, validation$medv) - y_hat_xgb)

xgb_pred_vs_actual <- ggplot(xgb_df, mapping = aes(x = as.numeric(y_hat_xgb), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "XGBoost model: Predicted vs actual", x = "Predicted", y = "Actual")

xgb_resid_vs_actual <- ggplot(xgb_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "XGBoost model: Residuals", x = "Residuals", y = "Actual")

xgb_qq <- ggplot(xgb_df, aes(sample = as.numeric(y_hat_xgb))) + stat_qq() +
  labs(title = "XGboost model: Q-Q plot") +
  stat_qq_line(color = "red")

xgb_hist_residuals <- ggplot(xgb_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "XGBoost model: Histogram of residuals, each bar = 10 rows of data_")


### Ensemble Bagged Random Forest Visualizations ####
ensemble_bag_rf_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_bag_rf_test_predict_value, ensemble_bag_rf_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_bag_rf)

ensemble_bag_rf_pred_vs_actual <- ggplot(ensemble_bag_rf_df, mapping = aes(x = as.numeric(predicted), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Bagged Random Forest model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_bag_rf_resid_vs_actual <- ggplot(ensemble_bag_rf_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Bagged Random Forest model: Residuals", x = "Residuals", y = "Actual")

ensemble_bag_rf_qq <- ggplot(ensemble_bag_rf_df, aes(sample = as.numeric(predicted))) + stat_qq() +
  labs(title = "Ensemble Bagged Random forest model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_bag_rf_hist_residuals <- ggplot(ensemble_bag_rf_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Bagged Random Forest model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Bayes Generalized Linear Models visualizations ####
ensemble_bayesglm_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_bayesglm_test_predict_value, ensemble_bayesglm_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_bayesglm)

ensemble_bayesglm_pred_vs_actual <- ggplot(ensemble_bayesglm_df, mapping = aes(x = as.numeric(ensemble_y_hat_bayesglm), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Bayes General Linear Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_bayesglm_resid_vs_actual <- ggplot(ensemble_bayesglm_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Bayes General Linear Model: Residuals", x = "Residuals", y = "Actual")

ensemble_bayesglm_qq <- ggplot(ensemble_bayesglm_df, aes(sample = as.numeric(ensemble_y_hat_bayesglm))) + stat_qq() +
  labs(title = "Ensemble Bayes General Linear Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_bayesglm_hist_residuals <- ggplot(ensemble_bayesglm_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Bayes General Linear Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Boosted Random Forest visualizations ####
ensemble_boost_rf_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_boost_rf_test_predict_value, ensemble_boost_rf_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_boost_rf)

ensemble_boost_rf_pred_vs_actual <- ggplot(ensemble_boost_rf_df, mapping = aes(x = as.numeric(predicted), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Boosted Random Forest Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_boost_rf_resid_vs_actual <- ggplot(ensemble_boost_rf_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Boosted Random Forest Model: Residuals", x = "Residuals", y = "Actual")

ensemble_boost_rf_qq <- ggplot(ensemble_boost_rf_df, aes(sample = as.numeric(predicted))) + stat_qq() +
  labs(title = "Ensemble Boosted Random Forest Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_boost_rf_hist_residuals <- ggplot(ensemble_boost_rf_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Boosted Random Forest Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Cubist Data Visualizations ####
ensemble_cubist_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_cubist_test_predict_value, ensemble_cubist_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_cubist)

ensemble_cubist_pred_vs_actual <- ggplot(ensemble_cubist_df, mapping = aes(x = as.numeric(predicted), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Cubist Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_cubist_resid_vs_actual <- ggplot(ensemble_cubist_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Cubist Model: Residuals", x = "Residuals", y = "Actual")

ensemble_cubist_qq <- ggplot(ensemble_cubist_df, aes(sample = as.numeric(actual))) + stat_qq() +
  labs(title = "Ensemble Cubist Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_cubist_hist_residuals <- ggplot(ensemble_cubist_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Cubist Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Earth Data Visualizations ####
ensemble_earth_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_earth_test_predict_value, ensemble_earth_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_earth)

ensemble_earth_pred_vs_actual <- ggplot(ensemble_earth_df, mapping = aes(x = as.numeric(actual), y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Earth Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_earth_resid_vs_actual <- ggplot(ensemble_earth_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Earth Model: Residuals", x = "Residuals", y = "Actual")

ensemble_earth_qq <- ggplot(ensemble_earth_df, aes(sample = as.numeric(actual))) + stat_qq() +
  labs(title = "Ensemble Earth Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_earth_hist_residuals <- ggplot(ensemble_earth_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Earth Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Generalized Additive Models visualizations ####
ensemble_gam_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_gam_test_predict_value, ensemble_gam_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_gam)

ensemble_gam_pred_vs_actual <- ggplot(ensemble_gam_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Generalized Additive Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_gam_resid_vs_actual <- ggplot(ensemble_gam_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Generalized Additive Model: Residuals", x = "Residuals", y = "Actual")

ensemble_gam_qq <- ggplot(ensemble_gam_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble Generalized Additve Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_gam_hist_residuals <- ggplot(ensemble_gam_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Generalized Additve Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble bagging ####
ensemble_bagging_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_bagging_test_predict_value, ensemble_bagging_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_bagging)

ensemble_bagging_pred_vs_actual <- ggplot(ensemble_bagging_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble bagging Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_bagging_resid_vs_actual <- ggplot(ensemble_bagging_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble bagging Model: Residuals", x = "Residuals", y = "Actual")

ensemble_bagging_qq <- ggplot(ensemble_bagging_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble bagging Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_bagging_hist_residuals <- ggplot(ensemble_bagging_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble bagging Model: Histogram of residuals, each bar = 10 rows of data")


#### Ensemble Linear Visualizations ####
ensemble_linear_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_linear_test_predict_value, ensemble_linear_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_linear)

ensemble_linear_pred_vs_actual <- ggplot(ensemble_linear_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Linear Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_linear_resid_vs_actual <- ggplot(ensemble_linear_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Linear Model: Residuals", x = "Residuals", y = "Actual")

ensemble_linear_qq <- ggplot(ensemble_linear_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble Linear Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_linear_hist_residuals <- ggplot(ensemble_linear_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Linear Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Partial Least Squares visualizations ####
ensemble_pls_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_pls_test_predict_value, ensemble_pls_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_pls)

ensemble_pls_pred_vs_actual <- ggplot(ensemble_pls_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Partial Least Squares Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_pls_resid_vs_actual <- ggplot(ensemble_pls_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Partial Least Squares Model: Residuals", x = "Residuals", y = "Actual")

ensemble_pls_qq <- ggplot(ensemble_pls_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble Partial Least Squares Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_pls_hist_residuals <- ggplot(ensemble_pls_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Partial Least Squares Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Principal Components Regression visualizations ####
ensemble_pcr_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_pcr_test_predict_value, ensemble_pcr_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_pcr)

ensemble_pcr_pred_vs_actual <- ggplot(ensemble_pcr_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Principal Components Regresion Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_pcr_resid_vs_actual <- ggplot(ensemble_pcr_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Principal Components Regresion Model: Residuals", x = "Residuals", y = "Actual")

ensemble_pcr_qq <- ggplot(ensemble_pcr_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble Principal Components Regresion Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_pcr_hist_residuals <- ggplot(ensemble_pcr_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Principal Components Regresion Model: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Rpart visualizations ####
ensemble_rpart_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_rpart_test_predict_value, ensemble_rpart_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_rpart)

ensemble_rpart_pred_vs_actual <- ggplot(ensemble_rpart_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Rpart Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_rpart_resid_vs_actual <- ggplot(ensemble_rpart_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Rpart Mode: Residuals", x = "Residuals", y = "Actual")

ensemble_rpart_qq <- ggplot(ensemble_rpart_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble Rpart Mode: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_rpart_hist_residuals <- ggplot(ensemble_rpart_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Rpart Mode: Histogram of residuals, each bar = 10 rows of data_")


#### Ensemble Tree visualizations ####
ensemble_tree_df <- data.frame(actual = c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble),
  predicted = as.numeric(c(ensemble_tree_test_predict_value, ensemble_tree_validation_predict_value)),
  residuals = as.numeric(c(ensemble_test$y_ensemble, ensemble_validation$y_ensemble)) - ensemble_y_hat_tree)

ensemble_tree_pred_vs_actual <- ggplot(ensemble_tree_df, mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") +
  labs(title = "Ensemble Generalized Additive Model: Predicted vs actual", x = "Predicted", y = "Actual")

ensemble_tree_resid_vs_actual <- ggplot(ensemble_tree_df, mapping = aes(x = residuals, y = actual)) +
  geom_point() +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Generalized Additive Model: Residuals", x = "Residuals", y = "Actual")

ensemble_tree_qq <- ggplot(ensemble_tree_df, aes(sample = predicted)) + stat_qq() +
  labs(title = "Ensemble Generalized Additve Model: Q-Q plot") +
  stat_qq_line(color = "red")

ensemble_tree_hist_residuals <- ggplot(ensemble_tree_df, mapping = aes(x = residuals)) +
  geom_histogram(bins = round(nrow(df)/10)) +
  geom_vline(xintercept = 0, color="red") +
  labs(title = "Ensemble Generalized Additve Model: Histogram of residuals, each bar = 10 rows of data_")

#### <------------------------------------------------------ 9. Summary Results ----------------------------------------------> ##########


summary_results <- data.frame(
  'Model' = c('Actual_data', 'Bagged Random Forest', 'Bagging', 'BayesGLM', 'BayesRNN', 'BoostedRF',
    'Cubist', 'Earth', 'Gradient Boosted', 'K-Nearest Neighbors',
    'Linear', 'LQS', 'Partial Least Squares', 'Principal Components',
    'Random Forest', 'Robust', 'Rpart', 'Smoothing Splines', 'Support Vector Machines', 'Trees', 'XGBoost',
    'Ensemble Bagged RF', 'Ensemble Bagging', 'Ensemble Bayes GLM', 'Ensemble Bayes RNN', 'Ensemble Boosted Random Forest',
    'Ensemble Cubist', 'Ensemble Earth',
    'Ensemble K-Nearest Neighbors', 'Ensemble Linear', 'Ensemble Partial Least Squares',
    'Ensemble Principal Components Analysis', 'Ensemble Random Forest', 'Ensemble Rpart','Ensemble GAM + Splines',
    'Ensemble Support Vector Machines', 'Ensemble Trees', 'Ensemble XGBoost'),
  
  'Mean_data' = round(c(actual_mean, bag_rf_predict_value_mean, bagging_predict_value_mean, bayesglm_predict_value_mean,
    bayesrnn_predict_value_mean, boost_rf_predict_value_mean,
    cubist_predict_value_mean, earth_predict_value_mean,
    gb_predict_value_mean, knn_predict_value_mean,
    linear_predict_value_mean, lqs_predict_value_mean,
    pls_predict_value_mean, pcr_predict_value_mean, rf_predict_value_mean,
    robust_predict_value_mean, rpart_predict_value_mean, gam_predict_value_mean, svm_predict_value_mean,
    tree_predict_value_mean, xgb_mean, ensemble_bag_rf_predict_value_mean, ensemble_bagging_predict_value_mean,
    ensemble_bayesglm_predict_value_mean, ensemble_bayesrnn_predict_value_mean, ensemble_boost_rf_predict_value_mean,
    ensemble_cubist_predict_value_mean, ensemble_earth_predict_value_mean,
    ensemble_knn_predict_value_mean,
    ensemble_linear_predict_value_mean,  ensemble_pls_predict_value_mean,
    ensemble_pcr_predict_value_mean, ensemble_rf_predict_value_mean,
    ensemble_rpart_predict_value_mean, ensemble_gam_predict_value_mean, ensemble_svm_predict_value_mean, ensemble_tree_predict_value_mean,
    ensemble_xgb_mean), 7),
  
  'Std_Dev_of_the_data' = round(c(actual_sd, bag_rf_sd_mean, bagging_sd_mean, bayesglm_sd_mean, bayesrnn_sd_mean, boost_rf_sd_mean,
    cubist_sd_mean, earth_sd_mean,
    gb_sd_mean, knn_sd_mean, 
    linear_sd_mean, lqs_sd_mean,
    pls_sd_mean, pcr_sd_mean, rf_sd_mean, robust_sd_mean,
    rpart_sd_mean, gam_sd_mean, svm_sd_mean, tree_sd_mean, xgb_sd,
    ensemble_bag_rf_sd_mean, ensemble_bagging_sd_mean, ensemble_bayesglm_sd_mean, ensemble_bayesrnn_sd_mean, ensemble_boost_rf_sd_mean,
    ensemble_cubist_sd_mean, ensemble_earth_sd_mean,
    ensemble_knn_sd_mean,
    ensemble_linear_sd_mean, ensemble_pls_sd_mean, ensemble_pcr_sd_mean, ensemble_rf_sd_mean,
    ensemble_rpart_sd_mean, ensemble_gam_sd_mean, ensemble_svm_sd_mean,
    ensemble_tree_sd_mean, ensemble_xgb_sd), 7),
  
  'Mean_Train_RMSE' = round(c(0, bag_rf_train_RMSE_mean, bagging_train_RMSE_mean, bayesglm_train_RMSE_mean, bayesrnn_train_RMSE_mean,
    boost_rf_train_RMSE_mean,
    cubist_train_RMSE_mean, earth_train_RMSE_mean,
    gb_train_RMSE_mean, knn_train_RMSE_mean,
    linear_train_RMSE_mean, lqs_train_RMSE_mean,
    pls_train_RMSE_mean, pcr_train_RMSE_mean, rf_train_RMSE_mean,
    robust_train_RMSE_mean, rpart_train_RMSE_mean, gam_train_RMSE_mean, svm_train_RMSE_mean,
    tree_train_RMSE_mean, xgb_train_RMSE_mean, ensemble_bag_rf_train_RMSE_mean, ensemble_bagging_train_RMSE_mean,
    ensemble_bayesglm_train_RMSE_mean, ensemble_bayesrnn_train_RMSE_mean,  ensemble_boost_rf_train_RMSE_mean,
    ensemble_cubist_train_RMSE_mean, ensemble_earth_train_RMSE_mean,
    ensemble_knn_train_RMSE_mean,
    ensemble_linear_train_RMSE_mean,  ensemble_pls_train_RMSE_mean,
    ensemble_pcr_train_RMSE_mean, ensemble_rf_train_RMSE_mean,
    ensemble_rpart_train_RMSE_mean, ensemble_gam_train_RMSE_mean, ensemble_svm_train_RMSE_mean, tree_train_RMSE_mean,
    ensemble_xgb_train_RMSE_mean), 7),
  
  'Mean_Test_RMSE' = round(c(0,bag_rf_test_RMSE_mean, bagging_test_RMSE_mean, bayesglm_test_RMSE_mean, bayesrnn_test_RMSE_mean,
    boost_rf_test_RMSE_mean,
    cubist_test_RMSE_mean, earth_test_RMSE_mean,
    gb_test_RMSE_mean, knn_test_RMSE_mean,
    linear_test_RMSE_mean, lqs_test_RMSE_mean,
    pls_test_RMSE_mean, pcr_test_RMSE_mean, rf_test_RMSE_mean, robust_test_RMSE_mean,
    rpart_test_RMSE_mean, gam_test_RMSE_mean, svm_test_RMSE_mean, tree_test_RMSE_mean, xgb_test_RMSE_mean,
    ensemble_bag_rf_test_RMSE_mean, ensemble_bagging_test_RMSE_mean, ensemble_bayesglm_test_RMSE_mean, ensemble_bayesrnn_test_RMSE_mean,
    ensemble_boost_rf_test_RMSE_mean, ensemble_cubist_test_RMSE_mean,
    ensemble_earth_test_RMSE_mean,
    ensemble_knn_test_RMSE_mean, ensemble_linear_test_RMSE_mean, ensemble_pls_test_RMSE_mean,
    ensemble_pcr_test_RMSE_mean, ensemble_rf_test_RMSE_mean, ensemble_rpart_test_RMSE_mean, ensemble_gam_test_RMSE_mean,
    ensemble_svm_test_RMSE_mean, tree_test_RMSE_mean, ensemble_xgb_test_RMSE_mean), 7),
  
  'Mean_Validation_RMSE' = round(c(0, bag_rf_validation_RMSE_mean, bagging_validation_RMSE_mean, bayesglm_validation_RMSE_mean,
    bayesrnn_validation_RMSE_mean, boost_rf_validation_RMSE_mean,
    cubist_validation_RMSE_mean, earth_validation_RMSE_mean,
    gb_validation_RMSE_mean, knn_validation_RMSE_mean, 
    linear_validation_RMSE_mean, lqs_validation_RMSE_mean,
    pls_validation_RMSE_mean, pcr_validation_RMSE_mean, rf_validation_RMSE_mean,
    robust_validation_RMSE_mean, rpart_validation_RMSE_mean, gam_validation_RMSE_mean, svm_validation_RMSE_mean,
    tree_validation_RMSE_mean, xgb_validation_RMSE_mean, ensemble_bag_rf_validation_RMSE_mean, ensemble_bagging_validation_RMSE_mean,
    ensemble_bayesglm_validation_RMSE_mean, ensemble_bayesrnn_validation_RMSE_mean, ensemble_boost_rf_validation_RMSE_mean,
    ensemble_cubist_validation_RMSE_mean,
    ensemble_earth_validation_RMSE_mean,
    ensemble_knn_validation_RMSE_mean,
    ensemble_linear_validation_RMSE_mean, ensemble_pls_validation_RMSE_mean, ensemble_pcr_validation_RMSE_mean,
    ensemble_rf_validation_RMSE_mean, ensemble_rpart_validation_RMSE_mean, ensemble_gam_validation_RMSE_mean,
    ensemble_svm_validation_RMSE_mean, tree_validation_RMSE_mean, ensemble_xgb_validation_RMSE_mean), 7),
  
  'Holdout_RMSE_mean' = round(c(0, bag_rf_holdout_RMSE_mean, bagging_holdout_RMSE_mean, bayesglm_holdout_RMSE_mean,
    bayesrnn_holdout_RMSE_mean, boost_rf_holdout_RMSE_mean,
    cubist_holdout_RMSE_mean, earth_holdout_RMSE_mean,
    gb_holdout_RMSE_mean, knn_holdout_RMSE_mean,
    linear_holdout_RMSE_mean, lqs_holdout_RMSE_mean, pls_holdout_RMSE_mean, pcr_holdout_RMSE_mean, rf_holdout_RMSE_mean,
    robust_holdout_RMSE_mean, rpart_holdout_RMSE_mean, gam_holdout_RMSE_mean, svm_holdout_RMSE_mean, tree_holdout_RMSE_mean, xgb_holdout_RMSE_mean,
    ensemble_bag_rf_holdout_RMSE_mean, ensemble_bagging_holdout_RMSE_mean, ensemble_bayesglm_holdout_RMSE_mean, ensemble_bayesrnn_holdout_RMSE_mean,
    ensemble_boost_rf_holdout_RMSE_mean, ensemble_cubist_holdout_RMSE_mean,
    ensemble_earth_holdout_RMSE_mean, ensemble_knn_holdout_RMSE_mean,
    ensemble_linear_holdout_RMSE_mean, ensemble_pls_holdout_RMSE_mean, ensemble_pcr_holdout_RMSE_mean, ensemble_rf_holdout_RMSE_mean,
    ensemble_rpart_holdout_RMSE_mean, ensemble_gam_holdout_RMSE_mean, ensemble_svm_holdout_RMSE_mean, tree_holdout_RMSE_mean,
    ensemble_xgb_holdout_RMSE_mean), 7),
  
  'RMSE_Std_Deviation' = round(c(0, bag_rf_holdout_RMSE_sd_mean, bagging_holdout_RMSE_sd_mean, bayesglm_holdout_RMSE_sd_mean,
    bayesrnn_holdout_RMSE_sd_mean, boost_rf_holdout_RMSE_sd_mean,
    cubist_holdout_RMSE_sd_mean, earth_holdout_RMSE_sd_mean,
    gb_holdout_RMSE_sd_mean, knn_holdout_RMSE_sd_mean,
    linear_holdout_RMSE_sd_mean, lqs_holdout_RMSE_sd_mean, pls_holdout_RMSE_sd_mean, pcr_holdout_RMSE_sd_mean,
    rf_holdout_RMSE_sd_mean, robust_holdout_RMSE_sd_mean, rpart_holdout_RMSE_sd_mean, gam_holdout_RMSE_sd_mean, svm_holdout_RMSE_sd_mean,
    tree_holdout_RMSE_sd_mean, xgb_holdout_RMSE_sd_mean, ensemble_bag_rf_holdout_RMSE_sd_mean, ensemble_bagging_holdout_RMSE_sd_mean,
    ensemble_bayesglm_holdout_RMSE_sd_mean, ensemble_bayesrnn_holdout_RMSE_sd_mean, ensemble_boost_rf_holdout_RMSE_sd_mean,
    ensemble_cubist_holdout_RMSE_sd_mean, ensemble_earth_holdout_RMSE_sd_mean,
    ensemble_knn_holdout_RMSE_sd_mean,
    ensemble_linear_holdout_RMSE_sd_mean, ensemble_pls_holdout_RMSE_sd_mean, ensemble_pcr_holdout_RMSE_sd_mean, ensemble_rf_holdout_RMSE_sd_mean,
    ensemble_rpart_holdout_RMSE_sd_mean, ensemble_gam_holdout_RMSE_sd_mean, ensemble_svm_holdout_RMSE_sd_mean, tree_holdout_RMSE_sd_mean,
    ensemble_xgb_holdout_RMSE_sd_mean), 7),
  
  'Over-Under-fitting' = round(c(0, bag_rf_overfitting_mean, bagging_overfitting_mean, bayesglm_overfitting_mean, bayesrnn_overfitting_mean,
    boost_rf_overfitting_mean,
    cubist_overfitting_mean, earth_overfitting_mean,
    gb_overfitting_mean, knn_overfitting_mean,
    linear_overfitting_mean, lqs_overfitting_mean,
    pls_overfitting_mean, pcr_overfitting_mean, rf_overfitting_mean,
    robust_overfitting_mean, rpart_overfitting_mean, gam_overfitting_mean, svm_overfitting_mean,
    tree_overfitting_mean, xgb_overfitting_mean, ensemble_bag_rf_overfitting_mean, ensemble_bagging_overfitting_mean,
    ensemble_bayesglm_overfitting_mean, ensemble_bayesrnn_overfitting_mean, ensemble_boost_rf_overfitting_mean,
    ensemble_cubist_overfitting_mean, ensemble_earth_overfitting_mean,
    ensemble_knn_overfitting_mean,
    ensemble_linear_overfitting_mean,  ensemble_pls_overfitting_mean,
    ensemble_pcr_overfitting_mean, ensemble_rf_overfitting_mean,
    ensemble_rpart_overfitting_mean, ensemble_gam_overfitting_mean, ensemble_svm_overfitting_mean, tree_overfitting_mean,
    ensemble_xgb_overfitting_mean), 7),
  
  'Over-Under-Fitting-range-min' = round(c(0, bag_rf_overfitting_range[1], bagging_overfitting_range[1], bayesglm_overfitting_range[1],
    bayesrnn_overfitting_range[1], boost_rf_overfitting_range[1],
    cubist_overfitting_range[1], earth_overfitting_range[1],
    gb_overfitting_range[1], knn_overfitting_range[1],
    linear_overfitting_range[1], lqs_overfitting_range[1],
    pls_overfitting_range[1], pcr_overfitting_range[1], rf_overfitting_range[1],
    robust_overfitting_range[1], rpart_overfitting_range[1], gam_overfitting_range[1], svm_overfitting_range[1],
    tree_overfitting_range[1], xgb_overfitting_range[1], ensemble_bag_rf_overfitting_range[1], ensemble_bagging_overfitting_range[1],
    ensemble_bayesglm_overfitting_range[1], ensemble_bayesrnn_overfitting_range[1], ensemble_boost_rf_overfitting_range[1],
    ensemble_cubist_overfitting_range[1], ensemble_earth_overfitting_range[1],
    ensemble_knn_overfitting_range[1],
    ensemble_linear_overfitting_range[1],  ensemble_pls_overfitting_range[1],
    ensemble_pcr_overfitting_range[1], ensemble_rf_overfitting_range[1],
    ensemble_rpart_overfitting_range[1], ensemble_gam_overfitting_range[1], ensemble_svm_overfitting_range[1], tree_overfitting_range[1],
    ensemble_xgb_overfitting_range[1]), 7),
  
  'Over-Under-Fitting-range-max' = round(c(0, bag_rf_overfitting_range[2], bagging_overfitting_range[2], bayesglm_overfitting_range[2],
    bayesrnn_overfitting_range[2], boost_rf_overfitting_range[2],
    cubist_overfitting_range[2], earth_overfitting_range[2],
    gb_overfitting_range[2], knn_overfitting_range[2],
    linear_overfitting_range[2], lqs_overfitting_range[2],
    pls_overfitting_range[2], pcr_overfitting_range[2], rf_overfitting_range[2],
    robust_overfitting_range[2], rpart_overfitting_range[2], gam_overfitting_range[2], svm_overfitting_range[2],
    tree_overfitting_range[2], xgb_overfitting_range[2], ensemble_bag_rf_overfitting_range[2], ensemble_bagging_overfitting_range[2],
    ensemble_bayesglm_overfitting_range[2], ensemble_bayesrnn_overfitting_range[2], ensemble_boost_rf_overfitting_range[2],
    ensemble_cubist_overfitting_range[2], ensemble_earth_overfitting_range[2],
    ensemble_knn_overfitting_range[2],
    ensemble_linear_overfitting_range[2],  ensemble_pls_overfitting_range[2],
    ensemble_pcr_overfitting_range[2], ensemble_rf_overfitting_range[2],
    ensemble_rpart_overfitting_range[2], ensemble_gam_overfitting_range[2], ensemble_svm_overfitting_range[2], tree_overfitting_range[2],
    ensemble_xgb_overfitting_range[2]), 7),
  
  'Duration' = round(c(0, bag_rf_duration_mean, bagging_duration_mean, bayesglm_duration_mean, bayesrnn_duration_mean, boost_rf_duration_mean,
    cubist_duration_mean, earth_duration_mean,
    gb_duration_mean, knn_duration_mean,
    linear_duration_mean, lqs_duration_mean,
    pls_duration_mean, pcr_duration_mean, rf_duration_mean,
    robust_duration_mean, rpart_duration_mean, gam_duration_mean, svm_duration_mean,
    tree_duration_mean, xgb_duration_mean, ensemble_bag_rf_duration_mean, ensemble_bagging_duration_mean,
    ensemble_bayesglm_duration_mean, ensemble_bayesrnn_duration_mean, ensemble_boost_rf_duration_mean,
    ensemble_cubist_duration_mean, ensemble_earth_duration_mean,
    ensemble_knn_duration_mean,
    ensemble_linear_duration_mean, ensemble_pls_duration_mean,
    ensemble_pcr_duration_mean, ensemble_rf_duration_mean,
    ensemble_rpart_duration_mean, ensemble_gam_duration_mean, ensemble_svm_duration_mean, ensemble_tree_duration_mean,
    ensemble_xgb_duration_mean), 7),
  
  'Final_Model' = c('Not applicable', 'bag_rf_train_fit', 'bagging_train_fit', 'bayesglm_train_fit', 'bayesrnn_train_fit', 'boost_rf_train_fit',
    'cubist_train_fit', 'earth_train_fit', 'gb_train_fit', 'knn_train_fit',
    'linear_train_fit', 'lqs_train_fit', 'pls_train_fit', 'pcr_train_fit',
    'rf_train_fit', 'robust_train_fit', 'rpart_train_fit', 'gam_train_fit', 'svm_train_fit', 'tree_train_fit', 'xgb_train_fit',
    'ensemble_bag_rf_train_fit', 'ensemble_bagging_train_fit', 'ensemble_bayesglm_train_fit', 'ensemble_bayesrnn_train_fit', 'ensemble_boost_rf_train_fit',
    'ensemble_cubist_train_fit', 'ensemble_earth_train_fit',
    'ensemble_knn_trian_fit', 'ensemble_linear_train_fit', 'ensemble_pls_train_fit',
    'ensemble_pcr_train_fit', 'ensemble_rf_train_fit', 'ensemble_rpart_train_fit', 'ensemble_gam_train_fit',
    'ensemble_svm_train_fit', 'ensemble_tree_train_fit', 'xgb_train_fit'),
  
  'Predict_On_New_Data' = c('Not applicable', 'predict(object=bag_rf_train_fit, newdata = newdata)',
    'predict(object=bagging_train_fit, newdata=newdata)',
    'predict(object=bayesglm_train_fit, newdata = newdata)', 'predict(object=bayesrnn_train_fit, newdata = newdata)',
    'predict(object=boost_rf_train_fit, newdata = newdata)',
    'predict(object=cubist_train_fit, newdata = newdata)', 'predict(object=earth_train_fit, newdata = newdata)',
    'predict(object=gb_train_fit, newdata = newdata)',
    'predict(object=knn_train_fit, newdata = newdata)', 'predict(object=linear_tune, newdata = newdata)',
    'predict(object=lqs_model, newdata = newdata)',
    'predict(object=pls_model, newdata = newdata)', 'predict(object=pcr_model, newdata = newdata)',
    'predict(object=rf_train_fit, newdata = newdata)', 'predict(object=robust_train_fit, newdata = newdata)',
    'predict(object=rpart_train_fit, newdata = newdata)', 'predict(object=gam_train_fit, newdata = newdata)',
    'predict(object=svm_train_fit, newdata = newdata)', 'predict(object=tree_train_fit, newdata = newdata',
    'predict(object=xgb_train_fit, newdata = newdata)',
    'predict(object=ensemble_bag_rf_train_fit, newdata = newdata)', 'predict(object=ensemble_bagging_train_fit,newdata=newdata)',
    'predict(object=ensemble_bayesglm_train_fit, newdata = newdata)', 'predict(object=ensemble_bayesrnn_train_fit, newdata = newdata)',
    'predict(object=ensemble_boost_rf_train_fit, newdata = newdata)',
    'predict(object=ensemble_cubist_train_fit, newdata = newdata)', 'predict(object=ensemble_earth_train_fit, newdata = newdata)',
    'predict(object=ensemble_knn_train_fit, newdata = newdata)', 'predict(object=ensemble_linear_train_fit, newdata = newdata)',
    'predict(object=ensemble_pls_train_fit, newdata = newdata)',
    'predict(object=ensemble_pcr_train_fit, newdata = newdata)', 'predict(object=ensemble_rf_train_fit, newdata = newdata)',
    'predict(object=ensemble_rpart_train_fit, newdata = newdata)', 'predict(object=ensemble_gam_train_fit, newdata = newdata)',
    'predict(object=ensemble_svm_train_fit, newdata = newdata)', 'predict(object=ensemble_tree_train_fit, newdata = newdata)',
    'predict(object=ensemble_xgb_train_fit, newdata = newdata)')
)

summary_results <- summary_results %>% arrange(Holdout_RMSE_mean)

reactable::reactable(summary_results, searchable = TRUE, pagination = FALSE, wrap = TRUE, fullWidth = TRUE, filterable = TRUE, bordered = TRUE,
  striped = TRUE, highlight = TRUE, rownames = TRUE, resizable = TRUE) %>%
  add_title("RMSE, means, fitting, model summaries of the train, test and validation sets")

data_visualizations <- summary_results[3,1]

if(data_visualizations[1] == 'Bagged Random Forest'){
  grid.arrange(bag_rf_pred_vs_actual, bag_rf_resid_vs_actual, bag_rf_qq, bag_rf_hist_residuals, ncol = 2)
  print(bag_rf_train_fit$best_model)
  print(summary(bag_rf_train_fit))
}
if(data_visualizations[1] == 'Bagging'){
    grid.arrange(bagging_pred_vs_actual, bagging_resid_vs_actual, bagging_qq, bagging_hist_residuals, ncol = 2)
    print(bagging_train_fit)
    print(summary(bagging_train_fit))
}
if(data_visualizations[1] == 'BayesGLM'){
    grid.arrange(bayesglm_pred_vs_actual, bayesglm_resid_vs_actual, bayesglm_qq, bayesglm_hist_residuals, ncol = 2)
    bayesglm_train_fit
    summary(bayesglm_train_fit)
}
if(data_visualizations[1] == 'BayesRNN'){
    grid.arrange(bayesrnn_pred_vs_actual, bayesrnn_resid_vs_actual, bayesrnn_qq, bayesrnn_hist_residuals, ncol = 2)
    bayesrnn_train_fit
    summary(bayesrnn_train_fit)
}
if(data_visualizations[1] == 'BoostedRF'){
    grid.arrange(boost_rf_pred_vs_actual, boost_rf_resid_vs_actual, boost_rf_qq, boost_rf_hist_residuals, ncol = 2)
    boost_rf_train_fit$best_model
    summary(boost_rf_train_fit$best_model)
}
if(data_visualizations[1] == 'Cubist'){
    grid.arrange(cubist_pred_vs_actual, cubist_resid_vs_actual, cubist_qq, cubist_hist_residuals, ncol = 2)
    cubist_train_fit
    summary(cubist_train_fit)
}
if(data_visualizations[1] == 'Earth'){
    grid.arrange(earth_pred_vs_actual, earth_resid_vs_actual, earth_qq, earth_hist_residuals, ncol = 2)
    earth_train_fit
    summary(earth_train_fit)
}
if(data_visualizations[1] == 'Gradient Boosted'){
    grid.arrange(gb_pred_vs_actual, gb_resid_vs_actual, gb_qq, gb_hist_residuals, ncol = 2)
    gb_train_fit
    summary(gb_train_fit)
}
if(data_visualizations[1] == 'K-Nearest Neighbors'){
    grid.arrange(knn_pred_vs_actual, knn_resid_vs_actual, knn_qq, knn_hist_residuals, ncol = 2)
    knn_train_fit
    summary(knn_train_fit)
}
if(data_visualizations[1] == 'Linear'){
    grid.arrange(linear_pred_vs_actual, linear_resid_vs_actual, linear_qq, linear_hist_residuals, ncol = 2)
    linear_train_fit$best_model
    summary(linear_train_fit$best_model)
}
if(data_visualizations[1] == 'LQS'){
    grid.arrange(lqs_pred_vs_actual, lqs_resid_vs_actual, lqs_qq, lqs_hist_residuals, ncol = 2)
    lqs_train_fit
    summary(lqs_train_fit)
}
if(data_visualizations[1] == 'Partial Least Squares'){
    grid.arrange(pls_pred_vs_actual, pls_resid_vs_actual, pls_qq, pls_hist_residuals, ncol = 2)
    pls_train_fit
    summary(pls_train_fit)
}
if(data_visualizations[1] == 'Principal Components'){
    grid.arrange(pcr_pred_vs_actual, pcr_resid_vs_actual, pcr_qq, pcr_hist_residuals, ncol = 2)
    pcr_train_fit
    summary(pcr_train_fit)
}
if(data_visualizations[1] == 'Random Forest'){
    grid.arrange(rf_pred_vs_actual, rf_resid_vs_actual, rf_qq, rf_hist_residuals, ncol = 2)
    rf_train_fit$best_model
    summary(rf_train_fit$best_model)
}
if(data_visualizations[1] == 'Robust'){
    grid.arrange(rf_pred_vs_actual, rf_resid_vs_actual, rf_qq, rf_hist_residuals, ncol = 2)
    rf_train_fit$best_model
    summary(rf_train_fit$best_model)
}
if(data_visualizations[1] == 'Rpart'){
    grid.arrange(rpart_pred_vs_actual, rpart_resid_vs_actual, rpart_qq, rpart_hist_residuals, ncol = 2)
    rpart_train_fit
    summary(rpart_train_fit)
}
if(data_visualizations[1] == 'Smoothing Splines'){
    grid.arrange(gam_pred_vs_actual, gam_resid_vs_actual, gam_qq, gam_hist_residuals, ncol = 2)
    gam_train_fit
    summary(gam_train_fit)
}
if(data_visualizations[1] == 'Support Vector Machines'){
    grid.arrange(svm_pred_vs_actual, svm_resid_vs_actual, svm_qq, svm_hist_residuals, ncol = 2)
    svm_train_fit$best_model
    summary(svm_train_fit$best_model)
}
if(data_visualizations[1] == 'Trees'){
    grid.arrange(tree_pred_vs_actual, tree_resid_vs_actual, tree_qq, tree_hist_residuals, ncol = 2)
    tree_train_fit
    summary(tree_train_fit)
}
if(data_visualizations[1] == 'XGBoost'){
    grid.arrange(tree_pred_vs_actual, tree_resid_vs_actual, tree_qq, tree_hist_residuals, ncol = 2)
    tree_train_fit
    summary(tree_train_fit)
}
if(data_visualizations[1] == 'Ensemble Bagged RF'){
    grid.arrange(ensemble_bag_rf_pred_vs_actual, ensemble_bag_rf_resid_vs_actual, ensemble_bag_rf_qq, ensemble_bag_rf_hist_residuals, ncol = 2)
    ensemble_bag_rf_train_fit
    summary(ensemble_bag_rf_train_fit)
}
if(data_visualizations[1] == 'Ensemble Bagging'){
    grid.arrange(ensemble_bagging_pred_vs_actual, ensemble_bagging_resid_vs_actual, ensemble_bagging_qq, ensemble_bagging_hist_residuals, ncol = 2)
    ensemble_bagging_train_fit
    summary(ensemble_bagging_train_fit)
}
if(data_visualizations[1] == 'Ensemble Bayes GLM'){
    grid.arrange(ensemble_bayesglm_pred_vs_actual, ensemble_bayesglm_resid_vs_actual, ensemble_bayesglm_qq, ensemble_bayesglm_hist_residuals, ncol = 2)
    ensemble_bayesglm_train_fit
    summary(ensemble_bayesglm_train_fit)
}
if(data_visualizations[1] == 'Ensemble Bayes RNN'){
    grid.arrange(ensemble_bayesglm_pred_vs_actual, ensemble_bayesglm_resid_vs_actual, ensemble_bayesglm_qq, ensemble_bayesglm_hist_residuals, ncol = 2)
    ensemble_bayesglm_train_fit
    summary(ensemble_bayesglm_train_fit)
}
if(data_visualizations[1] == 'Ensemble Boosted Random Forest'){
    grid.arrange(ensemble_boost_rf_pred_vs_actual, ensemble_boost_rf_resid_vs_actual, ensemble_boost_rf_qq, ensemble_boost_rf_hist_residuals, ncol = 2)
    ensemble_boost_rf_train_fit$best_model
    summary(ensemble_boost_rf_train_fit$best_model)
}
if(data_visualizations[1] == 'Ensemble Cubist'){
    grid.arrange(ensemble_cubist_pred_vs_actual, ensemble_cubist_resid_vs_actual, ensemble_cubist_qq, ensemble_cubist_hist_residuals, ncol = 2)
    ensemble_cubist_train_fit
    summary(ensemble_cubist_train_fit)
}
if(data_visualizations[1] == 'Ensemble Earth'){
    grid.arrange(ensemble_earth_pred_vs_actual, ensemble_earth_resid_vs_actual, ensemble_earth_qq, ensemble_earth_hist_residuals, ncol = 2)
    ensemble_earth_train_fit
    print(summary(ensemble_earth_train_fit))
}
if(data_visualizations[1] == 'Ensemble K-Nearest Neighbors'){
    grid.arrange(ensemble_earth_pred_vs_actual, ensemble_earth_resid_vs_actual, ensemble_earth_qq, ensemble_earth_hist_residuals, ncol = 2)
    ensemble_earth_train_fit
    print(summary(ensemble_earth_train_fit))
}
if(data_visualizations[1] == 'Ensemble Linear'){
    grid.arrange(ensemble_linear_pred_vs_actual, ensemble_linear_resid_vs_actual, ensemble_linear_qq, ensemble_linear_hist_residuals, ncol = 2)
    ensemble_linear_train_fit$best_model
    summary(ensemble_linear_train_fit$best_model)
}
if(data_visualizations[1] == 'Ensemble Partial Least Squares'){
    grid.arrange(ensemble_pls_pred_vs_actual, ensemble_pls_resid_vs_actual, ensemble_pls_qq, ensemble_pls_hist_residuals, ncol = 2)
    ensemble_pls_train_fit
    summary(ensemble_pls_train_fit)
}
if(data_visualizations[1] == 'Ensemble Principal Components Analysis'){
    grid.arrange(ensemble_pcr_pred_vs_actual, ensemble_pcr_resid_vs_actual, ensemble_pcr_qq, ensemble_pcr_hist_residuals, ncol = 2)
    ensemble_pcr_train_fit
    summary(ensemble_pcr_train_fit)
}
if(data_visualizations[1] == 'Ensemble Random Forest'){
    grid.arrange(ensemble_pcr_pred_vs_actual, ensemble_pcr_resid_vs_actual, ensemble_pcr_qq, ensemble_pcr_hist_residuals, ncol = 2)
    ensemble_pcr_train_fit
    summary(ensemble_pcr_train_fit)
}
if(data_visualizations[1] == 'Ensemble Rpart'){
    grid.arrange(ensemble_rpart_pred_vs_actual, ensemble_rpart_resid_vs_actual, ensemble_rpart_qq, ensemble_rpart_hist_residuals, ncol = 2)
    ensemble_rpart_train_fit
    summary(ensemble_rpart_train_fit)
}
if(data_visualizations[1] == 'Ensemble GAM + Splines'){
    grid.arrange(ensemble_gam_pred_vs_actual, ensemble_gam_resid_vs_actual, ensemble_gam_qq, ensemble_gam_hist_residuals, ncol = 2)
    ensemble_gam_train_fit
    summary(ensemble_gam_train_fit)
}
if(data_visualizations[1] == 'Ensemble Support Vector Machines'){
    grid.arrange(ensemble_svm_pred_vs_actual, ensemble_svm_resid_vs_actual, ensemble_svm_qq, ensemble_svm_hist_residuals, ncol = 2)
    ensemble_tree_train_fit
    summary(ensemble_tree_train_fit)
}
if(data_visualizations[1] == 'Ensemble Trees'){
    grid.arrange(ensemble_tree_pred_vs_actual, ensemble_tree_resid_vs_actual, ensemble_tree_qq, ensemble_tree_hist_residuals, ncol = 2)
    ensemble_tree_train_fit
    summary(ensemble_tree_train_fit)
}
if(data_visualizations[1] == 'Ensemble XGBoost'){
    grid.arrange(ensemble_xgb_pred_vs_actual, ensemble_tree_resid_vs_actual, ensemble_tree_qq, ensemble_tree_hist_residuals, ncol = 2)
    ensemble_tree_train_fit
    summary(ensemble_tree_train_fit)
}

#### <------------------------ 10. Strongest evidence based recommendations, suggestions for future research -----------------> ##########
