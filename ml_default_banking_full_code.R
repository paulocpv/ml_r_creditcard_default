# Setting the libraries we are gonna use------

library(caret) # for machine learning
library(tidyverse) # for ordering and ploting data
library(xgboost) # for xgboost
library(pROC) # for calculating the ROC curve and chose our best model

# Setting the significant digits to 3
options(digits = 3)

# Downloading the data--------------------

url <- ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv")

data <- read.csv(url, sep = ",")

# Our variable of interest is default. We rename the variable and turn into a categorical variable 

data_clean <- data |>
  rename(target = default) |>
  mutate(target = factor(target))

# Let's check our target distribution

prop.table(summary(data_clean$target))

# Data Partition---------------------

set.seed(42)
index <- createDataPartition(data_clean$target, times = 1, p = 0.2, list = FALSE)
train_set <- data_clean[-index,]
test_set <- data_clean[index,]

# Some exploratory analysis----------

# Establishing baseline prediction
set.seed(3)
y_hat <- sample(c(1, 2), length(index), replace = TRUE) |>
  factor(levels = levels(test_set$target))

mean(y_hat == test_set$target)

## Using credit history a predictor just as an example

data_clean |>
  group_by(credit_history) |>
  summarise(mean(target == 2))

y_hat_1 <- if_else(test_set$credit_history %in% c("fully repaid", "fully repaid this bank") , 2, 1)
mean(y_hat_1 == test_set$target)

# General Linear Model-----

set.seed(45)
fit_1 <- train(target ~ ., train_set,
               method = "glm")

mean(test_set$target == predict(fit_1, test_set))

matrix_1 <- confusionMatrix(predict(fit_1, test_set), as.factor(test_set$target))
matrix_1$byClass

# Knn Nearest Neighbor----

trControl <- trainControl(method  = "cv",
                          number  = 10)

set.seed(8)
fit_2 <- train(target ~ ., train_set,
               method = "knn",
               trControl  = trControl,
               metric     = "Accuracy",
               tuneGrid = data.frame(k = seq(3, 51, 2)))

mean(test_set$target == predict(fit_2, test_set))

matrix_2 <- confusionMatrix(predict(fit_2, test_set), as.factor(test_set$target))
matrix_2$byClass

# Random Forest----

set.seed(23)
fit_3 <- train(target ~ ., train_set,
                method = "rf",
                tuneGrid = data.frame(mtry = seq(1:7)),
                ntree = 100)

mean(test_set$target == predict(fit_3, test_set))

matrix_3 <- confusionMatrix(predict(fit_3, test_set), as.factor(test_set$target))
matrix_3$byClass

# XGB----

# Pre-processing train set
train_xgb_set <- train_set |>
  mutate_if(is.character, as.factor) |>
  mutate_if(is.factor, as.numeric) |>
  mutate(target = if_else(target == 1, 0, 1))

# Pre-processing test set
test_xgb_set <- test_set |>
  mutate_if(is.character, as.factor) |>
  mutate_if(is.factor, as.numeric) |>
  mutate(target = if_else(target == 1, 0, 1))

# Training model
xgb_train_matrix <- xgb.DMatrix(
  data = as.matrix(train_xgb_set[,!names(train_xgb_set) %in% c("target")]),
                         label = train_xgb_set$target)

xgb_params <- list(objective = "binary:logistic", eval_metric = "auc",
                   max_depth = 11, eta = 0.071, subsample = 0.99,
                   colsample_bytree = 0.85)

set.seed(17)
xgb_fit <- xgb.train(params = xgb_params,
                     data = xgb_train_matrix, verbose = 1,
                     nrounds = 155)

# Evaluate xgboost Model on Test Set

xgb_test <- xgb.DMatrix(
  data = as.matrix(test_xgb_set[,!names(test_xgb_set) %in% c("target")]))

xgb_predictions <- ifelse(predict(xgb_fit, newdata = xgb_test) > 0.5,1,0)

mean(test_xgb_set$target == xgb_predictions)

matrix_4 <- confusionMatrix(as.factor(xgb_predictions),
                            as.factor(test_xgb_set$target))
matrix_4$byClass

# Using the ROC Curve to evaluate the models

roc(as.numeric(test_set$target), as.numeric(predict(fit_1, test_set)))
roc(as.numeric(test_set$target), as.numeric(predict(fit_2, test_set)))
roc(as.numeric(test_set$target), as.numeric(predict(fit_3, test_set)))
roc(as.numeric(test_xgb_set$target), xgb_predictions)




