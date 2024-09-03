### Installing and loading necessary packages

#install.packages("mice")
install.packages("tidyverse")
#install.packages("readxl")
#install.packages("caret")
#install.packages("e1071")
#install.packages("glmnet")
#install.packages("MLmetrics")
#install.packages("caretEnsemble")
#install.packages("kernlab")
#install.packages("xgboost")
#install.packages("fastDummies")
install.packages("devtools")
#install.packages("pROC")
#install.packages("plotly")
#install.packages("tidymodels")
#install.packages("multiROC")

library(plotly)
library(tidymodels)
library(pROC)
library(devtools)
library(multiROC)
library(mice)
library(tidyverse)
library(readxl)

library(caret)
library(e1071)
library(glmnet)
library(MLmetrics)
library(caretEnsemble)
library(kernlab)
library(xgboost)
library(fastDummies)

### Loading the data set

# Setting working directory
setwd("./") #change it according to your specific path

# Importing our lizard data set
data_enyalius <- read_xlsx("herp-74-04-335_s02-edit.xlsx",
                           sheet = "Morphological data set",
                           range = NULL,
                           col_names = TRUE,
                           col_types = c("text","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","text","numeric","numeric","numeric","numeric","numeric","numeric"),
                           na = "NA")
class(data_enyalius)
str(data_enyalius)

# Transforming 'species name' and 'sex' into factors
data_enyalius$final_species_name <- as.factor(data_enyalius$final_species_name)
data_enyalius$sex <- as.factor(data_enyalius$sex)
# Transforming binary categories into factors
data_enyalius[,2:21] <- lapply(data_enyalius[,2:21] , factor)
str(data_enyalius)

data_enyalius_edit <- droplevels(data_enyalius[!data_enyalius$final_species_name == "bad_sample",])
data_enyalius_edit <- droplevels(data_enyalius_edit[!data_enyalius_edit$final_species_name == "unknown",])
keeps <- c("NPRC", "GF", "EGFS", "DHS", "TL4",
           "FL4", "nPVS", "nMS", "nVrS", "nVnS",
           "nPM", "ncPM", "nCRo", "nbNSpL", "ncIp", 
           "nbCoA", "nSbO", "nSpC", "TL", "n4TL",
           "nArT", "final_species_name")
data_enyalius_edit <- data_enyalius_edit[ ,keeps, drop = TRUE]
str(data_enyalius_edit)

### FABRICIUS VERIFICAR AQUI SE TODAS AS VARI�VEIS IMPORTANTES CONSTAM NA PLANILHA FINAL

### Multiple imputation of missing data with mice

# Plotting NA's
md.pattern(data_enyalius_edit, rotate.names = TRUE)

# Multiple imputation
imp <- mice(data_enyalius_edit,
            method = c("rf","rf","rf","rf","rf","rf","rf","rf","rf","rf",
                       "rf","rf","rf","rf","rf","rf","rf","rf","rf","rf",
                       "rf","rf"),
            maxit = 10, m = 10, print = FALSE, seed = 42)

# Listing the algorithms used for each trait
imp$meth

# Inspecting the quality of specific imputations
stripplot(imp, n4TL, pch = 20, xlab = "Imputation number", cex = 2)

# Prediction matrix
pred_matrix <- imp$pred

# Analyzing the convergence of imputations (the more lines are shuffled the better)
plot(imp)

# Exploring the matrix with imputed data
data_enyalius_imp <- complete(imp, 5)

md.pattern(data_enyalius_imp, rotate.names = TRUE)

### Processing data set

# Partitioning the data into training and test sets (0.7 and 0.3)
set.seed(42)
trainIndex <- createDataPartition(data_enyalius_imp$final_species_name,
                                  p = 0.7,
                                  list = FALSE) 
trainingSet <- data_enyalius_imp[trainIndex,] 
testSet <- data_enyalius_imp[-trainIndex,] 

# One hot encoding and creating a matrix only with the predictors, omitting the response variable.
dummyModel <- dummyVars(~ ., data = trainingSet) 
dummyModel 
trainingSetX <- as.data.frame(predict(dummyModel, newdata = trainingSet)) 

# Transforming the variables into values from 0 to 1, standardizing in z-scores.
rangeModel <- preProcess(trainingSetX, method = "range")
trainingSetX <- predict(rangeModel, newdata = trainingSetX)

# Adding the response variable to the training set
trainingSet <- cbind(trainingSet$final_species_name, trainingSetX)
names(trainingSet)[1] <- "final_species_name"

# Doing the same procedures with the test set 
testSet_dummy <- predict(dummyModel, testSet) 
testSet_range <- predict(rangeModel, testSet_dummy) 
testSet_range <- data.frame(testSet_range) 
testSet <- cbind(testSet$final_species_name, testSet_range) 
names(testSet) <- names(trainingSet) 
testSet$final_species_name <- as.factor(testSet$final_species_name)

trainingSet_final <- trainingSet[,-c(29:41)]
trainingSet_final[,2:13] <- lapply(trainingSet_final[,2:13] , factor)
str(trainingSet_final)

testSet_final <- testSet[,-c(29:41)]
testSet_final[,2:13] <- lapply(testSet_final[,2:13] , factor)
str(testSet_final)

### Training machine learning algorithms

### Defining a training control
ctrl <- trainControl(method = "repeatedcv", classProbs = TRUE, number = 5,
                     repeats = 5, summaryFunction = defaultSummary)

### Predicting

# rpart (decision tress)
set.seed(42) 
rpart <- train(final_species_name ~.,
               data = trainingSet_final,
               method = "rpart",
               metric = "Accuracy",
               trControl = ctrl,
               na.action = na.omit
               )
fitted_rpart <- predict(rpart, testSet_final)
length(testSet_final$final_species_name)
length(fitted_rpart)
cf_rpart <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fitted_rpart,
                mode = "everything")

# nnet
set.seed(42) 
nnet <- train(final_species_name ~.,
              data = trainingSet_final,
              method = "nnet",
              metric = "Accuracy",
              trControl = ctrl) 
fittednnet <- predict(nnet, testSet_final)
cf_nnet <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fittednnet,
                mode = "everything") 

# SVM
set.seed(42) 
svm_linear <- train(final_species_name ~.,
              data = trainingSet_final,
              method = "svmLinear",
              metric = "Accuracy",
              trControl = ctrl) 
fittedsvm <- predict(svm_linear, testSet_final)
cf_svm <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fittedsvm,
                mode = "everything")

# Random forest
set.seed(42) 
rf <- train(final_species_name ~.,
              data = trainingSet_final,
              method = "rf",
              metric = "Accuracy",
              trControl = ctrl) 
fittedrf <- predict(rf, testSet_final)
cf_rf <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fittedrf,
                mode = "everything")

# Random forest
set.seed(42) 
rf <- train(final_species_name ~.,
              data = trainingSet_final,
              method = "rf",
              metric = "Accuracy",
              trControl = ctrl) 
fittedrf <- predict(rf, testSet_final)
cf_rf <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fittedrf,
                mode = "everything")

# Logit
set.seed(42) 
logit <- train(final_species_name ~.,
              data = trainingSet_final,
              method = "LogitBoost",
              metric = "Accuracy",
              trControl = ctrl) 
fittedlogit <- predict(rf, testSet_final)
cf_logit <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fittedlogit,
                mode = "everything")

# KNN
set.seed(42) 
knn_train <- train(final_species_name ~.,
              data = trainingSet_final,
              method = "knn",
              metric = "Accuracy",
              trControl = ctrl) 
fittedknn <- predict(knn_train, testSet_final)
cf_knn <- confusionMatrix(reference = testSet_final$final_species_name,
                data = fittedknn,
                mode = "everything") 

### Comparing the performance of different ML algorithms
resamps <- resamples(list(NNET = nnet,
                          RPART = rpart,
                          RF = rf,
                          SVMLINEAR = svm_linear,
                          LOGITBOOST = logit,
                          KNN = knn_train))
resamps

# Boxplot comparing different algorithms
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4) 
theme1$plot.symbol$pch = 16 
theme1$plot.line$col = rgb(1, 0, 0, .7) 
theme1$plot.line$lwd <- 2 
trellis.par.set(theme1) 
bwplot(resamps, layout = c(2, 1)) 

resamps2 <- resamples(list(RF = rf,
                          LOGITBOOST = logit))
resamps2

# xyplot plots to compare models 
xyplot(resamps2, models=c("RF", "LOGITBOOST"))

### ROC curves
# Probabilities prediction
y_scores_logit <- predict(logit, type = 'prob')
pred <-"pred"
make.names(paste(names(y_scores_logit), pred))
colnames(y_scores_logit)[c(1:13)] <- make.names(paste(names(y_scores_logit)[c(1:13)], pred))

# One hot encode the labels in order to plot them
y_onehot <- dummy_cols(trainingSet_final$final_species_name)
categories <- unique(trainingSet_final$trainingSet_final)
categories
colnames(y_onehot) <- c("drop","E_boulengeri", "E_brasiliensis",
                        "E_capetinga", "E_perditus2", "E_bibronii",
                        "E_bilineatus", "E_catenatus2", "E_erythroceneus",
                        "E_pictus", "E_iheringii","E_perditus",
                        "E_leechii", "E_catenatus")
y_onehot <- subset(y_onehot, select = -c(drop))

# Getting scores for each species
scores_logit_binded <- cbind(y_scores_logit, y_onehot)
str(scores_logit_binded)

scores_logit_binded$E_boulengeri <- as.factor(scores_logit_binded$E_boulengeri)
roc_boulengeri <- roc_curve(data = scores_logit_binded, E_boulengeri, E_boulengeri.pred)
roc_boulengeri
roc_boulengeri$specificity <- 1 - roc_boulengeri$specificity
colnames(roc_boulengeri) <- c('threshold', 'tpr', 'fpr')
auc_boulengeri <- roc_auc(data =  scores_logit_binded, E_boulengeri, E_boulengeri.pred)
auc_boulengeri <- auc_boulengeri$.estimate
boulengeri <- paste('boulengeri (AUC=',toString(round(1-auc_boulengeri,2)),')',sep = '')

scores_logit_binded$E_brasiliensis <- as.factor(scores_logit_binded$E_brasiliensis)
roc_brasiliensis <- roc_curve(data = scores_logit_binded, E_brasiliensis, E_brasiliensis.pred)
roc_brasiliensis$specificity <- 1 - roc_brasiliensis$specificity
colnames(roc_brasiliensis) <- c('threshold', 'tpr', 'fpr')
auc_brasiliensis <- roc_auc(data = scores_logit_binded, E_brasiliensis, E_brasiliensis.pred)
auc_brasiliensis <- auc_brasiliensis$.estimate
brasiliensis <- paste('brasiliensis (AUC=',toString(round(1-auc_brasiliensis,2)),')', sep = '')

scores_logit_binded$E_capetinga <- as.factor(scores_logit_binded$E_capetinga)
roc_capetinga <- roc_curve(data = scores_logit_binded, E_capetinga, E_capetinga.pred)
roc_capetinga$specificity <- 1 - roc_capetinga$specificity
colnames(roc_capetinga) <- c('threshold', 'tpr', 'fpr')
auc_capetinga <- roc_auc(data = scores_logit_binded, E_capetinga, E_capetinga.pred)
auc_capetinga <- auc_capetinga$.estimate
capetinga <- paste('capetinga (AUC=',toString(round(1-auc_capetinga,2)),')',sep = '')

scores_logit_binded$E_perditus2 <- as.factor(scores_logit_binded$E_perditus2)
roc_perditus2 <- roc_curve(data = scores_logit_binded, E_perditus2, E_perditus2.pred)
roc_perditus2$specificity <- 1 - roc_perditus2$specificity
colnames(roc_perditus2) <- c('threshold', 'tpr', 'fpr')
auc_perditus2 <- roc_auc(data = scores_logit_binded, E_perditus2, E_perditus2.pred)
auc_perditus2 <- auc_perditus2$.estimate
perditus2 <- paste('perditus2 (AUC=',toString(round(1-auc_perditus2,2)),')',sep = '')

scores_logit_binded$E_bibronii <- as.factor(scores_logit_binded$E_bibronii)
roc_bibronii <- roc_curve(data = scores_logit_binded, E_bibronii, E_bibronii.pred)
roc_bibronii$specificity <- 1 - roc_bibronii$specificity
colnames(roc_bibronii) <- c('threshold', 'tpr', 'fpr')
auc_bibronii <- roc_auc(data = scores_logit_binded, E_bibronii, E_bibronii.pred)
auc_bibronii <- auc_bibronii$.estimate
bibronii <- paste('bibronii (AUC=',toString(round(1-auc_bibronii, 2)),')',sep = '')

scores_logit_binded$E_bilineatus <- as.factor(scores_logit_binded$E_bilineatus)
roc_bilineatus <- roc_curve(data = scores_logit_binded, E_bilineatus, E_bilineatus.pred)
roc_bilineatus$specificity <- 1 - roc_bilineatus$specificity
colnames(roc_bilineatus) <- c('threshold', 'tpr', 'fpr')
auc_bilineatus <- roc_auc(data = scores_logit_binded, E_bilineatus, E_bilineatus.pred)
auc_bilineatus <- auc_bilineatus$.estimate
bilineatus <- paste('bilineatus (AUC=',toString(round(1-auc_bilineatus, 2)),')',sep = '')

scores_logit_binded$E_catenatus2 <- as.factor(scores_logit_binded$E_catenatus2)
roc_catenatus2 <- roc_curve(data = scores_logit_binded, E_catenatus2, E_catenatus2.pred)
roc_catenatus2$specificity <- 1 - roc_catenatus2$specificity
colnames(roc_catenatus2) <- c('threshold', 'tpr', 'fpr')
auc_catenatus2 <- roc_auc(data = scores_logit_binded, E_catenatus2, E_catenatus2.pred)
auc_catenatus2 <- auc_catenatus2$.estimate
catenatus2 <- paste('catenatus2 (AUC=',toString(round(1-auc_catenatus2, 2)),')',sep = '')

scores_logit_binded$E_erythroceneus <- as.factor(scores_logit_binded$E_erythroceneus)
roc_erythroceneus <- roc_curve(data = scores_logit_binded, E_erythroceneus, E_erythroceneus.pred)
roc_erythroceneus$specificity <- 1 - roc_erythroceneus$specificity
colnames(roc_erythroceneus) <- c('threshold', 'tpr', 'fpr')
auc_erythroceneus <- roc_auc(data = scores_logit_binded, E_erythroceneus, E_erythroceneus.pred)
auc_erythroceneus <- auc_erythroceneus$.estimate
erythroceneus <- paste('erythroceneus (AUC=',toString(round(1-auc_erythroceneus, 2)),')',sep = '')

scores_logit_binded$E_pictus <- as.factor(scores_logit_binded$E_pictus)
roc_pictus <- roc_curve(data = scores_logit_binded, E_pictus, E_pictus.pred)
roc_pictus$specificity <- 1 - roc_pictus$specificity
colnames(roc_pictus) <- c('threshold', 'tpr', 'fpr')
auc_pictus <- roc_auc(data = scores_logit_binded, E_pictus, E_pictus.pred)
auc_pictus <- auc_pictus$.estimate
pictus <- paste('pictus (AUC=',toString(round(1-auc_pictus, 2)),')',sep = '')

scores_logit_binded$E_iheringii <- as.factor(scores_logit_binded$E_iheringii)
roc_iheringii <- roc_curve(data = scores_logit_binded, E_iheringii, E_iheringii.pred)
roc_iheringii$specificity <- 1 - roc_iheringii$specificity
colnames(roc_iheringii) <- c('threshold', 'tpr', 'fpr')
auc_iheringii  <- roc_auc(data = scores_logit_binded, E_iheringii, E_iheringii.pred)
auc_iheringii <- auc_iheringii$.estimate
iheringii <- paste('iheringii (AUC=',toString(round(1-auc_iheringii, 2)),')',sep = '')

scores_logit_binded$E_perditus <- as.factor(scores_logit_binded$E_perditus)
roc_perditus <- roc_curve(data = scores_logit_binded, E_perditus, E_perditus.pred)
roc_perditus$specificity <- 1 - roc_perditus$specificity
colnames(roc_perditus) <- c('threshold', 'tpr', 'fpr')
auc_perditus  <- roc_auc(data = scores_logit_binded, E_perditus, E_perditus.pred)
auc_perditus <- auc_perditus$.estimate
perditus <- paste('perditus (AUC=',toString(round(1-auc_perditus, 2)),')',sep = '')

scores_logit_binded$E_leechii <- as.factor(scores_logit_binded$E_leechii)
roc_leechii  <- roc_curve(data = scores_logit_binded, E_leechii, E_leechii.pred)
roc_leechii$specificity <- 1 - roc_leechii$specificity
colnames(roc_leechii) <- c('threshold', 'tpr', 'fpr')
auc_leechii  <- roc_auc(data = scores_logit_binded, E_leechii, E_leechii.pred)
auc_leechii <- auc_leechii$.estimate
leechii <- paste('leechii (AUC=',toString(round(1-auc_leechii, 2)),')',sep = '')

scores_logit_binded$E_catenatus <- as.factor(scores_logit_binded$E_catenatus)
roc_catenatus  <- roc_curve(data = scores_logit_binded, E_catenatus, E_catenatus.pred)
roc_catenatus$specificity <- 1 - roc_catenatus$specificity
colnames(roc_catenatus) <- c('threshold', 'tpr', 'fpr')
auc_catenatus <- roc_auc(data = scores_logit_binded, E_catenatus, E_catenatus.pred)
auc_catenatus <- auc_catenatus$.estimate
catenatus <- paste('catenatus (AUC=',toString(round(1-auc_catenatus, 2)),')',sep = '')

# Create an empty figure, and iteratively add a line for each class
fig <- plot_ly()%>%
  add_segments(x = 0, xend = 1, y = 0, yend = 1, line = list(dash = "dash", color = 'black'), showlegend = FALSE) %>%
  add_trace(data = roc_bibronii,x = ~fpr, y = ~tpr, mode = 'lines', name = bibronii, type = 'scatter')%>%
  add_trace(data = roc_bilineatus,x = ~fpr, y = ~tpr, mode = 'lines', name = bilineatus, type = 'scatter')%>%
  add_trace(data = roc_boulengeri,x = ~fpr, y = ~tpr, mode = 'lines', name = boulengeri, type = 'scatter')%>%
  add_trace(data = roc_brasiliensis,x = ~fpr, y = ~tpr, mode = 'lines', name = brasiliensis, type = 'scatter')%>%
  add_trace(data = roc_capetinga,x = ~fpr, y = ~tpr, mode = 'lines', name = capetinga, type = 'scatter')%>%
  add_trace(data = roc_catenatus,x = ~fpr, y = ~tpr, mode = 'lines', name = catenatus, type = 'scatter')%>%
  add_trace(data = roc_catenatus2,x = ~fpr, y = ~tpr, mode = 'lines', name = catenatus2, type = 'scatter')%>%
  add_trace(data = roc_erythroceneus,x = ~fpr, y = ~tpr, mode = 'lines', name = erythroceneus, type = 'scatter')%>%
  add_trace(data = roc_iheringii,x = ~fpr, y = ~tpr, mode = 'lines', name = iheringii, type = 'scatter')%>%
  add_trace(data = roc_leechii,x = ~fpr, y = ~tpr, mode = 'lines', name = leechii, type = 'scatter')%>%
  add_trace(data = roc_perditus,x = ~fpr, y = ~tpr, mode = 'lines', name = perditus, type = 'scatter')%>%
  add_trace(data = roc_perditus2,x = ~fpr, y = ~tpr, mode = 'lines', name = perditus2, type = 'scatter')%>%
  add_trace(data = roc_pictus,x = ~fpr, y = ~tpr, mode = 'lines', name = pictus, type = 'scatter')%>%
  layout(xaxis = list(
    title = "False Positive Rate"
  ), yaxis = list(
    title = "True Positive Rate"
  ),legend = list(x = 100, y = 0.5))
fig
