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
library(randomForest)
library(rpart)
library(RSNNS)
library(klaR)
library(kernlab)


### Fun��o
comp_alg <- function(data, list_alg, train_val, cv_folds, seed) {  
  # Definir seed para reprodutibilidade
  set.seed(seed)
  
  # Dividir os dados em treino e teste
  in_training <- createDataPartition(y = data[, ncol(data)],
                                     p = train_val, 
                                     list = F)
  train_set <- data[in_training,]
  test_set <- data[-in_training,]
  
  final_train_set <- train_set[,-ncol(train_set)] # Assume que a vari�vel resposta est� na �ltima coluna
  dependent_variable <- train_set[,ncol(train_set)]
  dependent_test_set <- test_set[,ncol(test_set)]
  
  # Definir controle do treino (Valida��o Cruzada com n folds)
  train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 10) 
  
  # Lista para armazenar os modelos treinados
  model_results <- list()
  
  # Loop para treinar cada algoritmo
  for (alg in list_alg) {
    print(paste("Treinando o modelo com algoritmo:", alg))
    
    # Treinar o modelo com o algoritmo atual
    model <- caret::train(x = final_train_set,
                   y = dependent_variable, 
                   method = alg, 
                   trControl = train_control, 
                   metric = "Accuracy")
    
    # Armazenar o modelo treinado na lista
    model_results[[alg]] <- model
  }
  
  # Avaliar o desempenho no conjunto de teste #### MUDAR
  evaluation_results <- list()
  for (alg in list_alg) {
    print(paste("Avaliando o modelo com algoritmo:", alg))
    predictions <- predict(model_results[[alg]], newdata = test_set)
    confusion_matrix <- confusionMatrix(predictions, dependent_test_set)
    evaluation_results[[alg]] <- confusion_matrix
    print(confusion_matrix)
  }
  
  # Retornar os resultados dos modelos e as avalia��es
  return(list(models = model_results, evaluations = evaluation_results))
}

# Testando a fun��o com o dataset 'iris'
library(datasets)
iris <- datasets::iris

default_mllist <- c("rf", "knn", "rpart", "nb", "mlp")
results <- comp_alg(data = iris,
                    list_alg = default_mllist,
                    train_val = 0.75,
                    seed = 123,
                    cv_folds = 5)



#results$acuracia
#Acuracia RF: 0.97


### A fazer:
#1 Especificar um valor default para o trainControl dentro da nossa fun��o, 
# mas dar a liberdade ao usu�rio de colocar o seu pr�prio input

#2 Mudar output da fun��o. Encontrar uma forma em que a fun��o retorne
# a acur�cia de cada modelo, rankeando do melhor para o pior