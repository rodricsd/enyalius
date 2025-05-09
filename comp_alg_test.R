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

### Fun��o
comp_alg <- function(data, 
                     list_alg = c("rpart", "nnet", "svmLinear", "rf", "LogitBoost", "knn") , 
                     train_val = 0.75, 
                     cv_folds, 
                     seed = 123, 
                     number = 5, 
                     repeats = 10) 
  {  
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
  train_control <- trainControl(method = "repeatedcv", number = number, repeats = repeats) 
  
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
  return(list(m_names = list_alg, models = model_results, evaluations = evaluation_results))
  }

# Testando a fun��o com o dataset 'iris'
library(datasets)
iris <- datasets::iris

seed <- 42

results <- comp_alg(data = iris,
                    train_val = 0.8,
                    seed = seed,
                    cv_folds = 5)

#df_results <- do.call(rbind.data.frame, results$models)

accuracy_list <- list()
kappa_list <- list()
for (x in results$m_names) {
  accuracy_list[[x]] <- results$models$results[1, "Accuracy"]
  kappa_list[[x]] <- results$models$results[1, "Kappa"]
}

res_scores <- data.frame(models_names = results$m_names,
                         accuracy = accuracy_list,
                         kappa = kappa_list)
summary(res_scores)


#print(results)

#results$acuracia
#Acuracia RF: 0.97


### A fazer:
## 1 Usar a fun��o postResample para obter acur�cia e kappa para cada um dos algoritmos
# Armazenar estes resultados em uma lista separada
# Transformar esta lista em data frame e colocar isso no return da fun��o
# Links uteis:
# https://topepo.github.io/caret/measuring-performance.html#measures-for-class-probabilities
# https://www.rdocumentation.org/packages/purrr/versions/0.2.5/topics/map

## 2 No dataframe reportado como resultado, rankear os modelos, do melhor para o pior

## 3 Encontrar um jeito de fazer a divis�o entre treino e teste de forma que n�o seja
# necess�rio assumir que a vari�vel resposta est� na �ltima coluna