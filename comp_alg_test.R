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

### Função
comp_alg <- function(data, 
                     target,
                     list_alg = c("rpart", "nnet", "svmLinear", "rf", "LogitBoost", "knn") , 
                     train_val = 0.75, 
                     cv_folds = 5, 
                     seed = 123, 
                     number = 5, 
                     repeats = 10,
                     verbose = TRUE) 
{  
  # Definir seed para reprodutibilidade
  set.seed(seed)
  
  # Dividir os dados em treino e teste
  in_training <- createDataPartition(y = data[[target]],
                                     p = train_val, 
                                     list = FALSE)
  train_set <- data[in_training,]
  test_set <- data[-in_training,]
  
  #final_train_set <- train_set[,-ncol(train_set)] # Assume que a variável resposta está na última coluna
  #dependent_variable <- train_set[,ncol(train_set)]
  #dependent_test_set <- test_set[,ncol(test_set)]
  
  final_train_set <- dplyr::select(train_set, -all_of(target))  # preditores
  dependent_variable <- train_set[[target]]                     # resposta de treino
  dependent_test_set <- test_set[[target]]                     # resposta de teste
  
  # Definir controle do treino (Validação Cruzada com n folds)
  train_control <- trainControl(method = "repeatedcv", number = number, repeats = repeats) 
  
  # Lista para armazenar os modelos treinados
  model_results <- list()
  evaluation_results <- list()
  
  accuracy <- c()
  accuracy_sd <- c()
  kappa <- c()
  kappa_sd <- c()
  
  # Loop para treinar cada algoritmo
  for (alg in list_alg) {
    cat("Training algorithm:", alg, "\n")
    
    # Treinar o modelo com o algoritmo atual
    model <- caret::train(x = final_train_set,
                          y = dependent_variable, 
                          method = alg, 
                          trControl = train_control, 
                          metric = "Accuracy")
    
    # Armazenar o modelo treinado na lista
    model_results[[alg]] <- model
  
    # Previsoes no teste
    predictions <- predict(model, dplyr::select(test_set, -all_of(target)))
    confusion_matrix <- confusionMatrix(predictions, dependent_test_set)
    evaluation_results[[alg]] <- confusion_matrix
    if (verbose) print(confusion_matrix)
  
    # Obter o modelo treinado com maior acuracia
    best_index <- which.max(model$results$Accuracy)
    accuracy[alg] <- model$results$Accuracy[best_index]
    accuracy_sd[alg] <- model$results$AccuracySD[best_index]
    kappa[alg] <- model$results$Kappa[best_index]
    kappa_sd[alg] <- model$results$KappaSD[best_index]
  }
  
  res_scores <- data.frame(accuracy = accuracy,
                           accuracy_sd = accuracy_sd,
                           kappa = kappa,
                           kappa_sd = kappa_sd)
  
  res_scores_ordered <- res_scores[order(res_scores$accuracy, decreasing = TRUE), ]
  best_model <- res_scores_ordered$algorithm[1]
  
  # Retornar os resultados dos modelos e as avaliações
  return(list(
    model_names = list_alg, 
    models = model_results, 
    evaluations = evaluation_results,
    metrics = res_scores_ordered,
    best_model = best_model
    ))
}

# Testando a função com o dataset 'iris'
library(datasets)
iris <- datasets::iris

results <- comp_alg(data = iris,
                    target = "Species",
                    train_val = 0.8,
                    seed = 42,
                    cv_folds = 5,
                    verbose = FALSE)

results$best_model
class(results$metrics)
results$metrics


#print(results)
#results$acuracia
#Acuracia RF: 0.97

### A fazer:
## 1 Combinar as duas funções em uma só, colocando o for loop da acurácia dentro da função anterior

## 2 Dar uma opção de quiet mode OU verbose, para visualizar ou não o andamento da função no console
# argumento na função para reportar na tela ou não

## 3 Encontrar um jeito de fazer a divisão entre treino e teste de forma que não seja
# necessário assumir que a variável resposta esteja na última coluna

## OBS: NÃO É MAIS NECESSÁRIO
# Usar a função postResample para obter acurácia e kappa para cada um dos algoritmos
# Armazenar estes resultados em uma lista separada
# Transformar esta lista em data frame e colocar isso no return da função
# Links uteis:
# https://topepo.github.io/caret/measuring-performance.html#measures-for-class-probabilities
# https://www.rdocumentation.org/packages/purrr/versions/0.2.5/topics/map
