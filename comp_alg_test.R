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
                     list_alg = c("rpart", "nnet", "svmLinear", "rf", "LogitBoost", "knn") , 
                     train_val = 0.75, 
                     cv_folds, 
                     seed = 123, 
                     number = 5, 
                     repeats = 10,
                     verbose) 
{  
  # Definir seed para reprodutibilidade
  set.seed(seed)
  
  # Dividir os dados em treino e teste
  in_training <- createDataPartition(y = data[, ncol(data)],
                                     p = train_val, 
                                     list = F)
  train_set <- data[in_training,]
  test_set <- data[-in_training,]
  
  final_train_set <- train_set[,-ncol(train_set)] # Assume que a variável resposta está na última coluna
  dependent_variable <- train_set[,ncol(train_set)]
  dependent_test_set <- test_set[,ncol(test_set)]
  
  # Definir controle do treino (Validação Cruzada com n folds)
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
    if (verbose)
      print(confusion_matrix)
  }
  
  accuracy <- c()
  accuracy_sd <- c()
  kappa <- c()
  kappa_sd <- c()
  for (x in results$m_names) {
    accuracy[x] <- max(model_results[[x]]$results["Accuracy"])
    accuracy_sd[x] <- max(model_results[[x]]$results["AccuracySD"])
    kappa[x] <- model_results[[x]]$results[1, "Kappa"]
    kappa_sd[x] <- model_results[[x]]$results[1, "KappaSD"]
  }
  
  res_scores <- data.frame(accuracy = accuracy,
                           accuracy_sd = accuracy_sd,
                           kappa = kappa,
                           kappa_sd = kappa_sd)
  
  for (x in results$m_names) {
    max_accuracy <- max(model_results[[x]]$results["Accuracy"])
    data <- subset(model_results[[x]]$results, Accuracy == max_accuracy)
    data <- select(data, Accuracy, Kappa, AccuracySD, KappaSD)
    res_scores_df <- rbind(res_scores_df, data)
  }
  print("Ok!")
  
  # res_scores_order <- res_scores[order(res_scores$accuracy, decreasing = T),]
  # Retornar os resultados dos modelos e as avaliações
  return(list(m_names = list_alg, 
              models = model_results, 
              evaluations = evaluation_results,
              order1 = res_scores[order(res_scores$accuracy, decreasing = T),],))
              #order2 = res_scores_df[order(res_scores_df$Accuracy, decreasing = T),]))
}

# Testando a função com o dataset 'iris'
library(datasets)
iris <- datasets::iris

seed <- 42

results <- comp_alg(data = iris,
                    train_val = 0.8,
                    seed = seed,
                    cv_folds = 5,
                    verbose = FALSE)

results$order1
#results$order2

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
