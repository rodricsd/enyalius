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

#' Compare Algoritmos de Amprendizado de Máquina.
#'
#' Esta função compara multiplos algoritmos de aprendizado de máquina.
#' @importFrom caret createDataPartition trainControl confusionMatrix
#' @export
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
  in_training <- caret::createDataPartition(y = data[[target]],
                                     p = train_val, 
                                     list = FALSE)
  train_set <- data[in_training,]
  test_set <- data[-in_training,]
  
  final_train_set <- dplyr::select(train_set, -all_of(target))  # preditores
  dependent_variable <- train_set[[target]]                     # resposta de treino
  dependent_test_set <- test_set[[target]]                     # resposta de teste
  
  # Definir controle do treino (Validação Cruzada com n folds)
  train_control <- caret::trainControl(method = "repeatedcv", number = number, repeats = repeats) 
  
  # Lista para armazenar os modelos treinados
  model_results <- list()
  evaluation_results <- list()
  
  accuracy <- c()
  accuracy_sd <- c()
  kappa <- c()
  kappa_sd <- c()
  dratio <- c()
  
  # Loop para treinar cada algoritmo
  for (alg in list_alg) {
    cat("Training algorithm:", alg, "\n")
    
    # Treinar o modelo com o algoritmo atual
    train_args <- list(x = final_train_set,
                         y = dependent_variable, 
                         method = alg, 
                         trControl = train_control, 
                         metric = "Accuracy")
    
    if (alg == "nnet") {
      train_args$trace <- FALSE
    }
    
    model <- do.call(caret::train, train_args)
    
    # Armazenar o modelo treinado na lista
    model_results[[alg]] <- model
  
    # Previsoes no teste
    predictions <- predict(model, dplyr::select(test_set, -all_of(target)))
    confusion_matrix <- caret::confusionMatrix(predictions, dependent_test_set)
    evaluation_results[[alg]] <- confusion_matrix
    if (verbose) print(confusion_matrix)
  
    # Obter o modelo treinado com maior acuracia
    best_index <- which.max(model$results$Accuracy)
    accuracy[alg] <- model$results$Accuracy[best_index]
    accuracy_sd[alg] <- model$results$AccuracySD[best_index]
    kappa[alg] <- model$results$Kappa[best_index]
    kappa_sd[alg] <- model$results$KappaSD[best_index]
    dratio[alg] <- (accuracy_sd[alg]/accuracy[alg]) - accuracy_sd[alg]
  }
  
  res_scores <- data.frame(
    algorithm = names(accuracy),
    accuracy = unname(accuracy),
    accuracy_sd = unname(accuracy_sd),
    kappa = unname(kappa),
    kappa_sd = unname(kappa_sd),
    dratio = unname(dratio)
  )
  
  #res_scores_ordered <- res_scores[order(res_scores$accuracy, decreasing = TRUE), ]
  #res_scores_ordered <- res_scores[order(res_scores$dratio, decreasing = FALSE), ]
  #best_model <- res_scores_ordered$algorithm[1]

  # Implementar a regra oneSE para seleção de modelo
  res_scores$accuracy_se <- res_scores$accuracy_sd / sqrt(number * repeats)
  
  best_acc_index <- which.max(res_scores$accuracy)
  best_accuracy <- res_scores$accuracy[best_acc_index]
  best_se <- res_scores$accuracy_se[best_acc_index]
  
  one_se_threshold <- best_accuracy - best_se
  
  res_scores$candidate <- res_scores$accuracy >= one_se_threshold
  
  # Dos candidatos, escolher aquele com a melhor estabilidade (menor sd) 
  candidates <- res_scores[res_scores$candidate, ]
  final_choice <- candidates[which.min(candidates$accuracy_sd), ]
  best_model <- as.character(final_choice$algorithm)
  
  # Retornar os resultados dos modelos e as avaliações
  result_list <- list(
    model_names = list_alg, 
    models = model_results, 
    evaluations = evaluation_results,
    metrics = res_scores,
    best_model = best_model,
    one_se_threshold = one_se_threshold
    )
  
  class(result_list) <- "diagnoseR_result"
  
  return(result_list)
}

#' @export
print.diagnoseR_result <- function(x, ...) {
  cat("\n--- Resultados da Regra do Erro Padrão ---\n")
  cat("Threshold (Melhor Acuracia - Melhor SE):", format(x$one_se_threshold, digits = 4), "\n")
  cat("Melhor modelo (mais estável dos candidatos):", x$best_model, "\n\n")
  cat("Metricas de Performance:\n")
  
  # Reordenar colunas para clareza e formatar digitos
  metrics_to_print <- x$metrics[, c("algorithm", "candidate", "accuracy", "accuracy_sd", "kappa", "dratio", "accuracy_se")]
  
  print(metrics_to_print, row.names = FALSE)
  
  cat("-------------------------------------\n")
}

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
