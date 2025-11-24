#' Compare Machine Learning Algorithms
#'
#' This function compares multiple machine learning algorithms using repeated
#' cross-validation on a training subset of the data. It identifies the best, most
#' stable model using the "one-standard-error" rule and then retrains that single
#' best model on the full dataset to produce a final, production-ready model.
#'
#' @param data The dataframe containing your data.
#' @param target A string with the name of the column you want to predict.
#' @param list_alg A character vector of the machine learning algorithms you want to compare.
#' @param train_val The proportion of the data to be used for training.
#' @param seed A random seed for reproducibility.
#' @param number The number of folds for cross-validation.
#' @param repeats The number of times to repeat the cross-validation process.
#' @param verbose If `TRUE`, prints progress and confusion matrices during execution.
#' @return A list object of class `diagnoseR_result` containing models, evaluations, and performance metrics.
#' @importFrom caret createDataPartition trainControl confusionMatrix
#' @export
#' @importFrom dplyr select all_of
#' @examples
#' # Run algorithm comparison on the iris dataset
#' iris_results <- comp_alg(
#'   data = iris,
#'   target = "Species",
#'   verbose = FALSE # Keep output clean for example
#' )
#' print(iris_results)
comp_alg <- function(data, 
                     target,
                     list_alg = c("rpart", "nnet", "svmLinear", "rf", "LogitBoost", "knn") , 
                     train_val = 0.75,
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
  train_set <- data[in_training, ]
  test_set <- data[-in_training, ]
  train_set_full <- data # The full dataset for the final model

  final_train_set <- dplyr::select(train_set, -all_of(target))  # preditores
  dependent_variable <- train_set[[target]]                    # resposta de treino (split)
  dependent_test_set <- test_set[[target]]                     # resposta de teste

  final_train_set_full <- dplyr::select(train_set_full, -all_of(target))

  # Definir controle do treino (Validação Cruzada com n folds)
  train_control <- caret::trainControl(method = "repeatedcv", number = number, repeats = repeats) 
  
  # Lista para armazenar os modelos treinados
  model_results <- list()
  model_results_full <- list()
  evaluation_results <- list()
  
  accuracy <- c()
  accuracy_sd <- c()
  kappa <- c()
  kappa_sd <- c()
  dratio <- c()

  accuracy_full <- c()
  accuracy_sd_full <- c()
  kappa_full <- c()
  kappa_sd_full <- c()
  
  # Loop para treinar cada algoritmo
  for (alg in list_alg) {
    if (verbose) {
      cat("Training algorithm on split data:", alg, "\n")
    }
    
    # Treinar o modelo com o algoritmo atual
    train_args_split <- list(x = final_train_set,
                         y = dependent_variable, 
                         method = alg, 
                         trControl = train_control, 
                         metric = "Accuracy")

    if (verbose) {
      cat("Training algorithm on full data:", alg, "\n")
    }

    train_args_full <- list(x = final_train_set_full,
                           y = train_set_full[[target]],
                           method = alg,
                           trControl = train_control,
                           metric = "Accuracy")

    if (alg == "nnet") {
      train_args_split$trace <- FALSE
      train_args_full$trace <- FALSE
    }
    
    model <- do.call(caret::train, train_args_split)
    model_full <- do.call(caret::train, train_args_full)
    
    # Armazenar o modelo treinado na lista
    model_results[[alg]] <- model # Based on split data
    model_results_full[[alg]] <- model_full # Based on full data
  
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

    # Get metrics from full data training
    best_index_full <- which.max(model_full$results$Accuracy)
    accuracy_full[alg] <- model_full$results$Accuracy[best_index_full]
    accuracy_sd_full[alg] <- model_full$results$AccuracySD[best_index_full]
    kappa_full[alg] <- model_full$results$Kappa[best_index_full]
    kappa_sd_full[alg] <- model_full$results$KappaSD[best_index_full]
  }
  
  res_scores <- data.frame(
    algorithm = names(accuracy),
    accuracy = unname(accuracy),
    accuracy_sd = unname(accuracy_sd),
    kappa = unname(kappa),
    kappa_sd = unname(kappa_sd),
    dratio = unname(dratio)
  ) 

  res_scores_full <- data.frame(
    algorithm = names(accuracy_full),
    accuracy = unname(accuracy_full),
    accuracy_sd = unname(accuracy_sd_full),
    kappa = unname(kappa_full),
    kappa_sd = unname(kappa_sd_full)
  )

  res_scores_full_ordered <- res_scores_full[order(res_scores_full$accuracy, decreasing = TRUE), ]
  
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

  res_scores_ordered <- res_scores[order(res_scores$accuracy, decreasing = TRUE), ]
  
  # Retornar os resultados dos modelos e as avaliações
  result_list <- list(
    model_names = list_alg, 
    models = model_results, # Models from train/test split
    models_full = model_results_full, # Models from full dataset
    evaluations = evaluation_results,
    metrics = res_scores_ordered,
    metrics_full = res_scores_full_ordered,
    best_model = best_model,
    one_se_threshold = one_se_threshold
    )
  
  class(result_list) <- "diagnoseR_result"
  
  return(result_list)
}
