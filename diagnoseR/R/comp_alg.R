#' Compare Machine Learning Algorithms
#'
#' This function compares multiple machine learning algorithms using repeated
#' cross-validation on a training subset of the data. It identifies the best, most
#' stable model using the "one-standard-error" rule and then retrains that single
#' best model on the full dataset to produce a final, production-ready model.
#' If `train_val` is set to 1.0, the function skips the train/test split and
#' performs model comparison using repeated cross-validation on the entire dataset.
#'
#' @param data The dataframe containing your data.
#' @param target A string with the name of the column to predict.
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
  # --- 1. Validação de Entradas ---
  if (!is.data.frame(data)) stop("O parâmetro 'data' deve ser um data.frame.")
  if (!is.character(target) || length(target) != 1) stop("'target' deve ser uma string com o nome da coluna alvo.")
  if (!target %in% names(data)) stop(paste("A coluna alvo '", target, "' não foi encontrada no data.frame.", sep=""))
  if (!is.numeric(train_val) || train_val <= 0 || train_val > 1) {
    stop("'train_val' deve ser um número entre 0 (exclusive) e 1 (inclusive).")
  }
  
  # Para classificação, a variável alvo deve ser um fator.
  if (!is.factor(data[[target]])) {
    if (verbose) message(paste("Convertendo a coluna alvo '", target, "' para fator.", sep=""))
    data[[target]] <- as.factor(data[[target]])
  }
  
  # Verificar se há dados suficientes para a validação cruzada
  min_class_count <- min(table(data[[target]]))
  if (min_class_count < number) {
    warning(paste0("O número de folds de CV ('number'=", number, ") é maior que o número de amostras na menor classe (", min_class_count, "). Isso pode causar erros. Considere reduzir 'number'."))
  }

  # Definir seed para reprodutibilidade
  set.seed(seed)

  # --- 2. Lógica de `train_val` ---
  max_train_val <- 1 - (number / nrow(data))
  
  if (train_val > max_train_val) {
    warning(paste0("O valor de 'train_val' (", train_val, ") é muito alto e pode causar erros em alguns modelos de ML com o 'caret'.\n",
                   "Ajustando 'train_val' para o valor máximo seguro de ", round(max_train_val, 4), " para este conjunto de dados.\n",
                   "O modelo final ainda será treinado com 100% dos dados."))
    train_val <- max_train_val
  }

  # Se train_val < 1, dividir os dados em treino e teste.
  # Se train_val == 1, usar o dataset completo para treino (a avaliação será via CV).
  if (train_val < 1.0) {
    in_training <- caret::createDataPartition(y = data[[target]],
                                       p = train_val, 
                                       list = FALSE)
    train_set <- data[in_training, ]
    test_set <- data[-in_training, ]
  } else {
    train_set <- data
    test_set <- NULL # Não haverá conjunto de teste
  }

  final_train_set <- dplyr::select(train_set, -all_of(target))  # preditores
  dependent_variable <- train_set[[target]]                    # resposta de treino (split)

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
    if (verbose) cat("Training algorithm:", alg, "\n")
    
    # Treinar o modelo com o algoritmo atual
    train_args <- list(x = final_train_set,
                         y = dependent_variable, 
                         method = alg, 
                         trControl = train_control, 
                         metric = "Accuracy")

    if (alg == "nnet") {
      train_args$trace <- FALSE
    }
    
    # Usar tryCatch para capturar erros durante o treino de um modelo específico
    model <- tryCatch({
      do.call(caret::train, train_args)
    }, error = function(e) {
      warning(paste("Falha ao treinar o algoritmo '", alg, "'. Pulando. Erro: ", e$message, sep=""))
      return(NULL) # Retorna NULL se o treino falhar
    })

    # Se o modelo falhou, pule para o próximo algoritmo
    if (is.null(model)) next

    # Armazenar o modelo treinado na lista
    model_results[[alg]] <- model
  
    # Previsões no teste (apenas se houver conjunto de teste)
    if (!is.null(test_set) && nrow(test_set) > 0) {
      predictions <- predict(model, dplyr::select(test_set, -all_of(target)))
      confusion_matrix <- caret::confusionMatrix(predictions, test_set[[target]])
      evaluation_results[[alg]] <- confusion_matrix
      if (verbose) print(confusion_matrix)
    }

    # Obter o modelo treinado com maior acuracia
    best_index <- which.max(model$results$Accuracy)
    accuracy[alg] <- model$results$Accuracy[best_index]
    accuracy_sd[alg] <- model$results$AccuracySD[best_index]
    kappa[alg] <- model$results$Kappa[best_index]
    kappa_sd[alg] <- model$results$KappaSD[best_index]
    dratio[alg] <- (accuracy_sd[alg]/accuracy[alg]) - accuracy_sd[alg]
  }
  
  if (length(model_results) == 0) {
    stop("Nenhum modelo foi treinado com sucesso. Verifique os avisos para mais detalhes.")
  }

  res_scores <- data.frame(
    algorithm = names(accuracy),
    accuracy = unname(accuracy),
    accuracy_sd = unname(accuracy_sd),
    kappa = unname(kappa),
    kappa_sd = unname(kappa_sd),
    dratio = unname(dratio)
  ) 

  # Implementar a regra oneSE para seleção de modelo
  res_scores$accuracy_se <- res_scores$accuracy_sd / sqrt(number * repeats)
  
  best_acc_index <- which.max(res_scores$accuracy)
  best_accuracy <- res_scores$accuracy[best_acc_index]
  best_se <- res_scores$accuracy_se[best_acc_index]
  
  one_se_threshold <- best_accuracy - best_se
  
  res_scores$candidate <- res_scores$accuracy >= one_se_threshold
  
  res_scores_ordered <- res_scores[order(res_scores$accuracy, decreasing = TRUE), ]
  
  # Dos candidatos, escolher aquele com a melhor estabilidade (menor sd) 
  candidates <- res_scores[res_scores$candidate, ]
  final_choice <- candidates[which.min(candidates$accuracy_sd), ]
  best_model_name <- as.character(final_choice$algorithm)

  # --- Treinar o melhor modelo com 100% dos dados ---
  if (verbose) cat("\nTraining the best model (", best_model_name, ") on the full dataset...\n")
  
  final_train_set_full <- dplyr::select(data, -all_of(target))
  final_model_full <- caret::train(x = final_train_set_full,
                                   y = data[[target]],
                                   method = best_model_name,
                                   trControl = trainControl(method = "none"), # No resampling
                                   metric = "Accuracy")
  
  # Retornar os resultados dos modelos e as avaliações
  result_list <- list(
    model_names = list_alg, 
    models = model_results, # Models from train/test split
    final_model = final_model_full, # Final model trained on full dataset
    evaluations = evaluation_results,
    metrics = res_scores_ordered,
    best_model = best_model_name,
    one_se_threshold = one_se_threshold
    )
  
  class(result_list) <- "diagnoseR_result"
  
  return(result_list)
}