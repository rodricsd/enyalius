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
#' @importFrom dplyr select all_of arrange desc
#' @export
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
  dependent_variable <- train_set_full[[target]]

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

#' @export
print.diagnoseR_result <- function(x, ...) {
  cat("\n--- Resultados da Regra do Erro Padrão ---\n")
  cat("Threshold (Melhor Acuracia - Melhor SE):", format(x$one_se_threshold, digits = 4), "\n")
  cat("Melhor modelo (mais estável dos candidatos):", x$best_model, "\n\n")
  cat("Metricas de Performance (CV no conjunto de treino):\n")
  
  # Reordenar colunas para clareza e formatar digitos
  metrics_to_print <- x$metrics[, c("algorithm", "candidate", "accuracy", "accuracy_sd", "kappa", "dratio", "accuracy_se")]
  
  print(metrics_to_print, row.names = FALSE)
  
  cat("\n--- Métricas do Treino com Dados Completos (Cross-Validation) ---\n")
  cat("Performance estimada no re-treino com 100% dos dados:\n")
  
  metrics_full_to_print <- x$metrics_full[, c("algorithm", "accuracy", "accuracy_sd", "kappa", "kappa_sd")]
  print(metrics_full_to_print, row.names = FALSE)
  
  cat("-------------------------------------\n")
}

#' Predict Outcomes Using the Best Model
#'
#' This function takes a `diagnoseR_result` object and new data, and returns
#' predictions using the best model identified by `comp_alg`. The best model
#' is chosen based on the one-standard-error rule and has been re-trained on the
#' full dataset for final use.
#' @param diagnoseR_result A list object of class `diagnoseR_result` from `comp_alg`.
#' @param newdata A data frame containing new observations for prediction. The columns
#'   must match the predictor variables used to train the original models.
#' @return A data frame containing the new data with an added column (`predicted_class`) for the predictions.
#' @export
#' @examples
#' # First, run comp_alg on the iris dataset
#' iris_results <- comp_alg(data = iris, target = "Species", verbose = FALSE, seed = 123)
#'
#' # Create some new data to predict on (must have same predictor columns)
#' new_iris_data <- data.frame(
#'   Sepal.Length = c(5.1, 7.0),
#'   Sepal.Width = c(3.5, 3.2),
#'   Petal.Length = c(1.4, 4.7),
#'   Petal.Width = c(0.2, 1.4)
#' )
#'
#' # Get predictions
#' predictions <- predict_best_model(iris_results, new_iris_data)
#' print(predictions)
#'
predict_best_model <- function(diagnoseR_result, newdata) {
  if (!inherits(diagnoseR_result, "diagnoseR_result")) {
    stop("Input must be an object of class 'diagnoseR_result' from the comp_alg function.")
  }

  best_model_name <- diagnoseR_result$best_model
  final_model <- diagnoseR_result$models_full[[best_model_name]]

  # --- Validation Step ---
  # Get the predictor variable names the model was trained on
  required_cols <- final_model$coefnames

  # Check if all required columns are present in the new data
  missing_cols <- setdiff(required_cols, names(newdata))

  if (length(missing_cols) > 0) {
    stop(paste0("The following required columns are missing from 'newdata': ",
                paste(missing_cols, collapse = ", ")))
  }

  predictions <- predict(final_model, newdata = newdata)

  results_df <- cbind(newdata, predicted_class = predictions)
  return(results_df)
}

#' Get Variable Importance from Trained Models
#'
#' This function extracts and summarizes variable importance from the models
#' trained by `comp_alg`.
#'
#' @param diagnoseR_result A list object of class `diagnoseR_result` from `comp_alg`.
#' @param model_source A string indicating which models to use. Can be `"split"` (default)
#'   to get importance from all models trained on the split data, or `"full"` to get
#'   importance from the single best model trained on the full dataset.
#' @return A list where each element is a data frame of variable importances for an algorithm,
#' sorted from most to least important.
#' @importFrom caret varImp
#' @export
#' @examples
#' # First, run comp_alg to get a results object
#' iris_results <- comp_alg(data = iris, target = "Species", verbose = FALSE, seed = 123)
#'
#' # Get importance from all models trained on the split data (default)
#' var_imp_split <- get_var_importance(iris_results, model_source = "split")
#' # Print importance for the 'rf' model
#' print(var_imp_split$rf)
#'
#' # Get importance from the single best model trained on the full data
#' var_imp_full <- get_var_importance(iris_results, model_source = "full")
#' # The list will only contain one element, for the best model
#' print(var_imp_full)
#'
get_var_importance <- function(diagnoseR_result, model_source = "split") {
  if (!inherits(diagnoseR_result, "diagnoseR_result")) {
    stop("Input must be an object of class 'diagnoseR_result' from the comp_alg function.")
  }
  
  if (!model_source %in% c("split", "full")) {
    stop("`model_source` must be either 'split' or 'full'.")
  }

  var_importance_list <- list()

  if (model_source == "split") {
    model_list <- diagnoseR_result$models
  } else { # model_source == "full"
    model_list <- diagnoseR_result$models_full
  }

  for (alg in names(model_list)) {
    model <- model_list[[alg]]

    # Use tryCatch in case varImp is not available for a model
    imp <- tryCatch({
      caret::varImp(model, scale = TRUE)
    }, error = function(e) {
      warning(paste("Could not calculate variable importance for", alg, ":", e$message))
      return(NULL)
    })

    if (!is.null(imp)) {
      imp_df <- as.data.frame(imp$importance)
      imp_df$Variable <- rownames(imp_df)
      rownames(imp_df) <- NULL

      # Check if an 'Overall' column exists for sorting.
      # If not (e.g., in multi-class cases), calculate row-wise max importance.
      if ("Overall" %in% names(imp_df)) {
        sort_col <- "Overall"
      } else {
        # Create a temporary 'Overall' column with the max importance across classes
        numeric_cols <- sapply(imp_df, is.numeric)
        if (sum(numeric_cols) > 0) {
          imp_df$Overall <- apply(imp_df[, numeric_cols, drop = FALSE], 1, max)
          sort_col <- "Overall"
        } else {
          warning(paste("Could not determine numeric importance columns for model:", alg))
          next 
        }
      }
      imp_df <- imp_df[order(imp_df[[sort_col]], decreasing = TRUE), ]
      var_importance_list[[alg]] <- imp_df
    }
  }

  return(var_importance_list)
}

#' Plot Variable Importance
#'
#' Creates a bar plot of variable importance for a specified model.
#'
#' @param importance_list The list of variable importances from `get_var_importance()`.
#' @param model_name The name of the model to plot (e.g., "rf", "LogitBoost").
#' @param top_n The number of top variables to display in the plot.
#' @return A `ggplot` object showing the variable importance plot.
#' @importFrom ggplot2 ggplot aes geom_bar coord_flip labs theme_minimal theme element_text
#' @importFrom utils head 
#' @examples
#' # First, run comp_alg to get a results object
#' iris_results <- comp_alg(data = iris, target = "Species", verbose = FALSE, seed = 123)
#'
#' # Get importance from the single best model trained on the full data
#' var_imp_full <- get_var_importance(iris_results, model_source = "full")
#'
#' # The name of the best model is stored in the results object
#' best_model_name <- iris_results$best_model
#'
#' # Plot the variable importance for the best model
#' if (length(var_imp_full) > 0) {
#'   plot_var_importance(var_imp_full, model_name = best_model_name)
#' }
#'
#' @export
plot_var_importance <- function(importance_list, model_name, top_n = 15) {
  if (!model_name %in% names(importance_list)) {
    stop(paste("Model '", model_name, "' not found. Available models are: ",
         paste(names(importance_list), collapse = ", ")))
  }

  imp_df <- importance_list[[model_name]]
  imp_df_top <- head(imp_df, top_n)

  # Explicitly use the 'Overall' column for plotting, which was created for sorting.
  if (!"Overall" %in% names(imp_df_top)) {
    stop("Could not find the 'Overall' importance column to plot.")
  }

  ggplot(imp_df_top, aes(x = reorder(Variable, .data[["Overall"]]), y = .data[["Overall"]])) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = paste("Top", top_n, "Variable Importance for", model_name), x = "Variables", y = "Importance") +
    theme_minimal()
}

#' Save All Variable Importance Plots
#'
#' Generates and saves variable importance plots for all models to a specified directory.
#'
#' @param importance_list The list of variable importances from `get_var_importance()`.
#' @param dataset_name Optional. A string to be included in the output filename (e.g., "iris").
#' @param output_dir The directory where the plot files will be saved. If it doesn't exist, it will be created.
#' @param top_n The number of top variables to display in each plot.
#' @param width The width of the saved plot in inches.
#' @param height The height of the saved plot in inches.
#' @param dpi The resolution for the saved plot.
#' @return Invisibly returns a character vector of the saved file paths.
#' @importFrom ggplot2 ggsave
#' @export
save_all_var_plots <- function(importance_list,
                               dataset_name = NULL,
                               output_dir = "variable_importance_plots",
                               top_n = 15,
                               width = 8,
                               height = 6,
                               dpi = 300) {
  if (!dir.exists(output_dir)) {
    message(paste("Creating directory:", output_dir))
    dir.create(output_dir, recursive = TRUE)
  }

  saved_files <- c()

  for (model_name in names(importance_list)) {
    p <- plot_var_importance(importance_list, model_name = model_name, top_n = top_n)

    if (!is.null(dataset_name) && nzchar(trimws(dataset_name))) {
      file_name <- paste0("var_imp_", dataset_name, "_", model_name, ".png")
    } else {
      file_name <- paste0("var_imp_", model_name, ".png")
    }
    file_path <- file.path(output_dir, file_name)

    ggplot2::ggsave(
      filename = file_path,
      plot = p,
      width = width,
      height = height,
      dpi = dpi
    )
    message(paste("Saved plot for", model_name, "to", file_path))
    saved_files <- c(saved_files, file_path)
  }
  invisible(saved_files)
}
