#' Preprocess a Dataset for Analysis
#'
#' This function provides a generic workflow for reading and preparing a dataset.
#' It handles file reading, filtering, column selection, and imputation of missing values.
#'
#' @param file_path Path to the data file (supports `.xlsx` or `.csv`).
#' @param text_cols A character vector of column names that should be treated as text/factors.
#' @param na_strings A character vector of strings to interpret as `NA` (missing) values.
#' @param sheet For Excel files, the name or index of the sheet to read.
#' @param filter_col Optional. The name of a column to filter by.
#' @param filter_values Optional. A vector of values to remove from the `filter_col`.
#' @param keep_cols Optional. A character vector of column names to keep. If `NULL`, all columns are kept.
#' @param seed A random seed for reproducibility of the missing data imputation.
#' @return A preprocessed data frame.
#' @importFrom tools file_ext
#' @importFrom readxl read_excel
#' @importFrom vroom vroom
#' @importFrom mice mice complete
#' @importFrom dplyr mutate_at vars all_of select filter
#' @importFrom magrittr %>%
#' @examples
#' # Create a temporary csv file for the example
#' temp_iris_path <- tempfile(fileext = ".csv")
#' write.csv(iris, temp_iris_path, row.names = FALSE)
#'
#' # Run the preprocessing function
#' processed_data <- preprocess_data(
#'   file_path = temp_iris_path,
#'   text_cols = "Species"
#' )
#'
#' # Clean up the temporary file
#' file.remove(temp_iris_path)
#'
#' @export
preprocess_data <- function(file_path,
                            text_cols,
                            na_strings = c("NA", "N/A", ""),
                            sheet = 1,
                            filter_col = NULL,
                            filter_values = NULL,
                            keep_cols = NULL,
                            seed = 42) {

  # Read data based on file extension
  ext <- tools::file_ext(file_path)
  if (ext == "xlsx") {
    df <- readxl::read_excel(file_path, sheet = sheet, na = na_strings)
  } else if (ext == "csv") {
    df <- vroom::vroom(file_path, delim = ",", na = na_strings, show_col_types = FALSE)
  } else {
    stop("Unsupported file type. Please use .xlsx or .csv.")
  }

  # Filter rows
  if (!is.null(filter_col) && !is.null(filter_values)) {
    if (!filter_col %in% names(df)) {
      stop(paste("filter_col '", filter_col, "' not found in the data.", sep = ""))
    }
    df <- df %>% dplyr::filter(!.data[[filter_col]] %in% filter_values)
  }

  # Select columns to keep
  if (!is.null(keep_cols)) {
    if (!all(keep_cols %in% names(df))) {
      stop("One or more columns in 'keep_cols' not found in the data.")
    }
    df <- df %>% dplyr::select(dplyr::all_of(keep_cols))
  }

  # Convert specified text columns to factors
  df <- df %>% dplyr::mutate_at(dplyr::vars(dplyr::all_of(text_cols)), as.factor)

  # Impute missing data using mice
  if (any(sapply(df, function(x) sum(is.na(x))) > 0)) {
    set.seed(seed)
    mice_imputed <- mice::mice(df, m = 1, method = 'pmm', printFlag = FALSE)
    df <- mice::complete(mice_imputed, 1)
  }

  return(df)
}

#' Compare Algoritmos de Amprendizado de Máquina.
#'
#' Esta função compara multiplos algoritmos de aprendizado de máquina.
#' @param data The dataframe containing your data.
#' @param target A string with the name of the column you want to predict.
#' @param list_alg A character vector of the machine learning algorithms you want to compare.
#' @param train_val The proportion of the data to be used for training.
#' @param seed A random seed for reproducibility.
#' @param number The number of folds for repeated cross-validation.
#' @param repeats The number of times to repeat the cross-validation.
#' @param verbose If `TRUE`, it prints the confusion matrix for each algorithm during execution.
#' @return A list object of class `diagnoseR_result` containing models, evaluations, and performance metrics.
#' @importFrom caret createDataPartition trainControl confusionMatrix
#' @importFrom dplyr select all_of arrange desc
#' @export
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
  
  final_train_set <- dplyr::select(train_set, -all_of(target))  # preditores
  dependent_variable <- train_set[[target]]                    # resposta de treino
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
    if (verbose) {
      cat("Training algorithm:", alg, "\n")
    }
    
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

  res_scores_ordered <- res_scores[order(res_scores$accuracy, decreasing = TRUE), ]
  
  # Retornar os resultados dos modelos e as avaliações
  result_list <- list(
    model_names = list_alg, 
    models = model_results, 
    evaluations = evaluation_results,
    metrics = res_scores_ordered,
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

#' Get Variable Importance from Trained Models
#'
#' This function extracts and summarizes variable importance from the models
#' trained by `comp_alg`.
#'
#' @param diagnoseR_result A list object of class `diagnoseR_result` from `comp_alg`.
#' @return A list where each element is a data frame of variable importances for an algorithm,
#' sorted from most to least important.
#' @importFrom caret varImp
#' @export
get_var_importance <- function(diagnoseR_result) {
  if (!inherits(diagnoseR_result, "diagnoseR_result")) {
    stop("Input must be an object of class 'diagnoseR_result' from the comp_alg function.")
  }

  var_importance_list <- list()

  for (alg in names(diagnoseR_result$models)) {
    model <- diagnoseR_result$models[[alg]]

    # Use tryCatch in case varImp is not available for a model
    imp <- tryCatch({
      caret::varImp(model, scale = TRUE)
    }, error = function(e) {
      warning(paste("Could not calculate variable importance for", alg, ":", e$message))
      return(NULL)
    })

    if (!is.null(imp)) {
      # Coerce to a standard data.frame to ensure subsetting with `drop=FALSE` works
      imp_df <- as.data.frame(imp$importance)
      # Add variable names as a column
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
          # If no numeric columns found, we can't sort. Warn and skip.
          warning(paste("Could not determine numeric importance columns for model:", alg))
          next # Skip to the next iteration of the loop
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
#' @export
plot_var_importance <- function(importance_list, model_name, top_n = 15) {
  if (!model_name %in% names(importance_list)) {
    stop(paste("Model '", model_name, "' not found. Available models are: ",
         paste(names(importance_list), collapse = ", ")))
  }

  imp_df <- importance_list[[model_name]]
  imp_df_top <- head(imp_df, top_n)

  # The importance column is the first one
  importance_col <- names(imp_df_top)[1]

  ggplot(imp_df_top, aes(x = reorder(Variable, .data[[importance_col]]), y = .data[[importance_col]])) +
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
