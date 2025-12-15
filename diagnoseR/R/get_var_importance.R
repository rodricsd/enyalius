#' Get Variable Importance from Trained Models
#'
#' This function extracts variable importance from the models trained by `comp_alg`.
#' It standardizes the output by ordering all variables according to their
#' importance in the best model (identified by `comp_alg`). This ensures that
#' all variables are present in the output for every model, facilitating direct comparisons.
#'
#' @param diagnoseR_result A list object of class `diagnoseR_result` from `comp_alg()`.
#' @return A list where each element is a data frame of variable importances for an algorithm,
#' sorted from most to least important based on the best model's ranking.
#' @importFrom caret varImp
#' @importFrom dplyr left_join
#' @export
#' @examples
#' # First, run comp_alg to get a results object
#' iris_results <- comp_alg(data = iris, target = "Species", verbose = FALSE, seed = 123)
#'
#' # Get importance from all models
#' var_imp_list <- get_var_importance(iris_results)
#' # Print importance for the 'rf' model
#' # Note the variables are ordered based on the best model's (svmLinear) importance
#' print(var_imp_list$rf)
#'
get_var_importance <- function(diagnoseR_result) {
  if (!inherits(diagnoseR_result, "diagnoseR_result")) {
    stop("Input must be an object of class 'diagnoseR_result' from the comp_alg function.")
  }

  # Helper function to process varImp results
  process_imp <- function(model, alg_name) {
    imp_obj <- tryCatch({
      caret::varImp(model, scale = TRUE)
    }, error = function(e) {
      warning(paste("Could not calculate variable importance for", alg_name, ":", e$message))
      return(NULL)
    })

    if (is.null(imp_obj)) return(NULL)

    imp_df <- as.data.frame(imp_obj$importance)
    
    # Standardize to a single 'Overall' importance column
    # This handles models that return importance per class (like nnet, LogitBoost)
    if (!"Overall" %in% names(imp_df)) {
      numeric_cols <- sapply(imp_df, is.numeric)
      if (sum(numeric_cols) > 0) {
        imp_df$Overall <- apply(imp_df[, numeric_cols, drop = FALSE], 1, max)
      } else {
        warning(paste("Could not determine numeric importance columns for model:", alg_name))
        return(NULL)
      }
    }
    
    imp_df$Variable <- rownames(imp_df)
    # Return only the essential columns
    return(imp_df[, c("Variable", "Overall")])
  }

  # 1. Get the master order from the best model (trained on split data)
  best_model_name <- diagnoseR_result$best_model
  best_model <- diagnoseR_result$models[[best_model_name]]
  
  master_imp <- process_imp(best_model, best_model_name)
  if (is.null(master_imp)) {
    stop("Could not calculate variable importance for the best model. Cannot proceed.")
  }
  
  master_imp <- master_imp[order(master_imp$Overall, decreasing = TRUE), ]
  master_variable_order <- master_imp$Variable
  
  # Create a template dataframe with all variables in the master order
  all_vars_template <- data.frame(Variable = master_variable_order)

  # 2. Process all models and conform them to the master order
  var_importance_list <- list()
  model_list <- diagnoseR_result$models

  for (alg in names(model_list)) {
    model <- model_list[[alg]]
    
    current_imp <- process_imp(model, alg)
    
    if (!is.null(current_imp)) {
      # Use left_join to merge, which preserves the order of the first dataframe (all_vars_template)
      # This ensures the final dataframe is always in the master order.
      final_imp <- dplyr::left_join(all_vars_template, current_imp, by = "Variable")
      
      # Replace NA values in 'Overall' with 0 for variables not used by the model
      final_imp$Overall[is.na(final_imp$Overall)] <- 0
      var_importance_list[[alg]] <- final_imp
    }
  }

  return(var_importance_list)
}
