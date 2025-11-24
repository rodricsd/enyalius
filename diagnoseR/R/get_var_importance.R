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