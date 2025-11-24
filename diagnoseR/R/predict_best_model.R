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