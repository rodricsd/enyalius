#' Plot Variable Importance
#'
#' Creates a bar plot of variable importance for a specified model.
#'
#' @param importance_list The list of variable importances from `get_var_importance()`.
#' @param model_name The name of the model to plot (e.g., "rf", "LogitBoost").
#' @param top_n The number of top variables to display in the plot.
#' @return A `ggplot` object showing the variable importance plot.
#' @importFrom ggplot2 ggplot aes geom_bar coord_flip labs theme_minimal
#' @importFrom rlang .data
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
  # Silence R CMD check NOTE about no visible binding for global variable 'Variable'
  Variable <- NULL

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