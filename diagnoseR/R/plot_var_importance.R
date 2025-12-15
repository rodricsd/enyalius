#' Plot Variable Importance
#'
#' Creates a bar plot of variable importance for a specified model. The variables
#' are ordered based on their importance in the best model (as determined by
#' `get_var_importance`), ensuring consistent ordering across all plots.
#'
#' @param importance_list The list of variable importances from `get_var_importance()`.
#' @param model_name The name of the model to plot (e.g., "rf", "LogitBoost").
#' @param top_n The number of top variables to display in the plot.
#' @return A `ggplot` object showing the variable importance plot.
#' @importFrom ggplot2 ggplot aes geom_bar coord_flip labs theme_minimal scale_x_discrete
#' @importFrom rlang .data
#' @importFrom utils head 
#' @examples
#' # First, run comp_alg to get a results object
#' iris_results <- comp_alg(data = iris, target = "Species", verbose = FALSE, seed = 123)
#'
#' # Get importance from all models
#' var_imp_list <- get_var_importance(iris_results)
#'
#' # Plot the variable importance for the 'rf' model
#' plot_var_importance(var_imp_list, model_name = "rf")
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

  # The data is already sorted. Convert 'Variable' to a factor to preserve this order in the plot.
  # The plot will be flipped, so we reverse the factor levels for correct top-to-bottom display.
  imp_df_top$Variable <- factor(imp_df_top$Variable, levels = rev(imp_df_top$Variable))

  ggplot(imp_df_top, aes(x = .data$Variable, y = .data$Overall)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = paste("Variable Importance for", model_name), subtitle = paste("Top", top_n, "variables (ordered by best model)"), x = "Variables", y = "Importance") +
    theme_minimal()
}