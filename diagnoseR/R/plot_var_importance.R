#' Plot Variable Importance
#'
#' Creates a bar plot of variable importance. Can plot a single model, or compare
#' multiple models using a grid or grouped bars. The variables are ordered based
#' on their importance in the best model (as determined by `get_var_importance`),
#' ensuring consistent ordering across all plots.
#'
#' @param importance_list The list of variable importances from `get_var_importance()`.
#' @param model_name The name of the model(s) to plot. Can be a single string, a character vector, or NULL (to plot all available models).
#' @param top_n The number of top variables to display in the plot.
#' @param type The type of plot to generate: "single" (default), "grid", or "grouped".
#' @param color_palette Optional. A vector of colors to use for the grouped plot.
#' @return A `ggplot` object showing the variable importance plot.
#' @importFrom ggplot2 ggplot aes geom_bar coord_flip labs theme_minimal scale_x_discrete facet_wrap scale_fill_manual scale_fill_viridis_d
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
#' # Plot all models in a grid (by passing NULL to model_name)
#' plot_var_importance(var_imp_list, type = "grid")
#'
#' # Plot all models with grouped bars
#' plot_var_importance(var_imp_list, type = "grouped")
#'
#' # Plot grouped bars with custom colors
#' plot_var_importance(var_imp_list, type = "grouped", color_palette = c("red", "blue", "green"))
#'
#' @export
plot_var_importance <- function(importance_list, model_name = NULL, top_n = 15, type = "single", color_palette = NULL) {
  # Silence R CMD check NOTE about no visible binding for global variable 'Variable'
  Variable <- NULL

  available_models <- names(importance_list)

  if (is.null(model_name)) {
    model_name <- available_models
    if (type == "single" && length(model_name) > 1) {
      # If user asks for a single plot of all models, default to a grid.
      type <- "grid"
    }
  }

  missing_models <- setdiff(model_name, available_models)
  if (length(missing_models) > 0) {
    stop(paste("Model(s) not found:", paste(missing_models, collapse = ", "),
               ". Available models are: ", paste(available_models, collapse = ", ")))
  }

  # Combine data for selected models
  plot_data <- do.call(rbind, lapply(model_name, function(m) {
    df <- head(importance_list[[m]], top_n)
    if (!"Overall" %in% names(df)) {
      stop(paste("Could not find the 'Overall' importance column for model", m))
    }
    df$Model <- m
    return(df)
  }))

  # The data is already sorted by get_var_importance.
  # We use the order of variables from the data to set factor levels.
  # Since we want the most important at the top (coord_flip), we reverse the order.
  plot_data$Variable <- factor(plot_data$Variable, levels = rev(unique(plot_data$Variable)))

  p <- ggplot(plot_data, aes(x = .data$Variable, y = .data$Overall))

  if (type == "grouped") {
    p <- p + geom_bar(stat = "identity", aes(fill = .data$Model), position = "dodge") +
      labs(title = "Variable Importance Comparison (Grouped)")

    if (!is.null(color_palette)) {
      p <- p + scale_fill_manual(values = color_palette)
    } else {
      p <- p + scale_fill_viridis_d()
    }
  } else if (type == "grid") {
    p <- p + geom_bar(stat = "identity", fill = "steelblue") +
      facet_wrap(~ .data$Model) +
      labs(title = "Variable Importance Comparison (Grid)")
  } else {
    # Single plot
    p <- p + geom_bar(stat = "identity", fill = "steelblue") +
      labs(title = paste("Variable Importance for", model_name[1]))
  }

  p + coord_flip() +
    labs(subtitle = paste("Top", top_n, "variables (ordered by best model)"),
         x = "Variables", y = "Importance") +
    theme_minimal()
}