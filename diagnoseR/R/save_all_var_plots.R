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