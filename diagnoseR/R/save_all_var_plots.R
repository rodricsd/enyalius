#' Save Variable Importance Plots
#'
#' Saves variable importance plots. Can save individual plots for each model,
#' or a single combined plot (grid or grouped).
#'
#' @param importance_list The list of variable importances from `get_var_importance()`.
#' @param dataset_name Optional. A string to be included in the output filename (e.g., "iris").
#' @param output_dir The directory where the plot files will be saved. If it doesn't exist, it will be created.
#' @param type The type of plot to save: "individual" (default, one file per model), "grid", or "grouped".
#' @param color_palette Optional. A vector of colors to use for the grouped plot.
#' @param top_n The number of top variables to display in each plot.
#' @param width The width of the saved plot in inches.
#' @param height The height of the saved plot in inches.
#' @param dpi The resolution for the saved plot.
#' @param format The file format to save the plots (e.g., "png", "pdf", "svg"). Default is "png".
#' @return Invisibly returns a character vector of the saved file paths.
#' @importFrom ggplot2 ggsave
#' @export
save_all_var_plots <- function(importance_list,
                               dataset_name = NULL,
                               output_dir = "variable_importance_plots",
                               type = "individual",
                               color_palette = NULL,
                               top_n = 15,
                               width = 8,
                               height = 6,
                               dpi = 300,
                               format = "png") {
  if (!dir.exists(output_dir)) {
    message(paste("Creating directory:", output_dir))
    dir.create(output_dir, recursive = TRUE)
  }

  saved_files <- c()
  ext <- tolower(format)

  if (type == "individual") {
    for (model_name in names(importance_list)) {
      p <- plot_var_importance(importance_list, model_name = model_name, top_n = top_n, type = "single")

      if (!is.null(dataset_name) && nzchar(trimws(dataset_name))) {
        file_name <- paste0("var_imp_", dataset_name, "_", model_name, ".", ext)
      } else {
        file_name <- paste0("var_imp_", model_name, ".", ext)
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
  } else if (type %in% c("grid", "grouped")) {
    p <- plot_var_importance(importance_list, model_name = NULL, top_n = top_n, type = type, color_palette = color_palette)

    # Adjust dimensions for combined plots for better readability
    plot_width <- if (type == "grid") width * 1.5 else width
    plot_height <- if (type == "grid") height * 1.5 else height * 1.2

    if (!is.null(dataset_name) && nzchar(trimws(dataset_name))) {
      file_name <- paste0("var_imp_", dataset_name, "_", type, ".", ext)
    } else {
      file_name <- paste0("var_imp_", type, ".", ext)
    }
    file_path <- file.path(output_dir, file_name)

    ggplot2::ggsave(filename = file_path, plot = p, width = plot_width, height = plot_height, dpi = dpi)
    message(paste("Saved", type, "plot to", file_path))
    saved_files <- c(saved_files, file_path)
  } else {
    stop("Invalid 'type' specified. Must be one of 'individual', 'grid', or 'grouped'.")
  }
  invisible(saved_files)
}