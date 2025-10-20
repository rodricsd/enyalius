#' Preprocesses a dataset from a file
#'
#' This function provides a generic workflow for reading and preprocessing a dataset.
#' It can read from `.xlsx` or `.csv` files, programmatically set column types,
#' filter rows, select columns, and impute missing data using MICE.
#'
#' @param file_path Path to the data file (`.xlsx` or `.csv`).
#' @param text_cols A character vector of column names to be explicitly treated as text/factors.
#' @param na_strings A character vector of strings to interpret as `NA` values.
#' @param sheet For Excel files, the name or index of the sheet to read.
#' @param filter_col Optional. The name of a column to filter by.
#' @param filter_values Optional. A vector of values to remove from `filter_col`.
#' @param keep_cols Optional. A character vector of column names to keep. If `NULL`, all columns are kept.
#' @param seed A random seed for reproducibility of the imputation.
#' @return A preprocessed dataframe with missing values imputed.
#' @importFrom tools file_ext
#' @importFrom readxl read_xlsx
#' @importFrom readr read_csv
#' @importFrom mice mice complete
#' @importFrom dplyr `%>%` filter select
#' @importFrom rlang .data
#' @export
#' @examples
#' \dontrun{
#' # Example for a CSV file
#' # write.csv(iris, "iris.csv", row.names = FALSE)
#' # processed_iris <- preprocess_data(
#' #   file_path = "iris.csv",
#' #   text_cols = "Species"
#' # )
#'
#' # Example for the Enyalius Excel file
#' # processed_enyalius <- preprocess_data(
#' #   file_path = "herp-74-04-335_s02-edit.xlsx",
#' #   sheet = "Morphological data set",
#' #   text_cols = c("final_species_name", "sex"),
#' #   na_strings = "NA",
#' #   filter_col = "final_species_name",
#' #   filter_values = c("bad_sample", "unknown"),
#' #   keep_cols = c("NPRC", "GF", "EGFS", "DHS", "TL4", "FL4", "nPVS", "nMS",
#' #                 "nVrS", "nVnS", "nPM", "ncPM", "nCRo", "nbNSpL", "ncIp",
#' #                 "nbCoA", "nSbO", "nSpC", "TL", "n4TL", "nArT", "final_species_name")
#' # )
#' }
preprocess_data <- function(file_path, text_cols = NULL, na_strings = c("NA", "N/A", ""), sheet = 1,
                            filter_col = NULL, filter_values = NULL, keep_cols = NULL, seed = 42) {
  # --- 1. Read Data ---
  ext <- tools::file_ext(file_path)
  if (ext == "xlsx") {
    raw_data <- readxl::read_xlsx(file_path, sheet = sheet, na = na_strings, guess_max = 2000)
  } else if (ext == "csv") {
    raw_data <- readr::read_csv(file_path, na = na_strings, guess_max = 2000, show_col_types = FALSE)
  } else {
    stop("Unsupported file type. Please use '.xlsx' or '.csv'.")
  }

  # --- 2. Filter rows and select columns (optional) ---
  data_edit <- raw_data
  if (!is.null(filter_col) && !is.null(filter_values)) {
    data_edit <- data_edit %>%
      dplyr::filter(!.data[[filter_col]] %in% filter_values)
  }
  if (!is.null(keep_cols)) {
    data_edit <- data_edit %>%
      dplyr::select(dplyr::all_of(keep_cols))
  }

  # --- 3. Convert specified text columns to factors ---
  if (!is.null(text_cols)) {
    # Ensure we only try to convert columns that actually exist in the (potentially subsetted) data
    text_cols_to_convert <- intersect(text_cols, names(data_edit))
    for (col in text_cols_to_convert) {
        data_edit[[col]] <- as.factor(data_edit[[col]])
    }
  }
  data_edit <- droplevels(data_edit)

  # --- 4. Impute Missing Data using MICE ---
  imp <- mice::mice(data_edit, m = 5, maxit = 5, method = 'pmm', print = FALSE, seed = seed)
  data_imputed <- mice::complete(imp, 1)

  return(data_imputed)
}