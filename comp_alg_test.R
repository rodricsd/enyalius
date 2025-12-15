### --- Example Workflow using the diagnoseR Package with Enyalius Data ---
library(ggplot2)
library(datasets)
library(diagnoseR)

# NOTE: This script assumes a function `preprocess_data` exists.
# If it is not part of the package, it should be sourced or defined before running.
# For reproducibility, we will mock this function if it's not available.
if (!exists("preprocess_data")) {
  # Mock function for testing purposes if the original is not available
  preprocess_data <- function(file_path, sheet, text_cols, na_strings, filter_col, filter_values, keep_cols, seed) {
    # This mock function will simply fail gracefully to indicate the dependency.
    # In a real scenario, you might load a pre-saved .RData file.
    stop("`preprocess_data` function not found. Please define it or load the 'enyalius_processed' data manually.")
  }
  # As we cannot run the above, we will skip the Enyalius part in a CI environment
  # and only run the Iris and train_val=1.0 tests.
  run_enyalius_test <- FALSE
} else {
  run_enyalius_test <- TRUE
}

# --- 1. Preprocess the Enyalius Dataset ---

# Define the columns to keep for the analysis
features_to_keep <- c("NPRC", "GF", "EGFS", "DHS", "TL4", "FL4", "nPVS", "nMS",
                      "nVrS", "nVnS", "nPM", "ncPM", "nCRo", "nbNSpL", "ncIp",
                      "nbCoA", "nSbO", "nSpC", "TL", "n4TL", "nArT", "final_species_name")

if (run_enyalius_test) {
  # Use the generic preprocess_data function to read, clean, and impute the data
  enyalius_processed <- preprocess_data(
    file_path = "herp-74-04-335_s02-edit.xlsx",
    sheet = "Morphological data set",
    text_cols = c("final_species_name", "sex"),
    na_strings = "NA",
    filter_col = "final_species_name",
    filter_values = c("bad_sample", "unknown"),
    keep_cols = features_to_keep,
    seed = 42
  )
}

# --- 2. Run the algorithm comparison ---

enyalius_results <- comp_alg(
  data = enyalius_processed,
  target = "final_species_name", # The column we want to predict
  seed = 123,
  number = 5,
  repeats = 10,
  verbose = FALSE # Set to TRUE to see confusion matrices
)

# --- 3. Print the final summary table ---
if (run_enyalius_test) {
  cat("\n\n--- Enyalius Dataset Analysis Results ---\n")
  print(enyalius_results)

  # --- 4. Get and Plot Variable Importance ---

  # First, calculate the importance scores for all models
  enyalius_var_imp <- get_var_importance(enyalius_results)

  # Print the importance for the best model (dynamically identified)
  cat(paste("\n\n--- Variable Importance for Best Model (", enyalius_results$best_model, ") ---\n"))
  print(enyalius_var_imp[[enyalius_results$best_model]])

  # Print importance for another model to verify consistent variable ordering
  cat("\n\n--- Variable Importance for 'rf' model (variables ordered by best model) ---\n")
  print(enyalius_var_imp$rf)

  save_all_var_plots(enyalius_var_imp, dataset_name = "enyalius", top_n = 15)
}

### --- Example Workflow using the diagnoseR Package with Iris Data ---

# --- 1. Setup: Load the iris dataset ---
data(iris)

# --- 2. Run the algorithm comparison ---
iris_results <- comp_alg(data = iris,
                         target = "Species", # The column we want to predict
                         seed = 123,
                         verbose = FALSE) # Set to TRUE to see confusion matrices

# --- 3. Print the final summary table ---
cat("\n\n--- Iris Dataset Analysis Results ---\n")
print(iris_results)

iris_var_imp <- get_var_importance(iris_results)

# Print the importance for the best model (dynamically identified)
cat(paste("\n\n--- Variable Importance for Best Model (", iris_results$best_model, ") ---\n"))
print(iris_var_imp[[iris_results$best_model]])

save_all_var_plots(iris_var_imp, dataset_name = "iris", top_n = 15)


### --- Example Workflow testing train_val = 1.0 ---

cat("\n\n--- Iris Dataset Analysis with train_val = 1.0 (Full Dataset CV) ---\n")

# --- 1. Run algorithm comparison on the full dataset ---
iris_results_full <- comp_alg(data = iris,
                              target = "Species",
                              train_val = 1.0, # Use 100% of data for CV
                              seed = 123,
                              verbose = FALSE)

# --- 2. Print the results ---
# Note: The 'evaluations' component should be empty as there was no test set.
print(iris_results_full)

# --- 3. Get and save variable importance plots ---
# This demonstrates that the rest of the workflow functions correctly
iris_var_imp_full <- get_var_importance(iris_results_full)

save_all_var_plots(iris_var_imp_full, dataset_name = "iris_full_cv", top_n = 15)

cat("\n\n--- Test script finished successfully. ---\n")
