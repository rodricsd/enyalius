### --- Example Workflow using the diagnoseR Package with Enyalius Data ---
library(ggplot2)
library(datasets)
library(diagnoseR)

# --- 1. Preprocess the Enyalius Dataset ---

# Define the columns to keep for the analysis
features_to_keep <- c("NPRC", "GF", "EGFS", "DHS", "TL4", "FL4", "nPVS", "nMS",
                      "nVrS", "nVnS", "nPM", "ncPM", "nCRo", "nbNSpL", "ncIp",
                      "nbCoA", "nSbO", "nSpC", "TL", "n4TL", "nArT", "final_species_name")

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
cat("\n\n--- Enyalius Dataset Analysis Results ---\n")
print(enyalius_results)

# --- 4. Get and Plot Variable Importance ---

# First, calculate the importance scores for all models
enyalius_var_imp <- get_var_importance(enyalius_results)

# Print the importance for the best model
cat("\n\n--- Variable Importance for Best Model (LogitBoost) ---\n")
print(enyalius_var_imp$LogitBoost)

# Now, create a plot for the best model
# You will need to have ggplot2 installed: install.packages("ggplot2")

save_all_var_plots(enyalius_var_imp, dataset_name = "enyalius", top_n = 15)

### --- Example Workflow using the diagnoseR Package with Iris Data ---

# --- 1. Setup: Load the iris dataset ---
data(iris)

# --- 2. Run the algorithm comparison ---
# Since the iris dataset is clean and doesn't require preprocessing like
# file reading or imputation, we can pass it directly to comp_alg.
iris_results <- comp_alg(data = iris,
                         target = "Species", # The column we want to predict
                         seed = 123,
                         verbose = FALSE) # Set to TRUE to see confusion matrices

# --- 3. Print the final summary table ---
cat("\n\n--- Iris Dataset Analysis Results ---\n")
print(iris_results)

iris_var_imp <- get_var_importance(iris_results)

# Print the importance for the best model
cat("\n\n--- Variable Importance for Best Model (svmLinear) ---\n")
print(iris_var_imp$svmLinear)

# Now, create a plot for the best model
# You will need to have ggplot2 installed: install.packages("ggplot2")

save_all_var_plots(iris_var_imp, dataset_name = "iris", top_n = 15)
