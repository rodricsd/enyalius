# diagnoseR: A Robust Framework for ML Model Comparison in R

# Tutorial: Using the diagnoseR Package

[!R-CMD-check](https://github.com/rodricsd/enyalius/actions/workflows/r.yml)
[!Codecov test coverage](https://app.codecov.io/gh/rodricsd/enyalius?branch=main)

The `diagnoseR` package offers a streamlined and robust framework for comparing machine learning models in R. It automates the process of training, evaluating, and selecting the best classification model from a list of candidates, using repeated cross-validation and the statistically sound "one-standard-error" rule for model selection.


## Key Features

*   **Automated Comparison**: Compare multiple `caret`-compatible algorithms with a single function call.
*   **Robust Model Selection**: Uses the "one-standard-error" rule to select a model that is not only accurate but also stable, avoiding overfitting.
*   **Flexible Evaluation**: Easily switch between a train/test split and a full-dataset cross-validation by changing a single parameter (`train_val`).
*   **Resilient Workflow**: A single failing algorithm won't stop the entire analysis. The function will issue a warning and continue with the remaining models.
*   **Consistent Reporting**: Variable importance plots are standardized and ordered based on the best model, making direct visual comparisons easy and intuitive.
*   **Production-Ready**: Automatically retrains the best model on 100% of the data, providing a final, optimized model ready for prediction.

## Installation
 
 You can install the `diagnoseR` package from GitHub using the `devtools` package.
 
 ```R
 # First, install devtools if you don't have it
 if (!require("devtools")) {
   install.packages("devtools")
 }
 
 # Install the diagnoseR package from the 'diagnoseR' subdirectory of the GitHub repo
 devtools::install_github("rodricsd/enyalius/diagnoseR")
 
 # Load the package
 library(diagnoseR)
 ```
 
 ### Local Installation (for Developers)
 
 If you have cloned the repository to your local machine, you can install the package directly from the source files. Make sure your R session's working directory is inside the package folder (`enyalius/diagnoseR`).
 
 ```R
 # Build and install the package locally
 devtools::install()
 
 # Load the package
 library(diagnoseR)
 ```
 
 ## Documentation
 
 The package is documented using `roxygen2`. If you are contributing to the package and make changes to the function documentation (the `#'` comments), you must regenerate the documentation files.
 
 ```R
 # From within the package directory ('enyalius/diagnoseR')
 devtools::document()
 ```
 
 ## A Complete Workflow Example

The primary function in this package is `comp_alg`. Let's walk through how to use it.

### Example

For this example, we'll use the built-in `iris` dataset.

```R
# Load the iris dataset
data(iris)

# Run the comparison function
# We want to predict the 'Species' column
results <- comp_alg(data = iris, target = "Species")

# Print the summary results
print(results)

# Get the standardized variable importance for all models
var_imp <- get_var_importance(results)

# Print the importance for the 'rf' model
# Note the variables are ordered based on the best model
print(var_imp$rf)
```
## A Complete Workflow Example

Real-world data is rarely clean. It often contains missing values, irrelevant columns, or incorrect data types. The `diagnoseR` package is designed to handle this entire workflow, from raw file to model comparison.

This example shows how to use `preprocess_data` to clean a dataset and then pass the result to `comp_alg` for analysis. We will use the Enyalius lizard dataset, which is stored in an Excel file.

```R
# Load the package
library(diagnoseR)

# --- 1. Preprocess the Enyalius Dataset ---

# Define which columns we want to keep for the analysis
features_to_keep <- c("NPRC", "GF", "EGFS", "DHS", "TL4", "FL4", "nPVS", "nMS",
                      "nVrS", "nVnS", "nPM", "ncPM", "nCRo", "nbNSpL", "ncIp",
                      "nbCoA", "nSbO", "nSpC", "TL", "n4TL", "nArT", "final_species_name")

# Use the preprocess_data function to read, clean, and impute the data
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

# --- 2. Compare Machine Learning Algorithms ---

enyalius_results <- comp_alg(
  data = enyalius_processed,
  target = "final_species_name", # The column we want to predict
  seed = 123
)

# --- 3. Print the final summary table ---
print(enyalius_results)

# --- 4. Get and print variable importance ---
enyalius_var_imp <- get_var_importance(enyalius_results)

# Print the importance for the best model
best_model_name <- enyalius_results$best_model
print(enyalius_var_imp[[best_model_name]])

# --- 5. Plot and save variable importance plots ---

# Plot the importance for the best model
plot_var_importance(enyalius_var_imp, model_name = best_model_name)

# Save all plots to a single PDF for easy comparison
save_all_var_plots(enyalius_var_imp, dataset_name = "enyalius_comparison")
```

## Function Reference

### `preprocess_data()`

This function provides a generic workflow for reading and preparing a dataset for analysis. It handles file reading, filtering, column selection, and imputation of missing values.

**Usage:**

```R
processed_data <- preprocess_data(file_path, text_cols, ...)
```

**Parameters:**

*   `file_path`: Path to the data file (supports `.xlsx` or `.csv`).
*   `text_cols`: A character vector of column names that should be treated as text/factors.
*   `na_strings`: A character vector of strings to interpret as `NA` (missing) values (default is `c("NA", "N/A", "")`).
*   `sheet`: For Excel files, the name or index of the sheet to read (default is `1`).
*   `filter_col`: Optional. The name of a column to filter by.
*   `filter_values`: Optional. A vector of values to remove from the `filter_col`.
*   `keep_cols`: Optional. A character vector of column names to keep. If `NULL`, all columns are kept.
*   `seed`: A random seed for reproducibility of the missing data imputation.

### Understanding the `comp_alg` function

The `comp_alg` function has several parameters you can customize:

*   `data`: The dataframe containing your data.
*   `target`: A string with the name of the column you want to predict.
*   `list_alg`: A character vector of the machine learning algorithms you want to compare. The default list is `c("rpart", "nnet", "svmLinear", "rf", "LogitBoost", "knn")`. You can find a list of available algorithms in the `caret` package documentation.
*   `train_val`: The proportion of the data to be used for training (default is 0.75).
    *   Set to `1.0` to skip the train/test split and use the entire dataset for model comparison via cross-validation.
    *   **Note:** To prevent errors during cross-validation, the function will automatically lower `train_val` if it's too high for the given dataset size and number of CV folds, issuing a warning. The final model is always trained on 100% of the data regardless.
*   `seed`: A random seed for reproducibility (default is 123).
*   `number`: The number of folds for repeated cross-validation (default is 5).
*   `repeats`: The number of times to repeat the cross-validation (default is 10).
*   `cv_folds`: (Obsoleto) Este parâmetro não é mais utilizado. Use `number` para o número de folds e `repeats` para o número de repetições.
*   `verbose`: If `TRUE` (the default), it prints the confusion matrix for each algorithm during execution.

### Interpreting the Output

The `comp_alg` function returns a list object of class `diagnoseR_result`. When you print this object, you'll see a summary of the results, including:

*   **One Standard Error Rule Results:**
    *   `Threshold`: The accuracy threshold calculated by the "one standard error" rule. This is the accuracy of the best model minus one standard error.
    *   `Best Model`: The model that is the most stable (lowest standard deviation of accuracy) among the candidates that have an accuracy above the threshold. This is often a more robust choice than simply picking the model with the highest accuracy.

*   **Performance Metrics:** A table with the following columns for each algorithm:
    *   `algorithm`: The name of the algorithm.
    *   `candidate`: `TRUE` if the model's accuracy is above the one standard error threshold.
    *   `accuracy`: The average accuracy from cross-validation.
    *   `accuracy_sd`: The standard deviation of the accuracy.
    *   `kappa`: The average Kappa statistic from cross-validation.
    *   `dratio`: A custom metric (`(accuracy_sd/accuracy) - accuracy_sd`).
    *   `accuracy_se`: The standard error of the accuracy.

The returned list object also contains the trained models and other useful information, which you can access for further analysis:

```R
# Access the list of models trained on the split data (used for comparison)
results$models

# Access the detailed evaluation results (confusion matrices from the test set)
results$evaluations

# Access the performance metrics as a data frame
results$metrics

# Get the name of the best model
results$best_model

# Access the final, production-ready model (the best model retrained on 100% of the data)
results$final_model
```
### `get_var_importance()`

Esta função extrai e padroniza a importância das variáveis de todos os modelos treinados pelo `comp_alg`. Uma característica principal é que ela ordena todas as variáveis de acordo com sua importância no **melhor modelo**, garantindo que todos os gráficos e tabelas tenham um layout consistente e comparável.

**Uso:**

```R
var_imp_list <- get_var_importance(results)

# Imprime a importância para o modelo 'rf' (as variáveis são ordenadas pelo ranking do melhor modelo)
print(var_imp_list$rf)
```
```
