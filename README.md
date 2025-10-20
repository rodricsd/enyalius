# Tutorial: Using the diagnoseR Package

The `diagnoseR` package provides a streamlined workflow to preprocess data, compare the performance of multiple machine learning algorithms, and select the best model based on robust statistical methods.

## Installation

This package can be installed directly from its GitHub repository using the `devtools` package. If you don't have `devtools` installed, open your R console and run:

```R
install.packages("devtools")
```

Once `devtools` is installed, you can install `diagnoseR` from a GitHub repository. **Note:** The following command assumes the package is located at `https://github.com/rodricsd/enyalius/diagnoseR`. You will need to replace `user/diagnoseR` with the actual repository path.

```R
devtools::install_github("rodricsd/enyalius/diagnoseR")
```

After installation, load the package into your R session:

```R
library(diagnoseR)
```

## Usage

The primary function in this package is `comp_alg`. Let's walk through how to use it.

### Example

For this example, we'll use the built-in `iris` dataset.

```R
# Load the iris dataset
data(iris)

# Run the comparison function
# We want to predict the 'Species' column
results <- comp_alg(data = iris, target = "Species")

# Print the results
print(results)
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
*   `cv_folds`: The number of folds for cross-validation (this parameter is not used in the current version of the function, which uses `number` and `repeats` instead).
*   `seed`: A random seed for reproducibility (default is 123).
*   `number`: The number of folds for repeated cross-validation (default is 5).
*   `repeats`: The number of times to repeat the cross-validation (default is 10).
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

The returned list object also contains the trained models and detailed evaluation results, which you can access for further analysis:

```R
# Access the trained models
results$models

# Access the detailed evaluation results (including confusion matrices)
results$evaluations

# Access the performance metrics as a data frame
results$metrics

# Get the name of the best model
results$best_model
```
