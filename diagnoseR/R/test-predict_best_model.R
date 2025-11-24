library(testthat)
library(diagnoseR)

# --- Setup: Create a reusable result object for tests ---
# We run comp_alg once on the iris dataset to get a valid diagnoseR_result object.
iris_results_for_test <- comp_alg(
  data = iris,
  target = "Species",
  verbose = FALSE,
  seed = 123
)

# --- Test Case 1: Successful Prediction ---
test_that("predict_best_model returns correct predictions and format", {

  # Create new data with the correct columns
  new_iris_data <- data.frame(
    Sepal.Length = c(5.1, 7.0),
    Sepal.Width = c(3.5, 3.2),
    Petal.Length = c(1.4, 4.7),
    Petal.Width = c(0.2, 1.4)
  )

  predictions_df <- predict_best_model(iris_results_for_test, new_iris_data)

  # Check 1: The output is a data frame
  expect_s3_class(predictions_df, "data.frame")

  # Check 2: The output contains the new 'predicted_class' column
  expect_true("predicted_class" %in% names(predictions_df))

  # Check 3: The number of rows in the output matches the input
  expect_equal(nrow(predictions_df), nrow(new_iris_data))
})

# --- Test Case 2: Validation Check for Missing Columns ---
test_that("predict_best_model throws an error for missing columns", {

  # Create new data that is missing the 'Petal.Length' column
  new_iris_data_bad <- data.frame(
    Sepal.Length = c(5.1),
    Sepal.Width = c(3.5),
    Petal.Width = c(0.2)
  )

  # Check that the function stops and provides the expected error message
  expect_error(
    predict_best_model(iris_results_for_test, new_iris_data_bad),
    "The following required columns are missing from 'newdata': Petal.Length"
  )
})