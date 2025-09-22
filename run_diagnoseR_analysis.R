# Load necessary libraries
library(readxl)
library(diagnoseR)

# Load the dataset
data <- readxl::read_excel("herp-74-04-335_s02-edit.xlsx")

# Specify the target variable
target_variable <- "final_species_name"

# Run the algorithm comparison
results <- comp_alg(data = data, target = target_variable)

# Print the results
print(results)

# Save the results to a text file
sink("analysis_results.txt")
print(results)
sink()

cat("Analysis complete. Results are printed above and saved to analysis_results.txt\n")