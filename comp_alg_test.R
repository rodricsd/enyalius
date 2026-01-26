### --- Example Workflow using the diagnoseR Package with Enyalius Data ---
library(ggplot2)
library(praznik) # For MRMR feature selection
# caret is a dependency of diagnoseR, but we load it for findCorrelation
library(caret)
library(datasets)
library(diagnoseR)

# NOTE: This script assumes a function `preprocess_data` exists.
# If it is not part of the package, it should be sourced or defined before running.
# For reproducibility, we will mock this function if it's not available.
if (!exists("preprocess_data")) {
  # Mock function for testing purposes if the original is not available
  preprocess_data <- function(file_path, sheet, text_cols, na_strings, filter_col, filter_values, include_col = NULL, include_values = NULL, keep_cols, seed) {
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
  cat("\n\n--- Enyalius Dataset Analysis Results (ALL DATA) ---\n")
  print(enyalius_results)

  # --- 4. Get and Plot Variable Importance ---

  # First, calculate the importance scores for all models
  enyalius_var_imp <- get_var_importance(enyalius_results)

  # Print the importance for the best model (dynamically identified)
  cat(paste("\n\n--- Variable Importance for Best Model: ", enyalius_results$best_model, " ---\n"))
  print(enyalius_var_imp[[enyalius_results$best_model]])

  save_all_var_plots(enyalius_var_imp, dataset_name = "enyalius_all_individual", type = "individual", top_n = 15)

  # --- 5. Feature Selection Demonstrations ---
  cat("\n\n--- Feature Selection Demonstrations ---\n")
  
  target_col <- "final_species_name"
  all_predictors <- setdiff(names(enyalius_processed), target_col)
  
  # A. Forward Selection (Incremental)
  cat("\n--- A. Forward Selection (Incremental) ---\n")
  forward_results <- list()
  
  # We iterate adding one variable at a time based on the list order
  for (i in 1:length(all_predictors)) {
    current_vars <- all_predictors[1:i]
    data_subset <- enyalius_processed[, c(current_vars, target_col)]
    
    cat(paste0("Forward Step ", i, ": Training with ", length(current_vars), " variables...\n"))
   
    # Run comp_alg (suppress verbose to avoid clutter)
    res <- comp_alg(data = data_subset, 
                    target = target_col, 
                    seed = 123,
                    number = 5,
                    repeats = 10,
                    verbose = FALSE)
    
    best_mod <- res$best_model
    best_acc <- res$metrics[res$metrics$algorithm == best_mod, "accuracy"]
    best_sd <- res$metrics[res$metrics$algorithm == best_mod, "accuracy_sd"]
    
    forward_results[[i]] <- data.frame(
      n_vars = length(current_vars),
      best_model = best_mod,
      accuracy = best_acc,
      accuracy_sd = best_sd,
      added_var = current_vars[i]
    )
  }
  df_forward <- do.call(rbind, forward_results)
  cat("\n")
  print(df_forward)

  p_fwd <- ggplot(df_forward, aes(x = n_vars, y = accuracy)) +
    geom_line() + geom_point() +
    geom_errorbar(aes(ymin = accuracy - accuracy_sd, ymax = accuracy + accuracy_sd), width = 0.2) +
    labs(title = "Forward Selection Accuracy", x = "Number of Variables", y = "Accuracy") +
    theme_minimal()
  ggsave("forward_selection.png", p_fwd, width = 8, height = 6)
  
  # B. Backward Selection (Decremental - Naive)
  cat("\n--- B. Backward Selection (Decremental - Naive) ---\n")
  backward_results <- list()
  
  # Start with all, remove one by one from the end of the list
  for (i in length(all_predictors):1) {
    current_vars <- all_predictors[1:i]
    data_subset <- enyalius_processed[, c(current_vars, target_col)]
    
    cat(paste0("Backward Step ", length(all_predictors) - i + 1, ": Training with ", length(current_vars), " variables...\n"))
    
    res <- comp_alg(data = data_subset, 
                    target = target_col, 
                    seed = 123,
                    number = 5,
                    repeats = 10,
                    verbose = FALSE)
    
    best_mod <- res$best_model
    best_acc <- res$metrics[res$metrics$algorithm == best_mod, "accuracy"]
    best_sd <- res$metrics[res$metrics$algorithm == best_mod, "accuracy_sd"]
    
    backward_results[[length(backward_results) + 1]] <- data.frame(
      n_vars = length(current_vars),
      best_model = best_mod,
      accuracy = best_acc,
      accuracy_sd = best_sd,
      removed_var = current_vars[i]
    )
  }
  df_backward <- do.call(rbind, backward_results)
  cat("\n")
  print(df_backward)

  p_bwd <- ggplot(df_backward, aes(x = n_vars, y = accuracy)) +
    geom_line() + geom_point() +
    geom_errorbar(aes(ymin = accuracy - accuracy_sd, ymax = accuracy + accuracy_sd), width = 0.2) +
    labs(title = "Backward Selection Accuracy", x = "Number of Variables", y = "Accuracy") +
    theme_minimal()
  ggsave("backward_selection.png", p_bwd, width = 8, height = 6)
  
  # C. Recursive Feature Elimination (RFE)
  cat("\n--- C. Recursive Feature Elimination (RFE) ---\n")
  rfe_vars <- all_predictors
  rfe_results <- list()
  
  while(length(rfe_vars) >= 1) {
    data_subset <- enyalius_processed[, c(rfe_vars, target_col)]
    cat(paste0("RFE: Training with ", length(rfe_vars), " variables...\n"))
    
    res <- comp_alg(data = data_subset, 
                    target = target_col, 
                    seed = 123,
                    number = 5,
                    repeats = 10,
                    verbose = FALSE)
    best_mod <- res$best_model
    best_acc <- res$metrics[res$metrics$algorithm == best_mod, "accuracy"]
    best_sd <- res$metrics[res$metrics$algorithm == best_mod, "accuracy_sd"]
    
    # Identify least important variable from the best model
    if (length(rfe_vars) > 1) {
      imp_list <- get_var_importance(res)
      worst_var <- as.character(tail(imp_list[[best_mod]]$Variable, 1))
    } else {
      worst_var <- rfe_vars
    }
    
    rfe_results[[length(rfe_results) + 1]] <- data.frame(
      n_vars = length(rfe_vars), 
      best_model = best_mod, 
      accuracy = best_acc, 
      accuracy_sd = best_sd, 
      removed_var = worst_var
    )
    
    if (length(rfe_vars) == 1) break
    rfe_vars <- setdiff(rfe_vars, worst_var)
  }
  df_rfe <- do.call(rbind, rfe_results)
  cat("\n")
  print(df_rfe)

  p_rfe <- ggplot(df_rfe, aes(x = n_vars, y = accuracy)) +
    geom_line() + geom_point() +
    geom_errorbar(aes(ymin = accuracy - accuracy_sd, ymax = accuracy + accuracy_sd), width = 0.2) +
    labs(title = "RFE Accuracy", x = "Number of Variables", y = "Accuracy") +
    theme_minimal()
  ggsave("rfe_selection.png", p_rfe, width = 8, height = 6)

  # --- 6. Advanced Feature Selection Methods ---

  # D. Entropy-based Selection (MRMR with praznik)
  cat("\n--- D. Entropy-based Selection (MRMR with praznik) ---\n")

  # Prepare data for praznik (needs to be discrete/factors)
  # We will discretize numeric columns into bins (e.g., 5 bins)
  data_for_mrmr <- enyalius_processed
  for (col in all_predictors) {
    if (is.numeric(data_for_mrmr[[col]])) {
      # Simple equal-width binning
      data_for_mrmr[[col]] <- cut(data_for_mrmr[[col]], breaks = 5, labels = FALSE)
    }
    data_for_mrmr[[col]] <- as.factor(data_for_mrmr[[col]])
  }

  # Run MRMR
  # k = length(all_predictors) to rank all variables
  mrmr_out <- praznik::MRMR(data_for_mrmr[, all_predictors], data_for_mrmr[[target_col]], k = length(all_predictors))
  
  # Get sorted predictors (praznik returns indices in $selection)
  sorted_predictors_mrmr <- all_predictors[mrmr_out$selection]

  cat("Predictors ordered by MRMR:\n")
  print(sorted_predictors_mrmr)

  mrmr_results <- list()
  for (i in 1:length(sorted_predictors_mrmr)) {
    # Select top i variables based on MRMR ranking
    selected_vars <- sorted_predictors_mrmr[1:i]
    # Reorder them to match the original dataset order (to avoid seed/randomness issues)
    current_vars <- intersect(all_predictors, selected_vars)
    data_subset <- enyalius_processed[, c(current_vars, target_col)]

    cat(paste0("MRMR Step ", i, ": Training with ", length(current_vars), " variables...\n"))

    res <- comp_alg(data = data_subset, 
                    target = target_col, 
                    seed = 123,
                    number = 5,
                    repeats = 10,
                    verbose = FALSE)

    best_mod <- res$best_model
    best_acc <- res$metrics[res$metrics$algorithm == best_mod, "accuracy"]
    best_sd <- res$metrics[res$metrics$algorithm == best_mod, "accuracy_sd"]

    mrmr_results[[i]] <- data.frame(
      n_vars = length(current_vars),
      best_model = best_mod,
      accuracy = best_acc,
      accuracy_sd = best_sd,
      added_var = sorted_predictors_mrmr[i]
    )
  }
  df_mrmr <- do.call(rbind, mrmr_results)
  cat("\n")
  print(df_mrmr)

  p_mrmr <- ggplot(df_mrmr, aes(x = n_vars, y = accuracy)) +
    geom_line() + geom_point() +
    geom_errorbar(aes(ymin = accuracy - accuracy_sd, ymax = accuracy + accuracy_sd), width = 0.2) +
    labs(title = "MRMR Forward Selection", x = "Number of Variables", y = "Accuracy") +
    theme_minimal()
  ggsave("mrmr_selection.png", p_mrmr, width = 8, height = 6)

  # E. Iterative High Correlation Removal
  cat("\n--- E. Iterative High Correlation Removal ---\n")

  # Select only numeric predictors using base R
  numeric_predictor_names <- all_predictors[sapply(enyalius_processed[, all_predictors], is.numeric)]
  current_vars <- numeric_predictor_names
  corr_results <- list()
  step <- 0

  # Loop to remove variables one by one based on highest correlation
  repeat {
    # Calculate correlation matrix for current variables
    cor_matrix <- cor(enyalius_processed[, current_vars])
    diag(cor_matrix) <- 0 # Ignore self-correlation
    
    # Find max absolute correlation
    max_cor <- max(abs(cor_matrix))
    
    # Stop if correlation is low (e.g., < 0.60) or few variables remain
    if (max_cor < 0.60 || length(current_vars) <= 2) {
      cat(paste0("Stopping: Max correlation is ", round(max_cor, 3), "\n"))
      break
    }
    
    step <- step + 1
    
    # Identify the pair with max correlation
    max_idx <- which(abs(cor_matrix) == max_cor, arr.ind = TRUE)[1, ]
    var1 <- rownames(cor_matrix)[max_idx[1]]
    var2 <- colnames(cor_matrix)[max_idx[2]]
    
    # Decide which to remove: the one with higher average correlation with all others
    mean_cor1 <- mean(abs(cor_matrix[var1, ]))
    mean_cor2 <- mean(abs(cor_matrix[var2, ]))
    var_to_remove <- if (mean_cor1 > mean_cor2) var1 else var2
    
    cat(paste0("Step ", step, ": Max Corr = ", round(max_cor, 3), 
               ". Removing '", var_to_remove, "' (correlated with '", if(var_to_remove==var1) var2 else var1, "')\n"))
    
    current_vars <- setdiff(current_vars, var_to_remove)
    
    # Train with remaining variables
    data_subset <- enyalius_processed[, c(current_vars, target_col)]
    cat(paste0("Training with ", length(current_vars), " variables...\n"))
    
    res <- comp_alg(data = data_subset, 
                    target = target_col, 
                    seed = 123,
                    number = 5,
                    repeats = 10,
                    verbose = FALSE)

    best_mod <- res$best_model
    best_acc <- res$metrics[res$metrics$algorithm == best_mod, "accuracy"]
    best_sd <- res$metrics[res$metrics$algorithm == best_mod, "accuracy_sd"]
    
    corr_results[[step]] <- data.frame(
      n_vars = length(current_vars),
      best_model = best_mod,
      accuracy = best_acc,
      accuracy_sd = best_sd,
      removed_var = var_to_remove,
      max_corr_at_step = max_cor
    )
  }
  
  if (length(corr_results) > 0) {
    df_corr <- do.call(rbind, corr_results)
    print(df_corr)
    
    p_corr <- ggplot(df_corr, aes(x = n_vars, y = accuracy)) +
      geom_line() + geom_point() +
      geom_errorbar(aes(ymin = accuracy - accuracy_sd, ymax = accuracy + accuracy_sd), width = 0.2) +
      labs(title = "Iterative Correlation Removal", x = "Number of Variables", y = "Accuracy") +
      theme_minimal()
    ggsave("correlation_selection.png", p_corr, width = 8, height = 6)
  }

  # Combined Plot (All Methods)
  df_forward$Method <- "Forward"
  df_backward$Method <- "Backward"
  df_rfe$Method <- "RFE"
  df_mrmr$Method <- "MRMR"
  if (exists("df_corr")) df_corr$Method <- "Correlation Filter"

  df_all <- rbind(df_forward[, c("n_vars", "accuracy", "accuracy_sd", "Method")],
                  df_backward[, c("n_vars", "accuracy", "accuracy_sd", "Method")],
                  df_rfe[, c("n_vars", "accuracy", "accuracy_sd", "Method")],
                  df_mrmr[, c("n_vars", "accuracy", "accuracy_sd", "Method")])
  
  if (exists("df_corr")) {
    df_all <- rbind(df_all, df_corr[, c("n_vars", "accuracy", "accuracy_sd", "Method")])
  }
  
  p_all <- ggplot(df_all, aes(x = n_vars, y = accuracy, color = Method)) +
    geom_line() + geom_point() +
    geom_errorbar(aes(ymin = accuracy - accuracy_sd, ymax = accuracy + accuracy_sd), width = 0.2) +
    labs(title = "Feature Selection Comparison (All Methods)", x = "Number of Variables", y = "Accuracy") +
    theme_minimal()
  ggsave("feature_selection_comparison_all.png", p_all, width = 10, height = 6)

  # F. Stochastic Search (Hill Climbing / Simulated Annealing)
  # Goal: Try to find a better subset of fixed size (e.g., 6) by swapping variables
  cat("\n--- F. Stochastic Search (Hill Climbing) ---\n")
  
  # Start with the top 6 variables from MRMR (a good starting point)
  # We want to see if we can beat the MRMR accuracy with the SAME number of variables
  n_vars_target <- 6
  current_vars <- sorted_predictors_mrmr[1:n_vars_target]
  
  # Baseline evaluation
  data_subset <- enyalius_processed[, c(current_vars, target_col)]
  res <- comp_alg(data = data_subset, target = target_col, seed = 123, number = 5, repeats = 5, verbose = FALSE)
  best_sa_acc <- res$metrics[res$metrics$algorithm == res$best_model, "accuracy"]
  
  cat(paste0("Baseline (MRMR Top ", n_vars_target, "): ", round(best_sa_acc, 4), 
             " | Vars: ", paste(current_vars, collapse=","), "\n"))
  
  # Simple loop to try and improve the subset
  # In a real scenario, you would run this for hundreds of iterations
  set.seed(123)
  for (i in 1:15) {
    # 1. Propose a change: swap one variable inside with one outside
    available_vars <- setdiff(all_predictors, current_vars)
    if(length(available_vars) == 0) break
    
    remove_cand <- sample(current_vars, 1)
    add_cand <- sample(available_vars, 1)
    
    new_vars <- c(setdiff(current_vars, remove_cand), add_cand)
    
    # 2. Evaluate new subset
    data_subset <- enyalius_processed[, c(new_vars, target_col)]
    # Using fewer repeats (5) for speed during search
    res <- comp_alg(data = data_subset, target = target_col, seed = 123, number = 5, repeats = 5, verbose = FALSE)
    new_acc <- res$metrics[res$metrics$algorithm == res$best_model, "accuracy"]
    
    # 3. Accept if better
    if (new_acc > best_sa_acc) {
      cat(paste0("Iter ", i, ": IMPROVED! ", round(best_sa_acc, 4), " -> ", round(new_acc, 4), "\n"))
      cat(paste0("  Swapped OUT: '", remove_cand, "' | IN: '", add_cand, "'\n"))
      best_sa_acc <- new_acc
      current_vars <- new_vars
    } else {
      cat(paste0("Iter ", i, ": No improvement (", round(new_acc, 4), ")\n"))
    }
  }
  
  cat("\nBest subset found by Stochastic Search:\n")
  print(current_vars)
  cat(paste0("Final Accuracy: ", round(best_sa_acc, 4), "\n"))
}

cat("\n\n--- Test script finished successfully. ---\n")
