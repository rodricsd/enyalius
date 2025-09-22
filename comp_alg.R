# Testando a função com o dataset 'iris'
library(datasets)
library(diagnoseR)

iris <- datasets::iris

results <- diagnoseR::comp_alg(data = iris,
                    target = "Species",
                    train_val = 0.8,
                    seed = 42,
                    cv_folds = 5,
                    verbose = FALSE)

results