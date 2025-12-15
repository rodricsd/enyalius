#' @export
print.diagnoseR_result <- function(x, ...) {
  cat("\n--- Resultados da Regra do Erro Padrão ---\n")
  cat("Threshold (Melhor Acuracia - Melhor SE):", format(x$one_se_threshold, digits = 4), "\n")
  cat("Melhor modelo (mais estável dos candidatos):", x$best_model, "\n\n")
  cat("Metricas de Performance (CV no conjunto de treino):\n")
  
  # Reordenar colunas para clareza e formatar digitos
  metrics_to_print <- x$metrics[, c("algorithm", "candidate", "accuracy", "accuracy_sd", "kappa", "dratio", "accuracy_se")]
  
  print(metrics_to_print, row.names = FALSE)
  
  cat("-------------------------------------\n")
}
