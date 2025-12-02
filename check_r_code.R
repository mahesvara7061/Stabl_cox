suppressPackageStartupMessages({
  for (p in c("readr","dplyr","survival","survminer","glmnet"))
    if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
  library(readr); library(dplyr); library(survival); library(survminer); library(glmnet)
})

# -------- EDIT THESE PATHS --------
expr_path <- "~/Documents/Labs/Pharmaco-Omics/first_project/data/processed/final/TCGA_CGGA_counts.csv"
clin_path <- "~/Documents/Labs/Pharmaco-Omics/first_project/data/processed/final/TCGA_CGGA_clinical.csv"
out_dir   <- dirname(expr_path)

# Optional: filter by histology (e.g., "GBM"). Set NULL to skip.
FILTER_HISTOLOGY <- NULL  # e.g., "GBM"

TOP_N_KM <- 5  # how many top genes to plot KM for

# -------- Load data --------
expr <- readr::read_csv(expr_path, show_col_types = FALSE)
stopifnot(ncol(expr) >= 3)
X <- as.matrix(expr[,-1, drop=FALSE])
rownames(X) <- expr[[1]]

clin <- readr::read_csv(clin_path, show_col_types = FALSE)

# Basic checks + rename for survival
stopifnot(all(c("sample_id","OS","Censor") %in% names(clin)))
clin <- clin %>%
  mutate(
    time  = as.numeric(OS),
    event = as.integer(Censor)  # 1 = death, 0 = censored
  )

# Optional histology filter
if (!is.null(FILTER_HISTOLOGY) && "Histology" %in% names(clin)) {
  clin <- clin %>% filter(.data$Histology == FILTER_HISTOLOGY)
}

# Align samples
common <- intersect(colnames(X), clin$sample_id)
stopifnot(length(common) >= 5)  # need enough samples
X <- X[, common, drop=FALSE]
clin <- clin %>%
  filter(sample_id %in% common) %>%
  arrange(match(sample_id, common))

# Remove rows with missing/invalid survival
clin <- clin %>% filter(!is.na(time), !is.na(event), time > 0)
common <- clin$sample_id
X <- X[, common, drop=FALSE]

# -------- Survival object --------
y <- with(clin, Surv(time, event))

# -------- Prepare data for glmnet (samples x genes) --------
# glmnet expects: rows = samples, columns = features
X_t <- t(X)  # transpose so samples are rows

# Remove genes with zero variance
gene_vars <- apply(X_t, 2, var, na.rm = TRUE)
X_t <- X_t[, gene_vars > 0, drop = FALSE]
message("Using ", ncol(X_t), " genes with non-zero variance")

# -------- Cox LASSO with cross-validation --------
message("Running Cox LASSO with 10-fold cross-validation...")
set.seed(123)  # for reproducibility

# cv.glmnet to find optimal lambda
cv_fit <- cv.glmnet(
  x = X_t,
  y = y,
  family = "cox",
  alpha = 1,        # 1 = LASSO, 0 = Ridge, 0.5 = Elastic Net
  nfolds = 10,
  standardize = TRUE,
  maxit = 1000
)

# Plot CV results
pdf(file.path(out_dir, "cox_lasso_cv_plot.pdf"), width = 8, height = 6)
plot(cv_fit)
title("Cox LASSO Cross-Validation", line = 2.5)
dev.off()
message("Saved: ", file.path(out_dir, "cox_lasso_cv_plot.pdf"))

# -------- Extract results at lambda.min and lambda.1se --------
# lambda.min: minimizes CV error
# lambda.1se: most regularized model within 1 SE of minimum

results_list <- list()
for (lambda_type in c("lambda.min", "lambda.1se")) {
  lambda_val <- cv_fit[[lambda_type]]
  
  # Extract coefficients
  coefs <- coef(cv_fit, s = lambda_val)
  coef_df <- data.frame(
    gene = rownames(coefs),
    coefficient = as.numeric(coefs),
    stringsAsFactors = FALSE
  )
  
  # Keep only non-zero coefficients
  coef_df <- coef_df %>%
    filter(coefficient != 0) %>%
    mutate(
      HR = exp(coefficient),
      abs_coef = abs(coefficient)
    ) %>%
    arrange(desc(abs_coef))
  
  results_list[[lambda_type]] <- coef_df
  
  # Save results
  out_csv <- file.path(out_dir, paste0("cox_lasso_", lambda_type, "_genes.csv"))
  readr::write_csv(coef_df, out_csv)
  message("Saved ", nrow(coef_df), " genes at ", lambda_type, ": ", out_csv)
}

# -------- Full model info --------
model_info <- data.frame(
  metric = c("lambda.min", "lambda.1se", "n_genes_min", "n_genes_1se"),
  value = c(
    cv_fit$lambda.min,
    cv_fit$lambda.1se,
    nrow(results_list$lambda.min),
    nrow(results_list$lambda.1se)
  )
)
readr::write_csv(model_info, file.path(out_dir, "cox_lasso_model_info.csv"))

# -------- Coefficient path plot --------
pdf(file.path(out_dir, "cox_lasso_coefficient_path.pdf"), width = 10, height = 6)
plot(cv_fit$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_fit$lambda.min), col = "red", lty = 2)
abline(v = log(cv_fit$lambda.1se), col = "blue", lty = 2)
legend("topright", 
       legend = c("lambda.min", "lambda.1se"), 
       col = c("red", "blue"), 
       lty = 2, 
       cex = 0.8)
title("Cox LASSO Coefficient Path", line = 2.5)
dev.off()
message("Saved: ", file.path(out_dir, "cox_lasso_coefficient_path.pdf"))

# -------- Optional: Kaplan-Meier plots for top genes --------
message("\nGenerating Kaplan-Meier plots for top ", TOP_N_KM, " genes...")

top_genes <- head(results_list$lambda.min, TOP_N_KM)$gene

for (g in top_genes) {
  if (!g %in% rownames(X)) next
  
  gene_expr <- X[g, ]
  median_expr <- median(gene_expr, na.rm = TRUE)
  
  clin$group <- ifelse(gene_expr > median_expr, "High", "Low")
  
  fit <- survfit(Surv(time, event) ~ group, data = clin)
  
  p <- ggsurvplot(
    fit,
    data = clin,
    pval = TRUE,
    risk.table = TRUE,
    title = paste0("Gene: ", g),
    xlab = "Time (days)",
    ylab = "Overall Survival Probability",
    legend.title = "Expression",
    legend.labs = c("High", "Low"),
    palette = c("red", "blue")
  )
  
  pdf_file <- file.path(out_dir, paste0("KM_", g, ".pdf"))
  pdf(pdf_file, width = 8, height = 8)
  print(p)
  dev.off()
}

message("\n=== Cox LASSO Analysis Complete ===")
message("Selected genes at lambda.min: ", nrow(results_list$lambda.min))
message("Selected genes at lambda.1se: ", nrow(results_list$lambda.1se))