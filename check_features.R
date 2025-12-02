suppressPackageStartupMessages({
  for (p in c("readr","dplyr","survival","survminer"))
    if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
  library(readr); library(dplyr); library(survival); library(survminer)
})

# -------- EDIT THESE PATHS --------
expr_path <- "/home/mahesvara/Documents/Labs/Pharmaco-Omics/first_project/source/results_survival_10/fold_data_files/fold1_mRNA/counts_train.csv"
clin_path <- "/home/mahesvara/Documents/Labs/Pharmaco-Omics/first_project/source/results_survival_10/fold_data_files/fold1_mRNA/clinical_train.csv"
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

# -------- Univariate Cox for all genes --------
cox_one <- function(v) {
  # Skip if constant or all NA
  if (all(!is.finite(v)) || sd(v, na.rm=TRUE) == 0) return(c(beta=NA, HR=NA, z=NA, p=NA))
  sm <- tryCatch(summary(coxph(y ~ v)), error = function(e) NULL)
  if (is.null(sm)) return(c(beta=NA, HR=NA, z=NA, p=NA))
  c(beta = sm$coefficients[1,"coef"],
    HR   = exp(sm$coefficients[1,"coef"]),
    z    = sm$coefficients[1,"z"],
    p    = sm$coefficients[1,"Pr(>|z|)"])
}

message("Running univariate Cox on ", nrow(X), " genes ...")
res <- t(apply(X, 1, cox_one))
df  <- as.data.frame(res)
df$gene <- rownames(X)
df <- df[complete.cases(df$p), ]
df$FDR <- p.adjust(df$p, method = "BH")
df <- df[order(df$p), ]

# Save results
out_csv <- file.path(out_dir, "univariate_cox_all_genes.csv")
readr::write_csv(df, out_csv)
message("Saved: ", out_csv)
