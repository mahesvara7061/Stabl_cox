# ============================================================================
# BATCH MACHINE LEARNING PIPELINE RUNNER
# ============================================================================
# This function runs multiple ML models (RF, SVM, PLS-DA) repeatedly with
# different random seeds to assess model stability and performance variability
# 
# OUTPUTS:
#   - summary_runs.csv: Performance metrics for each run
#   - varImp/*.csv: Variable importance scores for each run
#   - roc/*.tiff: ROC curve plots for each run
#   - seeds_used.csv: Record of seeds used in each run
# ============================================================================

# ----------------------------------------------------------------------------
# SOURCE ML FUNCTIONS
# Loads the ML pipeline functions defined in MLFunction.R
# MODIFY: Update path to match your directory structure
# ----------------------------------------------------------------------------
# source("D:/Labs/Pharmaco-Omics/ibd meta-analysis/source/ML_Function_Refactored.R")
source("/mnt/d/Labs/Pharmaco-Omics/stabl/Stabl_cox/ML_Function_Refactored.R")
# ============================================================================
# MAIN BATCH RUNNER FUNCTION
# ============================================================================
run_ml_batch <- function(
    df,                                             # Input dataframe with 'Label' column
    models = c("RF", "SVM", "PLSDA"),              # MODIFY: Models to run
    classname = c("CD", "UC"),                     # MODIFY: Your class labels
    drop_cols = NULL,                               # MODIFY: Columns to exclude from training
    non_continuous_cols = c("Male", "Female"),     # MODIFY: Numeric but non-continuous columns
    n_runs = 200,                                   # MODIFY: Number of times to run each model
    seed_mode = c("increment", "file"),            # MODIFY: How to generate seeds
    base_seed = 20250966,                          # MODIFY: Starting seed for increment mode
    seeds_file = NULL,                              # MODIFY: Path to CSV file with seeds
    start_index = 1,                                # For future use (currently unused)
    out_dir = "results",                            # MODIFY: Output directory path
    split_ratio = 0.7,                              # MODIFY: Train/test split ratio
    kouter = 5,                                     # MODIFY: Outer CV folds
    kinner = 5,                                     # MODIFY: Inner CV folds
    dataprocessing_method = c("center", "scale")   # MODIFY: SVM preprocessing methods
) {
  
  # --------------------------------------------------------------------------
  # LOAD REQUIRED LIBRARIES
  # Suppress startup messages for cleaner output
  # --------------------------------------------------------------------------
  suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(purrr)
    library(tibble)
    library(ggplot2)
  })
  
  # --------------------------------------------------------------------------
  # INPUT VALIDATION
  # Ensure dataframe has required 'Label' column
  # --------------------------------------------------------------------------
  stopifnot("Label" %in% names(df))
  
  # --------------------------------------------------------------------------
  # DATA PREPARATION - COLUMN EXCLUSION
  # Remove specified columns that should not be used in training
  # MODIFY drop_cols parameter to exclude unwanted columns
  # --------------------------------------------------------------------------
  if (!is.null(drop_cols)) {
    drop_cols <- intersect(drop_cols, names(df))
    if (length(drop_cols) > 0) {
      df <- dplyr::select(df, -all_of(drop_cols))
      cat(sprintf("Dropped %d columns: %s\n", 
                  length(drop_cols), paste(drop_cols, collapse = ", ")))
    }
  }
  
  # --------------------------------------------------------------------------
  # IDENTIFY CONTINUOUS NUMERIC COLUMNS FOR SVM PREPROCESSING
  # SVM requires preprocessing of continuous features only
  # Excludes 'Label' and non-continuous numeric columns (e.g., binary indicators)
  # --------------------------------------------------------------------------
  all_numeric <- names(dplyr::select(df, where(is.numeric)))
  continuous_cols <- setdiff(all_numeric, c("Label", non_continuous_cols))
  
  cat(sprintf("Identified %d continuous columns for SVM preprocessing\n", 
              length(continuous_cols)))
  
  # --------------------------------------------------------------------------
  # SEED GENERATION STRATEGY
  # Two modes available:
  #   - "increment": Generate seeds sequentially from base_seed
  #   - "file": Load seeds from CSV file (must have 'seed' column)
  # --------------------------------------------------------------------------
  seed_mode <- match.arg(seed_mode)
  
  if (seed_mode == "file") {
    # Load seeds from external file
    stopifnot(!is.null(seeds_file), file.exists(seeds_file))
    seeds_tbl <- readr::read_csv(seeds_file, show_col_types = FALSE)
    stopifnot("seed" %in% names(seeds_tbl))
    seeds_vec <- seeds_tbl$seed
    
    # Verify sufficient seeds available
    if (length(seeds_vec) < n_runs) {
      stop(sprintf("Seed file contains only %d seeds but n_runs = %d", 
                   length(seeds_vec), n_runs))
    }
    seeds_vec <- seeds_vec[seq_len(n_runs)]
    cat(sprintf("Loaded %d seeds from file: %s\n", n_runs, seeds_file))
    
  } else {
    # Generate sequential seeds
    seeds_vec <- base_seed + seq_len(n_runs) - 1
    cat(sprintf("Generated %d seeds starting from %d\n", n_runs, base_seed))
  }
  
  # --------------------------------------------------------------------------
  # OUTPUT DIRECTORY SETUP
  # Create main output directory and save seed information
  # --------------------------------------------------------------------------
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save seeds used for reproducibility
  readr::write_csv(
    tibble(run_id = seq_len(n_runs), seed = seeds_vec),
    file.path(out_dir, "seeds_used.csv")
  )
  cat(sprintf("Output directory: %s\n", normalizePath(out_dir)))
  
  # --------------------------------------------------------------------------
  # HELPER FUNCTION: PREPARE MODEL DIRECTORIES
  # Creates directory structure for each model:
  #   - model_name/
  #   - model_name/varImp/  (variable importance CSVs)
  #   - model_name/roc/     (ROC curve plots)
  # --------------------------------------------------------------------------
  prep_model_dirs <- function(model_name) {
    model_dir <- file.path(out_dir, model_name)
    dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(model_dir, "varImp"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(model_dir, "roc"), recursive = TRUE, showWarnings = FALSE)
    return(model_dir)
  }
  
  # --------------------------------------------------------------------------
  # HELPER FUNCTION: APPEND SUMMARY ROW
  # Appends performance metrics to summary CSV file
  # Creates new file with headers if it doesn't exist
  # --------------------------------------------------------------------------
  append_summary_row <- function(model_dir, row_df) {
    out_csv <- file.path(model_dir, "summary_runs.csv")
    
    if (!file.exists(out_csv)) {
      # First write - create file with headers
      readr::write_csv(row_df, out_csv)
    } else {
      # Subsequent writes - append without headers
      write.table(row_df, file = out_csv, sep = ",", row.names = FALSE,
                  col.names = FALSE, append = TRUE, qmethod = "double")
    }
  }
  
  # --------------------------------------------------------------------------
  # HELPER FUNCTION: CREATE SUMMARY ROW
  # Constructs a single row of performance metrics for summary table
  # Includes: run_id, seed, AUC, sensitivity, specificity, balanced accuracy
  # --------------------------------------------------------------------------
  make_summary_row <- function(run_id, seed, auc_mean, Metrics_Table) {
    tibble(
      run_id = run_id, 
      seed = seed, 
      auc_mean = as.numeric(auc_mean)
    ) %>%
      bind_cols(Metrics_Table %>% mutate(across(everything(), as.numeric)))
  }
  
  # --------------------------------------------------------------------------
  # HELPER FUNCTION: SAVE VARIABLE IMPORTANCE AND ROC CURVE
  # Saves two outputs for each run:
  #   1. Variable importance scores as CSV
  #   2. ROC curve as high-resolution TIFF (600 DPI)
  # Files are numbered with 3-digit zero-padded run IDs (001, 002, ...)
  # --------------------------------------------------------------------------
  save_varimp_and_roc <- function(model_dir, run_id, varImp_df, p_RocCurve) {
    # Construct file paths with zero-padded numbering
    vi_path <- file.path(model_dir, "varImp", sprintf("%03d.csv", run_id))
    roc_path <- file.path(model_dir, "roc", sprintf("%03d.tiff", run_id))
    
    # Convert rownames to column for CSV export
    vi_to_write <- varImp_df %>% tibble::rownames_to_column(var = "feature")
    readr::write_csv(vi_to_write, vi_path)
    
    # Save ROC plot as high-quality TIFF
    ggsave(
      filename = roc_path, 
      plot = p_RocCurve, 
      width = 6,        # MODIFY: Adjust plot dimensions
      height = 6, 
      dpi = 600         # MODIFY: Adjust resolution
    )
  }
  
  # --------------------------------------------------------------------------
  # MODEL SELECTION AND VALIDATION
  # Validate requested models and prepare their directories
  # --------------------------------------------------------------------------
  models <- toupper(models)
  allowed <- c("RF", "SVM", "PLSDA", "LOGIT")
  
  if (any(!models %in% allowed)) {
    stop(sprintf("Invalid model(s): %s. Allowed models: RF, SVM, PLSDA, LOGIT", 
                 paste(setdiff(models, allowed), collapse = ", ")))
  }
  
  # Create directories for selected models
  dirs <- list()
  if ("RF" %in% models) {
    dirs$RF <- prep_model_dirs("RF")
    cat("Prepared directory for Random Forest\n")
  }
  if ("SVM" %in% models) {
    dirs$SVM <- prep_model_dirs("SVM")
    cat("Prepared directory for Support Vector Machine\n")
  }
  if ("PLSDA" %in% models) {
    dirs$PLSDA <- prep_model_dirs("PLSDA")
    cat("Prepared directory for PLS-DA\n")
  }
  if ("LOGIT" %in% models) {
    dirs$LOGIT <- prep_model_dirs("LOGIT")
    cat("Prepared directory for Logistic Regression\n")
  }
  
  cat(sprintf("\n%s\n", paste(rep("=", 70), collapse = "")))
  cat(sprintf("Starting batch run: %d iterations for %d model(s)\n", 
              n_runs, length(models)))
  cat(sprintf("%s\n\n", paste(rep("=", 70), collapse = "")))
  
  # --------------------------------------------------------------------------
  # MAIN EXECUTION LOOP
  # Runs each model n_runs times with different seeds
  # Uses try() to catch errors without stopping entire batch
  # --------------------------------------------------------------------------
  for (i in seq_len(n_runs)) {
    seed_r <- seeds_vec[i]
    
    cat(sprintf(">>> Run %d/%d | Seed = %d | Models = [%s]\n",
                i, n_runs, seed_r, paste(models, collapse = ", ")))
    
    # ------------------------------------------------------------------------
    # RANDOM FOREST EXECUTION
    # Runs RF model and saves results
    # Errors are caught and logged without stopping the batch
    # ------------------------------------------------------------------------
    if ("RF" %in% models) {
      rf_ok <- try({
        # Run Random Forest model
        rf_res <- rf_perform_std_Revised(
          data_input_ML = df,
          classname = classname,
          seed = seed_r,
          split_ratio = split_ratio,
          k_fold_outer = kouter,
          k_fold_inner = kinner,
          useModel_for_varImp = TRUE
        )
        
        # Save summary metrics
        rf_row <- make_summary_row(i, seed_r, rf_res$auc_mean, rf_res$Metrics_Table)
        append_summary_row(dirs$RF, rf_row)
        
        # Save variable importance and ROC curve
        save_varimp_and_roc(dirs$RF, i, rf_res$varImp_score_df, rf_res$p_RocCurve)
        
        cat(sprintf("    [RF] AUC = %.3f | Complete\n", rf_res$auc_mean))
      }, silent = TRUE)
      
      # Log error if RF failed
      if (inherits(rf_ok, "try-error")) {
        error_msg <- attr(rf_ok, "condition")$message
        message(sprintf("[WARNING][RF] Run %d failed: %s", i, error_msg))
      }
    }
    
    # ------------------------------------------------------------------------
    # PLS-DA EXECUTION
    # Runs PLS-DA model and saves results
    # ------------------------------------------------------------------------
    if ("PLSDA" %in% models) {
      pls_ok <- try({
        # Run PLS-DA model
        pls_res <- plsda_perform_std_Revised(
          data_input_ML = df,
          classname = classname,
          seed = seed_r,
          split_ratio = split_ratio,
          k_fold_outer = kouter,
          k_fold_inner = kinner,
          useModel_for_varImp = TRUE,
          column_to_process = continuous_cols,
          dataprocessing_method = dataprocessing_method
        )
        
        # Save summary metrics
        pls_row <- make_summary_row(i, seed_r, pls_res$auc_mean, pls_res$Metrics_Table)
        append_summary_row(dirs$PLSDA, pls_row)
        
        # Save variable importance and ROC curve
        save_varimp_and_roc(dirs$PLSDA, i, pls_res$varImp_score_df, pls_res$p_RocCurve)
        
        cat(sprintf("    [PLS-DA] AUC = %.3f | Complete\n", pls_res$auc_mean))
      }, silent = TRUE)
      
      # Log error if PLS-DA failed
      if (inherits(pls_ok, "try-error")) {
        error_msg <- attr(pls_ok, "condition")$message
        message(sprintf("[WARNING][PLS-DA] Run %d failed: %s", i, error_msg))
      }
    }
    
    # ------------------------------------------------------------------------
    # SVM EXECUTION
    # Runs SVM model with preprocessing on continuous columns
    # ------------------------------------------------------------------------
    if ("SVM" %in% models) {
      svm_ok <- try({
        # Run SVM model
        svm_res <- svm_perform_std_Revised(
          data_input_ML = df,
          classname = classname,
          column_to_process = continuous_cols,
          dataprocessing_method = dataprocessing_method,
          seed = seed_r,
          split_ratio = split_ratio,
          k_fold_outer = kouter,
          k_fold_inner = kinner,
          useModel_for_varImp = TRUE
        )
        
        # Save summary metrics
        svm_row <- make_summary_row(i, seed_r, svm_res$auc_mean, svm_res$Metrics_Table)
        append_summary_row(dirs$SVM, svm_row)
        
        # Save variable importance and ROC curve
        save_varimp_and_roc(dirs$SVM, i, svm_res$varImp_score_df, svm_res$p_RocCurve)
        
        cat(sprintf("    [SVM] AUC = %.3f | Complete\n", svm_res$auc_mean))
      }, silent = TRUE)
      
      # Log error if SVM failed
      if (inherits(svm_ok, "try-error")) {
        error_msg <- attr(svm_ok, "condition")$message
        message(sprintf("[WARNING][SVM] Run %d failed: %s", i, error_msg))
      }
    }

    if ("LOGIT" %in% models) {
      logit_ok <- try({
        # Run LOGIT model
        logit_res <- logit_perform_std_Revised(
          data_input_ML = df,
          classname = classname,
          seed = seed_r,
          split_ratio = split_ratio,
          k_fold_outer = kouter,
          k_fold_inner = kinner,
          useModel_for_varImp = TRUE,
          column_to_process = continuous_cols,
          dataprocessing_method = dataprocessing_method
        )
        
        # Save summary metrics
        logit_row <- make_summary_row(i, seed_r, logit_res$auc_mean, logit_res$Metrics_Table)
        append_summary_row(dirs$LOGIT, logit_row)
        
        # Save variable importance and ROC curve
        save_varimp_and_roc(dirs$LOGIT, i, logit_res$varImp_score_df, logit_res$p_RocCurve)
        
        cat(sprintf("    [LOGIT] AUC = %.3f | Complete\n", logit_res$auc_mean))
      }, silent = TRUE)
      
      # Log error if LOGIT failed
      if (inherits(logit_ok, "try-error")) {
        error_msg <- attr(logit_ok, "condition")$message
        message(sprintf("[WARNING][LOGIT] Run %d failed: %s", i, error_msg))
      }
    }
    
    closeAllConnections()
    
    gc()
    
    Sys.sleep(0.5)
    
    cat("\n")
  }
  
  # --------------------------------------------------------------------------
  # COMPLETION MESSAGE
  # Display final summary and output location
  # --------------------------------------------------------------------------
  cat(sprintf("\n%s\n", paste(rep("=", 70), collapse = "")))
  cat(">>> BATCH RUN COMPLETED SUCCESSFULLY\n")
  cat(sprintf("Results saved to: %s\n", normalizePath(out_dir)))
  cat(sprintf("%s\n", paste(rep("=", 70), collapse = "")))
  
  invisible(TRUE)
}

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
# Prepare your data
my_data <- read.csv("/mnt/d/Labs/Pharmaco-Omics/stabl/data/HCC/tcga_vst_only_HCC_recurrence_correct_num_genes/ML_Input_TCGA_LIHC_Recurrence_5_genes.csv", row.names = 1, check.names = FALSE)

# Run batch analysis with clinical covariates included
run_ml_batch(
  df= my_data,
  models            = c("LOGIT", "RF", "PLSDA", "SVM"),
  classname         = c("Recurrence", "Non_Recurrence"),
  drop_cols         = NULL,
  non_continuous_cols = NULL,      # numeric but non-continuous columns
  n_runs            = 50,
  seed_mode         = c("increment","file"),
  base_seed         = 20251202,
  seeds_file        = NULL,
  start_index       = 1,
  out_dir           = "/mnt/d/labs/pharmaco-omics/stabl/results_HCC_Recurrence_5_genes",
  split_ratio       = 0.7,
  kouter            = 5,
  kinner            = 5,
  dataprocessing_method = c("center","scale")
)
# run_ml_batch(
#   df= my_data,
#   models            = c("SVM"),
#   classname         = c("IBD", "HC"),
#   drop_cols         = NULL,
#   non_continuous_cols = NULL,      # numeric but non-continuous columns
#   n_runs            = 200,
#   seed_mode         = c("increment","file"),
#   base_seed         = 20251202,
#   seeds_file        = NULL,  
#   start_index       = 1,
#   out_dir           = "results_IBD_HC_SVM_val",
#   split_ratio       = 0.7,
#   kouter            = 5,
#   kinner            = 5,
#   dataprocessing_method = c("center","scale")
# )

