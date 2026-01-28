suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(tibble); library(stringr)
})

# ==============================================================================
# SECTION 1: HELPER FUNCTIONS
# ==============================================================================

# --- 1.1 Read Summary CSV (Handles duplicates and missing columns) ---
read_summary <- function(path) {
  if (!file.exists(path)) {
    # warning(sprintf("File not found: %s", path))
    return(NULL)
  }
  dat <- readr::read_csv(path, show_col_types = FALSE)
  need <- c("run_id", "auc_mean")
  if (!all(need %in% names(dat))) {
    return(NULL)
  }
  
  dat <- dat %>%
    transmute(
      run_id   = as.integer(run_id),
      auc_mean = suppressWarnings(as.numeric(auc_mean))
    ) %>%
    filter(is.finite(auc_mean))
  
  # Handle duplicate run_ids by averaging
  dups <- dat %>% count(run_id) %>% filter(n > 1)
  if (nrow(dups) > 0) {
    dat <- dat %>%
      group_by(run_id) %>%
      summarise(auc_mean = mean(auc_mean), .groups = "drop")
  } else {
    dat <- arrange(dat, run_id)
  }
  dat
}

# --- 1.2 Plot Distribution ---
plot_model_dist <- function(df, model_name, dataset_name, mu, sd, x_run, outfile,
                            show_selected = TRUE, bins = 30, base_size = 13) {
  if (is.na(mu) || !is.finite(mu)) mu <- mean(df$auc_mean, na.rm = TRUE)
  if (is.na(sd) || !is.finite(sd) || sd <= 0) sd <- 1e-9
  
  outdir <- dirname(outfile)
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
  p <- ggplot(df, aes(x = auc_mean)) +
    geom_histogram(aes(y = after_stat(density)), bins = bins, alpha = 0.5, fill = "grey70") +
    geom_density(aes(color = "Kernel Density"), linewidth = 1, key_glyph = "path") +
    stat_function(fun = dnorm, args = list(mean = mu, sd = sd),
                  aes(color = "Normal Fit"), linewidth = 1, key_glyph = "path") +
    geom_vline(aes(xintercept = mu, linetype = "Mean"), color = "black", linewidth = 0.6) +
    labs(
      title    = paste0(dataset_name, " | ", model_name),
      subtitle = sprintf("Mean=%.4f | SD=%.4f | Selected=%.4f", mu, sd, x_run),
      x = "AUC Mean", y = "Density", color = "Curve", linetype = "Reference"
    ) +
    scale_color_manual(values = c("Kernel Density" = "#1f77b4", "Normal Fit" = "#2ca02c")) +
    scale_linetype_manual(values = c("Mean" = "dotted", "Selected" = "solid")) +
    theme_minimal(base_size = base_size)
  
  if (isTRUE(show_selected) && !is.na(x_run)) {
    p <- p + geom_vline(aes(xintercept = x_run, linetype = "Selected"),
                        color = "red", linewidth = 0.6)
  }
  
  ggsave(outfile, plot = p, width = 7, height = 5, dpi = 300)
  p
}

# --- 1.3 Helper to copy ROC/VarImp files ---
copy_associated_files <- function(src_dir, model, run_id, out_dir, prefix_name) {
  run_str <- sprintf("%03d", run_id)
  clean_prefix <- str_replace_all(prefix_name, "[^A-Za-z0-9]", "_")
  
  # Copy ROC TIFF
  roc_dir <- file.path(src_dir, model, "roc")
  if (dir.exists(roc_dir)) {
    tiff_files <- list.files(roc_dir, pattern = paste0(run_str, ".tiff?$"), full.names = TRUE, ignore.case = TRUE)
    if (length(tiff_files) > 0) {
      dest <- file.path(out_dir, paste0(clean_prefix, "_ROC_", model, ".tiff"))
      file.copy(tiff_files[1], dest, overwrite = TRUE)
    }
  }
  
  # Copy VarImp CSV
  vi_path <- file.path(src_dir, model, "varImp", paste0(run_str, ".csv"))
  if (file.exists(vi_path)) {
    dest <- file.path(out_dir, paste0(clean_prefix, "_VarImp_", model, ".csv"))
    file.copy(vi_path, dest, overwrite = TRUE)
  }
}

# ==============================================================================
# SECTION 2: SINGLE DATASET ANALYSIS (From V1)
# ==============================================================================
# Use this when you want to analyze ONE folder only.
analyze_auc_single <- function(
    input_dir, models = c("RF", "PLSDA", "SVM", "LOGIT"),
    top_k = 10, pick  = c("first", "second", "kth"), pick_k_index = 2, out_dir = NULL
) {
  pick <- match.arg(pick)
  paths <- setNames(file.path(input_dir, models, "summary_runs.csv"), models)
  
  # Load Data
  data_list <- lapply(names(paths), function(m) {
    d <- read_summary(paths[[m]])
    if (!is.null(d)) d$model <- m
    d
  })
  names(data_list) <- models
  
  valid_models <- models[!sapply(data_list, is.null)]
  if (length(valid_models) == 0) stop("No valid models found.")
  data_list <- data_list[valid_models]
  
  # Intersect IDs
  list_of_ids <- lapply(data_list, `[[`, "run_id")
  common_ids  <- Reduce(intersect, list_of_ids)
  if (length(common_ids) == 0) stop("No common run_ids found.")
  
  for (m in valid_models) {
    data_list[[m]] <- data_list[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
  }
  
  stats <- lapply(data_list, function(d) list(mean = mean(d$auc_mean), sd = sd(d$auc_mean)))
  
  # Calculate Z-scores for stability
  merged <- data_list[[valid_models[1]]] %>% select(run_id, auc_mean)
  names(merged)[names(merged) == "auc_mean"] <- paste0("auc_", valid_models[1])
  
  if (length(valid_models) > 1) {
    for (m in valid_models[-1]) {
      tmp <- data_list[[m]] %>% select(run_id, auc_mean)
      names(tmp)[names(tmp) == "auc_mean"] <- paste0("auc_", m)
      merged <- inner_join(merged, tmp, by = "run_id")
    }
  }
  
  merged_calc <- merged
  z_cols <- c()
  for (m in valid_models) {
    auc_col <- paste0("auc_", m)
    z_col   <- paste0("z_", m)
    mu <- stats[[m]]$mean; sd <- stats[[m]]$sd
    if(is.na(sd) || sd==0) sd <- 1e-9
    merged_calc[[z_col]] <- abs((merged_calc[[auc_col]] - mu) / sd)
    z_cols <- c(z_cols, z_col)
  }
  
  merged_calc$sum_z2 <- rowSums(merged_calc[z_cols]^2)
  final_df <- merged_calc %>% arrange(sum_z2, run_id)
  
  # Select Run
  top_tbl <- slice_head(final_df, n = top_k)
  if (pick == "first") selected <- top_tbl[1, , drop = FALSE]
  else if (pick == "second") selected <- top_tbl[2, , drop = FALSE]
  else selected <- top_tbl[pick_k_index, , drop = FALSE]
  
  if (is.null(out_dir)) out_dir <- dirname(normalizePath(input_dir))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  readr::write_csv(top_tbl,  file.path(out_dir, "best_run_topk.csv"))
  readr::write_csv(selected, file.path(out_dir, "best_run_selected.csv"))
  
  # Plot & Copy Files
  run_id_selected <- selected$run_id[1]
  
  for (m in valid_models) {
    # Plot
    plot_model_dist(
      data_list[[m]], m, "Single Dataset", stats[[m]]$mean, stats[[m]]$sd, 
      selected[[paste0("auc_", m)]], file.path(out_dir, paste0("auc_dist_", m, ".png"))
    )
    # Copy Files
    copy_associated_files(input_dir, m, run_id_selected, out_dir, prefix_name = paste0("Single_", m))
  }
  
  message(sprintf("[Single] Analysis Done. Selected Run ID: %d", run_id_selected))
}

# ==============================================================================
# SECTION 3: PAIRED / BATCH ANALYSIS (From V2)
# ==============================================================================

# --- 3.1 Core Function to Process One Pair ---
analyze_pair_core <- function(dir1, dir2, folder_name, display_name, label1 = "4_genes", label2 = "8_genes", models = c("LOGIT"), out_dir) {
  
  # Load Data helper
  load_d <- function(dpath) {
    res <- list()
    for (m in models) {
      tmp <- read_summary(file.path(dpath, m, "summary_runs.csv"))
      if (!is.null(tmp)) res[[m]] <- tmp
    }
    res
  }
  
  data1 <- load_d(dir1)
  data2 <- load_d(dir2)
  valid_models <- intersect(names(data1), names(data2))
  
  if (length(valid_models) == 0) return(NULL) 
  
  # Find Common IDs
  ids1 <- Reduce(intersect, lapply(data1[valid_models], `[[`, "run_id"))
  ids2 <- Reduce(intersect, lapply(data2[valid_models], `[[`, "run_id"))
  common_ids <- intersect(ids1, ids2)
  
  if (length(common_ids) == 0) return(NULL)
  
  pair_out_dir <- file.path(out_dir, folder_name)
  dir.create(pair_out_dir, recursive = TRUE, showWarnings = FALSE)
  
  stats_list <- list()
  
  for (m in valid_models) {
    d1 <- data1[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
    d2 <- data2[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
    
    # Wilcoxon Signed Rank Test (Paired)
    w_test <- wilcox.test(d1$auc_mean, d2$auc_mean, alternative = "two.sided")
    
    # Store Stats
    stats_list[[m]] <- data.frame(
      Folder_ID = folder_name,
      Display_Title = display_name,
      Model = m,
      Mean_1 = mean(d1$auc_mean), SD_1 = sd(d1$auc_mean),
      Mean_2 = mean(d2$auc_mean), SD_2 = sd(d2$auc_mean),
      P_Value = w_test$p.value
    )
    
    # Boxplot
    plot_df <- bind_rows(d1 %>% mutate(Group = label1), d2 %>% mutate(Group = label2))
    p_val_str <- format(w_test$p.value, scientific = FALSE, digits = 5)
    
    p_comp <- ggplot(plot_df, aes(x = Group, y = auc_mean, fill = Group)) +
      geom_boxplot(alpha = 0.6) + geom_jitter(width = 0.2, alpha = 0.3) +
      labs(title = paste0(display_name, " | ", m), 
           subtitle = paste0("Paired Wilcoxon, p = ", p_val_str)) +
      theme_minimal() + theme(legend.position = "none")
    ggsave(file.path(pair_out_dir, paste0("Boxplot_", m, ".png")), p_comp, width = 5, height = 5)
    
    # Find Best Run (Representative) based on minimum combined deviation
    z_score_comb <- (abs(scale(d1$auc_mean))^2 + abs(scale(d2$auc_mean))^2)
    best_idx <- which.min(z_score_comb)
    best_run_id <- d1$run_id[best_idx]
    
    # Draw Distribution
    plot_model_dist(d1, m, paste0(display_name, " (", label1, ")"), 
                    mean(d1$auc_mean), sd(d1$auc_mean), d1$auc_mean[best_idx], 
                    file.path(pair_out_dir, paste0("Dist_", label1, "_", m, ".png")))
    
    plot_model_dist(d2, m, paste0(display_name, " (", label2, ")"), 
                    mean(d2$auc_mean), sd(d2$auc_mean), d2$auc_mean[best_idx], 
                    file.path(pair_out_dir, paste0("Dist_", label2, "_", m, ".png")))
    
    # Copy ROC/VarImp Files (Feature from V1 brought to V2)
    copy_associated_files(dir1, m, best_run_id, pair_out_dir, prefix_name = paste0(label1))
    copy_associated_files(dir2, m, best_run_id, pair_out_dir, prefix_name = paste0(label2))
  }
  
  return(bind_rows(stats_list))
}

# --- 3.2 Batch Processing Function ---
run_batch_comparison <- function(
    root_dir_4genes, 
    root_dir_8genes, 
    output_root,
    name_mapping = NULL, 
    models = c("LOGIT")
) {
  
  message("========== BẮT ĐẦU QUÉT VÀ GHÉP CẶP THƯ MỤC ==========")
  
  subs_4 <- list.dirs(root_dir_4genes, full.names = FALSE, recursive = FALSE)
  subs_8 <- list.dirs(root_dir_8genes, full.names = FALSE, recursive = FALSE)
  
  # Matching logic
  df_4 <- tibble(actual_name_4 = subs_4) %>% mutate(base_name = str_remove(actual_name_4, "_4_genes$"))
  df_8 <- tibble(actual_name_8 = subs_8) %>% mutate(base_name = str_remove(actual_name_8, "_8_genes$"))
  
  pairs_df <- inner_join(df_4, df_8, by = "base_name")
  
  if (nrow(pairs_df) == 0) stop("Không tìm thấy cặp folder khớp nhau!")
  message(sprintf("Tìm thấy %d cặp folder khớp nhau.", nrow(pairs_df)))
  
  all_stats_results <- list()
  
  for (i in 1:nrow(pairs_df)) {
    base_name <- pairs_df$base_name[i]
    folder_4  <- pairs_df$actual_name_4[i]
    folder_8  <- pairs_df$actual_name_8[i]
    
    # Apply Mapping
    if (!is.null(name_mapping) && base_name %in% names(name_mapping)) {
      display_title <- name_mapping[[base_name]]
    } else {
      display_title <- base_name 
    }
    
    message(sprintf(">>> Processing: %s (Title: %s)", base_name, display_title))
    
    tryCatch({
      res_df <- analyze_pair_core(
        dir1 = file.path(root_dir_4genes, folder_4), 
        dir2 = file.path(root_dir_8genes, folder_8), 
        folder_name = base_name,
        display_name = display_title, 
        label1 = "4_genes", 
        label2 = "8_genes",
        models = models,
        out_dir = output_root
      )
      
      if (!is.null(res_df)) all_stats_results[[base_name]] <- res_df
      
    }, error = function(e) {
      message(sprintf("   [ERROR] %s: %s", base_name, e$message))
    })
  }
  
  # Summarize CSV
  if (length(all_stats_results) > 0) {
    final_table <- bind_rows(all_stats_results)
    
    formatted_table <- final_table %>%
      mutate(
        `4 genes (Mean ± SD)` = sprintf("%.4f ± %.4f", Mean_1, SD_1),
        `8 genes (Mean ± SD)` = sprintf("%.4f ± %.4f", Mean_2, SD_2),
        `P-value` = format(P_Value, scientific = FALSE, trim = TRUE), 
        Significant = ifelse(P_Value < 0.05, "YES", "NO")
      ) %>%
      select(
        `Comparison Title` = Display_Title,
        Model,
        `4 genes (Mean ± SD)`,
        `8 genes (Mean ± SD)`,
        `P-value`,
        Significant
      )
    
    out_csv <- file.path(output_root, "MASTER_STATISTICS_COMPARISON.csv")
    readr::write_csv(formatted_table, out_csv)
    message(sprintf("\nSUCCESS! Saved Summary CSV: %s", normalizePath(out_csv)))
  } else {
    warning("Không có kết quả nào được tạo ra.")
  }
}

# ==============================================================================
# SECTION 4: EXAMPLE USAGE
# ==============================================================================

# --- OPTION A: BATCH COMPARISON (4 genes vs 8 genes) ---
# Uncomment and edit below to run:

# mapping_names <- c(
#   "results_IBD_MA_actUC_UCrem_GSE186507" = "GSE186507: Active UC vs UC Remission",
#   "results_IBD_MA_UC_HC_GSE186507"       = "GSE186507: UC vs HC"
# )

# run_batch_comparison(
#   root_dir_4genes = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/4_genes",
#   root_dir_8genes = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/8_genes",
#   output_root     = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507_COMPARISON_SUMMARY",
#   name_mapping    = mapping_names,
#   models          = c("LOGIT")
# )

# --- OPTION B: SINGLE FOLDER ANALYSIS ---
# Uncomment and edit below to run:

analyze_auc_single(
  input_dir = "/mnt/d/Labs/Pharmaco-Omics/stabl/results_HCC_Recurrence_32_genes",
  models    = c("LOGIT", "RF", "PLSDA", "SVM"),
  out_dir   = "/mnt/d/Labs/Pharmaco-Omics/stabl/results_HCC_Recurrence_32_genes_SUMMARY"
)