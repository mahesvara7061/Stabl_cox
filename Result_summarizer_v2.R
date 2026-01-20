suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(tibble); library(stringr)
})

# ==============================================================================
# 1. HELPER FUNCTIONS (Đọc file & Vẽ hình)
# ==============================================================================

read_summary <- function(path) {
  if (!file.exists(path)) return(NULL)
  dat <- readr::read_csv(path, show_col_types = FALSE)
  if (!all(c("run_id", "auc_mean") %in% names(dat))) return(NULL)
  
  dat <- dat %>%
    transmute(run_id = as.integer(run_id), auc_mean = suppressWarnings(as.numeric(auc_mean))) %>%
    filter(is.finite(auc_mean))
  
  # Xử lý trùng lặp run_id
  if (nrow(count(dat, run_id) %>% filter(n > 1)) > 0) {
    dat <- dat %>% group_by(run_id) %>% summarise(auc_mean = mean(auc_mean), .groups = "drop")
  }
  arrange(dat, run_id)
}

plot_model_dist <- function(df, model_name, dataset_name, mu, sd, x_run, outfile) {
  if (!dir.exists(dirname(outfile))) dir.create(dirname(outfile), recursive = TRUE, showWarnings = FALSE)
  
  p <- ggplot(df, aes(x = auc_mean)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.5, fill = "grey70") +
    geom_density(aes(color = "Density"), linewidth = 1) +
    stat_function(fun = dnorm, args = list(mean = mu, sd = sd), aes(color = "Normal Fit"), linewidth = 1) +
    geom_vline(aes(xintercept = mu), linetype = "dotted") +
    labs(title = paste0(dataset_name, " | ", model_name),
         subtitle = sprintf("Mean=%.4f | SD=%.4f", mu, sd)) +
    theme_minimal()
  
  if (!is.na(x_run)) p <- p + geom_vline(aes(xintercept = x_run), color = "red")
  ggsave(outfile, plot = p, width = 6, height = 4, dpi = 150)
}

# ==============================================================================
# 2. CORE FUNCTION (Xử lý 1 cặp)
# ==============================================================================
analyze_pair_core <- function(dir1, dir2, folder_name, display_name, label1 = "4_genes", label2 = "8_genes", models = c("LOGIT"), out_dir) {
  
  # Load Data
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
    
    # Wilcoxon Signed Rank Test (Paired) vì chung run_id
    w_test <- wilcox.test(d1$auc_mean, d2$auc_mean, alternative = "two.sided")
    
    # Lưu dữ liệu thô trước, sau này format sau
    stats_list[[m]] <- data.frame(
      Folder_ID = folder_name,
      Display_Title = display_name, # Cột này chứa tên đẹp do user nhập
      Model = m,
      Mean_1 = mean(d1$auc_mean), SD_1 = sd(d1$auc_mean),
      Mean_2 = mean(d2$auc_mean), SD_2 = sd(d2$auc_mean),
      P_Value = w_test$p.value
    )
    
    # Vẽ Boxplot với Tiêu đề đẹp (Display Name)
    plot_df <- bind_rows(d1 %>% mutate(Group = label1), d2 %>% mutate(Group = label2))
    
    # Format p-value cho title biểu đồ (decimal)
    p_val_str <- format(w_test$p.value, scientific = FALSE, digits = 5)
    
    p_comp <- ggplot(plot_df, aes(x = Group, y = auc_mean, fill = Group)) +
      geom_boxplot(alpha = 0.6) + geom_jitter(width = 0.2, alpha = 0.3) +
      labs(title = paste0(display_name, " | ", m), 
           subtitle = paste0("Paired Wilcoxon, p = ", p_val_str)) +
      theme_minimal() + theme(legend.position = "none")
    ggsave(file.path(pair_out_dir, paste0("Boxplot_", m, ".png")), p_comp, width = 5, height = 5)
    
    # Vẽ Distribution (Best Run Selection)
    z_score <- (abs(scale(d1$auc_mean))^2 + abs(scale(d2$auc_mean))^2)
    best_idx <- which.min(z_score)
    
    # Lưu ảnh dist (Vẫn dùng tên file system để tránh lỗi ký tự đặc biệt)
    plot_model_dist(d1, m, paste0(display_name, " (", label1, ")"), mean(d1$auc_mean), sd(d1$auc_mean), d1$auc_mean[best_idx], file.path(pair_out_dir, paste0("Dist_", label1, "_", m, ".png")))
    plot_model_dist(d2, m, paste0(display_name, " (", label2, ")"), mean(d2$auc_mean), sd(d2$auc_mean), d2$auc_mean[best_idx], file.path(pair_out_dir, paste0("Dist_", label2, "_", m, ".png")))
  }
  
  return(bind_rows(stats_list))
}

# ==============================================================================
# 3. BATCH FUNCTION (ĐÃ CẬP NHẬT: NAME MAPPING & CSV FORMATTING)
# ==============================================================================
run_batch_comparison <- function(
    root_dir_4genes, 
    root_dir_8genes, 
    output_root,
    name_mapping = NULL, # Tham số mới: List map tên folder -> tên hiển thị
    models = c("LOGIT")
) {
  
  message("========== BẮT ĐẦU QUÉT VÀ GHÉP CẶP THƯ MỤC ==========")
  
  # 1. Lấy danh sách folder
  subs_4 <- list.dirs(root_dir_4genes, full.names = FALSE, recursive = FALSE)
  subs_8 <- list.dirs(root_dir_8genes, full.names = FALSE, recursive = FALSE)
  
  # 2. Map cặp dựa trên Base Name
  df_4 <- tibble(actual_name_4 = subs_4) %>% mutate(base_name = str_remove(actual_name_4, "_4_genes$"))
  df_8 <- tibble(actual_name_8 = subs_8) %>% mutate(base_name = str_remove(actual_name_8, "_8_genes$"))
  
  pairs_df <- inner_join(df_4, df_8, by = "base_name")
  
  if (nrow(pairs_df) == 0) stop("Không tìm thấy cặp folder khớp nhau!")
  
  message(sprintf("Tìm thấy %d cặp folder khớp nhau.", nrow(pairs_df)))
  all_stats_results <- list()
  
  # 3. Loop xử lý
  for (i in 1:nrow(pairs_df)) {
    base_name <- pairs_df$base_name[i]
    folder_4  <- pairs_df$actual_name_4[i]
    folder_8  <- pairs_df$actual_name_8[i]
    
    # --- CHECK NAME MAPPING ---
    # Nếu base_name có trong danh sách mapping thì dùng tên đẹp, ngược lại dùng base_name gốc
    if (!is.null(name_mapping) && base_name %in% names(name_mapping)) {
      display_title <- name_mapping[[base_name]]
    } else {
      display_title <- base_name 
    }
    
    message(sprintf("\n>>> Đang xử lý: %s (Title: %s)", base_name, display_title))
    
    tryCatch({
      res_df <- analyze_pair_core(
        dir1 = file.path(root_dir_4genes, folder_4), 
        dir2 = file.path(root_dir_8genes, folder_8), 
        folder_name = base_name,
        display_name = display_title, # Truyền tên đẹp vào hàm core
        label1 = "4_genes", 
        label2 = "8_genes",
        models = models,
        out_dir = output_root
      )
      
      if (!is.null(res_df)) all_stats_results[[base_name]] <- res_df
      
    }, error = function(e) {
      message(sprintf("   [ERROR] Lỗi tại %s: %s", base_name, e$message))
    })
  }
  
  # 4. TỔNG HỢP VÀ FORMAT CSV (PHẦN QUAN TRỌNG)
  message("\n========== TỔNG HỢP KẾT QUẢ ==========")
  
  if (length(all_stats_results) > 0) {
    final_table <- bind_rows(all_stats_results)
    
    # --- FORMATTING THE TABLE ---
    formatted_table <- final_table %>%
      mutate(
        # 1. Tạo cột 4 genes (Mean ± SD)
        `4 genes (Mean ± SD)` = sprintf("%.4f ± %.4f", Mean_1, SD_1),
        
        # 2. Tạo cột 8 genes (Mean ± SD)
        `8 genes (Mean ± SD)` = sprintf("%.4f ± %.4f", Mean_2, SD_2),
        
        # 3. Format P-value: Decimal, không khoa học (scientific=FALSE)
        # trim=TRUE để bỏ khoảng trắng thừa, digits=20 để đảm bảo hiện đủ số nhỏ
        `P-value` = format(P_Value, scientific = FALSE, trim = TRUE), 
        
        # 4. Cột Significant
        Significant = ifelse(P_Value < 0.05, "YES", "NO")
      ) %>%
      # Chọn và sắp xếp cột cho đẹp
      select(
        `Comparison Title` = Display_Title,
        Model,
        `4 genes (Mean ± SD)`,
        `8 genes (Mean ± SD)`,
        `P-value`,
        Significant
        # Có thể giữ lại các cột gốc nếu cần debug: Folder_ID, etc.
      )
    
    out_csv <- file.path(output_root, "MASTER_STATISTICS_COMPARISON.csv")
    readr::write_csv(formatted_table, out_csv)
    
    message(sprintf("SUCCESS! File CSV chuẩn format đã lưu tại:\n%s", normalizePath(out_csv)))
    print(head(formatted_table))
    
  } else {
    warning("Không có kết quả nào được tạo ra.")
  }
}

# ==============================================================================
# 4. CÁCH SỬ DỤNG VỚI DANH SÁCH TÊN (EXAMPLE USAGE)
# ==============================================================================

# --- A. KHAI BÁO DANH SÁCH TÊN HIỂN THỊ (MAPPING) ---
# Cấu trúc: "Tên_Folder_Gốc_Đã_Bỏ_Hậu_Tố" = "Tiêu Đề Bạn Muốn Hiển Thị"
my_mapping_list <- c(
  "results_IBD_MA_actUC_UCrem_GSE186507" = "GSE186507: Active UC vs UC Remission",
  "results_IBD_MA_actCD_CDrem_GSE186507" = "GSE186507: Active CD vs CD Remission",
  "results_IBD_MA_CD_HC_GSE186507"       = "GSE186507: CD vs HC",
  "results_IBD_MA_UC_HC_GSE186507"       = "GSE186507: UC vs HC",
  "results_IBD_MA_UC_CD_GSE186507"       = "GSE186507: UC vs CD",
  "results_IBD_MA_IBD_HC_GSE186507"      = "GSE186507: UC/CD vs HC",
  "results_IBD_MA_act_rem_GSE186507" = "GSE186507: Active vs Remission"

  # Thêm các cặp khác vào đây...
)

# --- B. ĐƯỜNG DẪN FOLDER ---
root_4g <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/4_genes" 
root_8g <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/8_genes"
output_folder <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507_COMPARISON_SUMMARY"

# --- C. CHẠY LỆNH ---
run_batch_comparison(
  root_dir_4genes = root_4g,
  root_dir_8genes = root_8g,
  output_root = output_folder,
  name_mapping = my_mapping_list,  # <--- Truyền list tên vào đây
  models = c("LOGIT") 
)