# # suppressPackageStartupMessages({
# #   library(readr); library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(tibble); library(stringr)
# # })

# # #--- helper: đọc 1 summary_runs.csv (run_id, auc_mean) ---
# # read_summary <- function(path) {
# #   if (!file.exists(path)) {
# #     warning(sprintf("File not found: %s", path))
# #     return(NULL)
# #   }
# #   dat <- readr::read_csv(path, show_col_types = FALSE)
# #   need <- c("run_id", "auc_mean")
# #   if (!all(need %in% names(dat))) {
# #     warning(sprintf("File %s missing columns run_id or auc_mean", basename(path)))
# #     return(NULL)
# #   }
  
# #   dat <- dat %>%
# #     transmute(
# #       run_id   = as.integer(run_id),
# #       auc_mean = suppressWarnings(as.numeric(auc_mean))
# #     ) %>%
# #     filter(is.finite(auc_mean))
  
# #   # phát hiện trùng run_id -> gộp trung bình
# #   dups <- dat %>% count(run_id) %>% filter(n > 1)
# #   if (nrow(dups) > 0) {
# #     message(sprintf("[WARN] %s có %d run_id bị trùng -> gộp trung bình.",
# #                     basename(path), nrow(dups)))
# #     dat <- dat %>%
# #       group_by(run_id) %>%
# #       summarise(auc_mean = mean(auc_mean), .groups = "drop")
# #   } else {
# #     dat <- arrange(dat, run_id)
# #   }
# #   dat
# # }

# # #--- helper: vẽ phân phối AUC + normal fit ---
# # plot_model_dist <- function(df, model_name, mu, sd, x_run, outfile,
# #                             show_selected = TRUE, bins = 30, base_size = 13) {
# #   # Kiểm tra tính hợp lệ của mu, sd
# #   if (is.na(mu) || !is.finite(mu)) mu <- mean(df$auc_mean, na.rm = TRUE)
# #   if (is.na(sd) || !is.finite(sd) || sd <= 0) sd <- 1e-9
  
# #   outdir <- dirname(outfile)
# #   if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
# #   p <- ggplot(df, aes(x = auc_mean)) +
# #     geom_histogram(aes(y = after_stat(density)), bins = bins, alpha = 0.5, fill = "grey70") +
# #     geom_density(aes(color = "Kernel Density"), linewidth = 1, key_glyph = "path") +
# #     stat_function(fun = dnorm,
# #                   args = list(mean = mu, sd = sd),
# #                   aes(color = "Normal Fit"),
# #                   linewidth = 1, key_glyph = "path") +
# #     geom_vline(aes(xintercept = mu, linetype = "Mean"),
# #                color = "black", linewidth = 0.6) +
# #     labs(
# #       title    = paste0(model_name, " | Distribution"),
# #       subtitle = sprintf("Mean = %.4f | SD = %.4f | Selected AUC = %.4f", mu, sd, x_run),
# #       x = "AUC Mean", y = "Density", color = "Curve", linetype = "Reference"
# #     ) +
# #     scale_color_manual(values = c("Kernel Density" = "#1f77b4", "Normal Fit" = "#2ca02c")) +
# #     scale_linetype_manual(values = c("Mean" = "dotted", "Selected" = "solid")) +
# #     guides(color = guide_legend(override.aes = list(fill = NA, linewidth = 1.2))) +
# #     theme_minimal(base_size = base_size)
  
# #   if (isTRUE(show_selected)) {
# #     p <- p + geom_vline(aes(xintercept = x_run, linetype = "Selected"),
# #                         color = "red", linewidth = 0.6)
# #   }
  
# #   ggsave(outfile, plot = p, width = 7, height = 5, dpi = 300)
# #   message(sprintf("[%s] Saved plot: %s", model_name, normalizePath(outfile)))
# #   p
# # }

# # #--- hàm chính: Phân tích 1 cohort (1 thư mục) với nhiều models ---
# # analyze_auc_single <- function(
# #     input_dir,
# #     models = c("RF", "PLSDA", "SVM", "LOGIT"), # Đã thêm LOGIT
# #     top_k = 10,                        
# #     pick  = c("first", "second", "kth"), 
# #     pick_k_index = 2,                 
# #     out_dir = NULL                    
# # ) {
# #   pick <- match.arg(pick)
  
# #   # 1. Tạo đường dẫn và đọc dữ liệu
# #   # Cấu trúc giả định: input_dir / <MODEL_NAME> / summary_runs.csv
# #   paths <- setNames(file.path(input_dir, models, "summary_runs.csv"), models)
  
# #   # Đọc dữ liệu vào list
# #   data_list <- lapply(names(paths), function(m) {
# #     d <- read_summary(paths[[m]])
# #     if (!is.null(d)) d$model <- m # gán nhãn model
# #     d
# #   })
# #   names(data_list) <- models
  
# #   # Loại bỏ các model không đọc được file (nếu có)
# #   valid_models <- models[!sapply(data_list, is.null)]
# #   if (length(valid_models) < length(models)) {
# #     warning("Một số model không tìm thấy file csv: ", paste(setdiff(models, valid_models), collapse=", "))
# #   }
# #   data_list <- data_list[valid_models]
# #   stopifnot(length(data_list) > 0)
  
# #   # 2. Tìm giao run_id (Common IDs) giữa tất cả các model có dữ liệu
# #   list_of_ids <- lapply(data_list, `[[`, "run_id")
# #   common_ids  <- Reduce(intersect, list_of_ids)
  
# #   if (length(common_ids) == 0) {
# #     stop("Không tìm thấy run_id chung nào giữa các model!")
# #   }
# #   message(sprintf("Số lượng run_id chung tìm thấy: %d", length(common_ids)))
  
# #   # Lọc data theo common_ids và sắp xếp
# #   for (m in valid_models) {
# #     data_list[[m]] <- data_list[[m]] %>% 
# #       filter(run_id %in% common_ids) %>% 
# #       arrange(run_id)
# #   }
  
# #   # 3. Tính thống kê (Mean, SD) cho từng model
# #   stats <- lapply(data_list, function(d) {
# #     mu <- mean(d$auc_mean)
# #     s  <- sd(d$auc_mean)
# #     if (is.na(s) || s == 0) s <- 1e-9
# #     list(mean = mu, sd = s)
# #   })
  
# #   # In thông tin thống kê
# #   msg <- paste(sapply(valid_models, function(m) {
# #     sprintf("%s=%.4f(%.4f)", m, stats[[m]]$mean, stats[[m]]$sd)
# #   }), collapse = " | ")
# #   message("[Stats] ", msg)
  
# #   # 4. Ghép bảng (Merge)
# #   # Tạo bảng rộng: run_id | auc_RF | auc_PLSDA | ...
# #   # merged <- data_list[[1]] %>% select(run_id, auc_mean) %>% rename(!!paste0("auc_", valid_models[1]) := auc_mean)
# #   merged <- data_list[[1]] %>% select(run_id, auc_mean)
# #   # FIX: Use base R to rename instead of !! := syntax to avoid errors
# #   names(merged)[names(merged) == "auc_mean"] <- paste0("auc_", valid_models[1])
# #   if (length(valid_models) > 1) {
# #     for (m in valid_models[-1]) {
# #       tmp <- data_list[[m]] %>% select(run_id, auc_mean)
# #       # Rename cột auc thành auc_<Model>
# #       col_name <- paste0("auc_", m)
# #       # tmp <- tmp %>% rename(!!col_name := auc_mean)
# #       # merged <- inner_join(merged, tmp, by = "run_id")
# #       names(tmp)[names(tmp) == "auc_mean"] <- col_name
# #       merged <- inner_join(merged, tmp, by = "run_id")
# #     }
# #   }
  
# #   # 5. Tính toán tiêu chí chọn (Z-score stability)
# #   # Với mỗi model, tính Z = |x - mean| / sd
# #   # Sau đó tính tổng Z bình phương
  
# #   merged_calc <- merged
# #   z_cols <- c()
# #   dev_cols <- c()
  
# #   for (m in valid_models) {
# #     auc_col <- paste0("auc_", m)
# #     z_col   <- paste0("z_", m)
    
# #     mu <- stats[[m]]$mean
# #     sd <- stats[[m]]$sd
    
# #     # Tính Z-score và Deviation
# #     merged_calc[[z_col]] <- abs((merged_calc[[auc_col]] - mu) / sd)
    
# #     z_cols <- c(z_cols, z_col)
# #     dev_cols <- c(dev_cols, abs(merged_calc[[auc_col]] - mu)) # absolute deviation
# #   }
  
# #   # Tính tổng hợp các chỉ số xếp hạng
# #   # sum_z2: Tổng bình phương Z-scores (càng nhỏ càng gần mean của mọi model)
# #   merged_calc$sum_z2 <- rowSums(merged_calc[z_cols]^2)
# #   merged_calc$max_z  <- apply(merged_calc[z_cols], 1, max)
# #   merged_calc$sum_abs_dev <- rowSums(merged_calc[, z_cols, drop=FALSE] * 0 + 1) # Dummy logic, replacing below
  
# #   # Recalculate sum_abs_dev manually to be safe
# #   total_dev <- 0
# #   for (m in valid_models) {
# #     auc_col <- paste0("auc_", m)
# #     total_dev <- total_dev + abs(merged_calc[[auc_col]] - stats[[m]]$mean)
# #   }
# #   merged_calc$sum_abs_dev <- total_dev
  
# #   # Sắp xếp: Ưu tiên run ổn định nhất (gần mean nhất)
# #   final_df <- merged_calc %>%
# #     arrange(sum_z2, max_z, sum_abs_dev, run_id)
  
# #   # 6. Chọn Top K và Pick
# #   top_tbl <- slice_head(final_df, n = top_k)
  
# #   if (pick == "first") {
# #     selected <- top_tbl[1, , drop = FALSE]
# #   } else if (pick == "second") {
# #     if(nrow(top_tbl) < 2) stop("Not enough rows for 'second'")
# #     selected <- top_tbl[2, , drop = FALSE]
# #   } else {
# #     # kth
# #     if (pick_k_index < 1 || pick_k_index > nrow(top_tbl)) stop("pick_k_index out of range")
# #     selected <- top_tbl[pick_k_index, , drop = FALSE]
# #   }
  
# #   # 7. Lưu kết quả
# #   if (is.null(out_dir)) {
# #     out_dir <- dirname(normalizePath(input_dir))
# #   }
# #   dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
# #   readr::write_csv(top_tbl,  file.path(out_dir, "best_run_topk.csv"))
# #   readr::write_csv(selected, file.path(out_dir, "best_run_selected.csv"))
# #   message(sprintf("Saved Selected Run Info: %s", normalizePath(file.path(out_dir, "best_run_selected.csv"))))
  
# #   # 8. Vẽ biểu đồ phân phối cho từng model
# #   plot_files <- list()
# #   for (m in valid_models) {
# #     auc_val <- selected[[paste0("auc_", m)]]
# #     fname   <- file.path(out_dir, paste0("auc_dist_", m, ".png"))
    
# #     plot_model_dist(
# #       df = data_list[[m]],
# #       model_name = paste0(m, " Model"),
# #       mu = stats[[m]]$mean,
# #       sd = stats[[m]]$sd,
# #       x_run = auc_val,
# #       outfile = fname
# #     )
# #     plot_files[[m]] <- fname
# #   }
  
# #   # 9. Gợi ý đường dẫn varImp
# #   run_str <- sprintf("%03d", selected$run_id[1])
# #   varimp_paths <- setNames(file.path(input_dir, valid_models, "varImp", paste0(run_str, ".csv")), valid_models)
  
# #   # 10. Copy ROC TIFF files từ LOGIT/roc vào thư mục summary
# #   roc_files <- list()
# #   for (m in valid_models) {
# #     roc_dir <- file.path(input_dir, m, "roc")
# #     if (dir.exists(roc_dir)) {
# #       # Tìm file TIFF có ID trùng với selected run
# #       tiff_pattern <- paste0(run_str, ".tiff?$")  # Hỗ trợ cả .tif và .tiff
# #       tiff_files <- list.files(roc_dir, pattern = tiff_pattern, full.names = TRUE, ignore.case = TRUE)
      
# #       if (length(tiff_files) > 0) {
# #         for (tiff_file in tiff_files) {
# #           # Copy file vào thư mục summary
# #           dest_file <- file.path(out_dir, paste0("roc_", m, "_", basename(tiff_file)))
# #           file.copy(tiff_file, dest_file, overwrite = TRUE)
# #           message(sprintf("[%s] Copied ROC plot: %s -> %s", m, basename(tiff_file), basename(dest_file)))
# #           roc_files[[m]] <- dest_file
# #         }
# #       } else {
# #         message(sprintf("[%s] Không tìm thấy file ROC TIFF với ID %s", m, run_str))
# #       }
# #     }
# #   }
  
# #   list(
# #     common_ids   = common_ids,
# #     top_table    = top_tbl,
# #     selected     = selected,
# #     varimp_paths = varimp_paths,
# #     roc_files    = roc_files,
# #     out_dir      = out_dir
# #   )
# # }

# # # res <- analyze_auc_single(
# # #   input_dir = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE177044/results_IBD_MA_UC_HC_GSE177044_4_genes", # Thư mục chứa các folder RF, PLSDA, SVM, LOGIT
# # #   models    = c("LOGIT"),
# # #   top_k     = 10,
# # #   pick      = "first",
# # #   out_dir   = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE177044_summary/results_IBD_MA_UC_HC_GSE177044_4_genes"
# # # )


# # # res <- analyze_auc_single(
# # #   input_dir = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/8_genes/results_IBD_MA_UC_HC_GSE186507_8_genes", # Thư mục chứa các folder RF, PLSDA, SVM, LOGIT
# # #   models    = c("LOGIT"),
# # #   top_k     = 10,
# # #   pick      = "first",
# # #   out_dir   = "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507_summary/results_IBD_MA_UC_HC_GSE186507_8_genes"
# # # )

# # #--- Chạy phân tích cho tất cả các folder trong 8_genes ---
# # base_input_dir <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE177044"
# # base_output_dir <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE177044_summary"

# # # Tạo thư mục output nếu chưa có
# # if (!dir.exists(base_output_dir)) {
# #   dir.create(base_output_dir, recursive = TRUE, showWarnings = FALSE)
# # }

# # # Lấy danh sách tất cả các folder con
# # subfolders <- list.dirs(base_input_dir, full.names = FALSE, recursive = FALSE)

# # # Lọc bỏ các folder trống hoặc không hợp lệ
# # subfolders <- subfolders[subfolders != ""]

# # message(sprintf("Tìm thấy %d folder(s) để xử lý:", length(subfolders)))
# # print(subfolders)

# # # Lưu kết quả vào list
# # results_list <- list()

# # # Loop qua từng folder
# # for (folder_name in subfolders) {
# #   message(sprintf("\n========== Đang xử lý: %s ==========", folder_name))
  
# #   input_path <- file.path(base_input_dir, folder_name)
# #   output_path <- file.path(base_output_dir, folder_name)
  
# #   tryCatch({
# #     res <- analyze_auc_single(
# #       input_dir = input_path,
# #       models    = c("LOGIT"),
# #       top_k     = 10,
# #       pick      = "first",
# #       out_dir   = output_path
# #     )
    
# #     results_list[[folder_name]] <- res
# #     message(sprintf("[SUCCESS] Hoàn thành xử lý: %s", folder_name))
    
# #   }, error = function(e) {
# #     message(sprintf("[ERROR] Lỗi khi xử lý %s: %s", folder_name, e$message))
# #     results_list[[folder_name]] <- list(error = e$message)
# #   })
# # }

# # message("\n========== HOÀN THÀNH TẤT CẢ ==========")
# # message(sprintf("Đã xử lý: %d/%d folder(s)", 
# #                 sum(sapply(results_list, function(x) is.null(x$error))), 
# #                 length(subfolders)))

# suppressPackageStartupMessages({
#   library(readr); library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(tibble); library(stringr)
# })

# # ==============================================================================
# # 1. HELPER FUNCTIONS
# # ==============================================================================

# #--- Đọc file summary_runs.csv ---
# read_summary <- function(path) {
#   if (!file.exists(path)) {
#     warning(sprintf("File not found: %s", path))
#     return(NULL)
#   }
#   dat <- readr::read_csv(path, show_col_types = FALSE)
#   need <- c("run_id", "auc_mean")
#   if (!all(need %in% names(dat))) {
#     warning(sprintf("File %s missing columns run_id or auc_mean", basename(path)))
#     return(NULL)
#   }
  
#   dat <- dat %>%
#     transmute(
#       run_id   = as.integer(run_id),
#       auc_mean = suppressWarnings(as.numeric(auc_mean))
#     ) %>%
#     filter(is.finite(auc_mean))
  
#   # Gộp trung bình nếu trùng run_id
#   dups <- dat %>% count(run_id) %>% filter(n > 1)
#   if (nrow(dups) > 0) {
#     # message(sprintf("[WARN] %s có %d run_id bị trùng -> gộp trung bình.", basename(path), nrow(dups)))
#     dat <- dat %>%
#       group_by(run_id) %>%
#       summarise(auc_mean = mean(auc_mean), .groups = "drop")
#   } else {
#     dat <- arrange(dat, run_id)
#   }
#   dat
# }

# #--- Vẽ phân phối AUC ---
# plot_model_dist <- function(df, model_name, dataset_name, mu, sd, x_run, outfile,
#                             show_selected = TRUE, bins = 30, base_size = 13) {
#   if (is.na(mu) || !is.finite(mu)) mu <- mean(df$auc_mean, na.rm = TRUE)
#   if (is.na(sd) || !is.finite(sd) || sd <= 0) sd <- 1e-9
  
#   outdir <- dirname(outfile)
#   if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
#   p <- ggplot(df, aes(x = auc_mean)) +
#     geom_histogram(aes(y = after_stat(density)), bins = bins, alpha = 0.5, fill = "grey70") +
#     geom_density(aes(color = "Kernel Density"), linewidth = 1, key_glyph = "path") +
#     stat_function(fun = dnorm, args = list(mean = mu, sd = sd),
#                   aes(color = "Normal Fit"), linewidth = 1, key_glyph = "path") +
#     geom_vline(aes(xintercept = mu, linetype = "Mean"), color = "black", linewidth = 0.6) +
#     labs(
#       title    = paste0(dataset_name, " | ", model_name),
#       subtitle = sprintf("Mean=%.4f | SD=%.4f | Selected=%.4f", mu, sd, x_run),
#       x = "AUC Mean", y = "Density", color = "Curve", linetype = "Reference"
#     ) +
#     scale_color_manual(values = c("Kernel Density" = "#1f77b4", "Normal Fit" = "#2ca02c")) +
#     scale_linetype_manual(values = c("Mean" = "dotted", "Selected" = "solid")) +
#     theme_minimal(base_size = base_size)
  
#   if (isTRUE(show_selected)) {
#     p <- p + geom_vline(aes(xintercept = x_run, linetype = "Selected"),
#                         color = "red", linewidth = 0.6)
#   }
  
#   ggsave(outfile, plot = p, width = 7, height = 5, dpi = 300)
#   p
# }

# # ==============================================================================
# # 2. SINGLE DATASET ANALYSIS (Hàm cũ của bạn)
# # ==============================================================================
# analyze_auc_single <- function(
#     input_dir, models = c("RF", "PLSDA", "SVM", "LOGIT"),
#     top_k = 10, pick  = c("first", "second", "kth"), pick_k_index = 2, out_dir = NULL
# ) {
#   pick <- match.arg(pick)
#   paths <- setNames(file.path(input_dir, models, "summary_runs.csv"), models)
#   data_list <- lapply(names(paths), function(m) {
#     d <- read_summary(paths[[m]])
#     if (!is.null(d)) d$model <- m
#     d
#   })
#   names(data_list) <- models
  
#   valid_models <- models[!sapply(data_list, is.null)]
#   if (length(valid_models) == 0) stop("No valid models found.")
#   data_list <- data_list[valid_models]
  
#   list_of_ids <- lapply(data_list, `[[`, "run_id")
#   common_ids  <- Reduce(intersect, list_of_ids)
#   if (length(common_ids) == 0) stop("No common run_ids found.")
  
#   for (m in valid_models) {
#     data_list[[m]] <- data_list[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
#   }
  
#   stats <- lapply(data_list, function(d) list(mean = mean(d$auc_mean), sd = sd(d$auc_mean)))
  
#   merged <- data_list[[valid_models[1]]] %>% select(run_id, auc_mean)
#   names(merged)[2] <- paste0("auc_", valid_models[1])
  
#   if (length(valid_models) > 1) {
#     for (m in valid_models[-1]) {
#       tmp <- data_list[[m]] %>% select(run_id, auc_mean)
#       names(tmp)[2] <- paste0("auc_", m)
#       merged <- inner_join(merged, tmp, by = "run_id")
#     }
#   }
  
#   merged_calc <- merged
#   z_cols <- c()
#   for (m in valid_models) {
#     auc_col <- paste0("auc_", m)
#     z_col   <- paste0("z_", m)
#     mu <- stats[[m]]$mean; sd <- stats[[m]]$sd
#     if(is.na(sd) || sd==0) sd <- 1e-9
#     merged_calc[[z_col]] <- abs((merged_calc[[auc_col]] - mu) / sd)
#     z_cols <- c(z_cols, z_col)
#   }
  
#   merged_calc$sum_z2 <- rowSums(merged_calc[z_cols]^2)
#   final_df <- merged_calc %>% arrange(sum_z2, run_id)
  
#   top_tbl <- slice_head(final_df, n = top_k)
#   if (pick == "first") selected <- top_tbl[1, , drop = FALSE]
#   else if (pick == "second") selected <- top_tbl[2, , drop = FALSE]
#   else selected <- top_tbl[pick_k_index, , drop = FALSE]
  
#   if (is.null(out_dir)) out_dir <- dirname(normalizePath(input_dir))
#   dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
#   readr::write_csv(top_tbl,  file.path(out_dir, "best_run_topk.csv"))
#   readr::write_csv(selected, file.path(out_dir, "best_run_selected.csv"))
  
#   # Copy files and Plot (Simplified for brevity as logic is same)
#   run_str <- sprintf("%03d", selected$run_id[1])
  
#   for (m in valid_models) {
#     plot_model_dist(data_list[[m]], m, "Single Dataset", stats[[m]]$mean, stats[[m]]$sd, 
#                     selected[[paste0("auc_", m)]], file.path(out_dir, paste0("auc_dist_", m, ".png")))
    
#     # Copy ROC
#     roc_dir <- file.path(input_dir, m, "roc")
#     tiff_files <- list.files(roc_dir, pattern = paste0(run_str, ".tiff?$"), full.names = TRUE)
#     if (length(tiff_files) > 0) file.copy(tiff_files, file.path(out_dir, paste0("roc_", m, "_", basename(tiff_files[1]))), overwrite = TRUE)
    
#     # Copy VarImp
#     vi_path <- file.path(input_dir, m, "varImp", paste0(run_str, ".csv"))
#     if(file.exists(vi_path)) file.copy(vi_path, file.path(out_dir, paste0("varImp_", m, ".csv")), overwrite = TRUE)
#   }
  
#   message(sprintf("[Single] Selected Run ID: %d", selected$run_id))
# }

# # ==============================================================================
# # 3. PAIRED DATASET ANALYSIS (Hàm Mới)
# # ==============================================================================
# analyze_auc_paired <- function(
#     dir1, dir2, 
#     name1 = "Dataset1", name2 = "Dataset2",
#     models = c("LOGIT"), 
#     out_dir = NULL
# ) {
  
#   message(sprintf("\n>>> START PAIRED ANALYSIS: %s vs %s", name1, name2))
  
#   # --- Step 1: Load Data for both datasets ---
#   load_dataset <- function(path, lbl) {
#     res <- list()
#     for (m in models) {
#       f <- file.path(path, m, "summary_runs.csv")
#       d <- read_summary(f)
#       if (!is.null(d)) {
#         d$model <- m
#         res[[m]] <- d
#       }
#     }
#     res
#   }
  
#   data1 <- load_dataset(dir1, name1)
#   data2 <- load_dataset(dir2, name2)
  
#   valid_models <- intersect(names(data1), names(data2))
#   if (length(valid_models) == 0) stop("Không có model chung hợp lệ giữa 2 dataset.")
  
#   message("Common Models: ", paste(valid_models, collapse = ", "))
  
#   # --- Step 2: Find Intersection of IDs across ALL models and BOTH datasets ---
#   ids1 <- Reduce(intersect, lapply(data1[valid_models], `[[`, "run_id"))
#   ids2 <- Reduce(intersect, lapply(data2[valid_models], `[[`, "run_id"))
#   common_ids <- intersect(ids1, ids2)
  
#   if (length(common_ids) == 0) stop("Không tìm thấy Run ID chung giữa 2 Dataset!")
#   message(sprintf("Found %d common Run IDs across both datasets.", length(common_ids)))
  
#   # --- Step 3: Calculate Z-scores combined ---
#   # Tạo bảng tổng hợp chứa RunID
#   final_df <- data.frame(run_id = common_ids)
#   total_z2 <- rep(0, length(common_ids))
  
#   stats_info <- list()
  
#   for (m in valid_models) {
#     # Dataset 1 stats
#     d1 <- data1[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
#     mu1 <- mean(d1$auc_mean); sd1 <- sd(d1$auc_mean); if(sd1==0) sd1 <- 1e-9
#     z1 <- abs((d1$auc_mean - mu1) / sd1)
    
#     # Dataset 2 stats
#     d2 <- data2[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
#     mu2 <- mean(d2$auc_mean); sd2 <- sd(d2$auc_mean); if(sd2==0) sd2 <- 1e-9
#     z2 <- abs((d2$auc_mean - mu2) / sd2)
    
#     # Accumulate Z-squared (Euclidean distance from mean in multi-dim space)
#     total_z2 <- total_z2 + (z1^2) + (z2^2)
    
#     # Save info for plotting later
#     stats_info[[m]] <- list(d1=d1, mu1=mu1, sd1=sd1, d2=d2, mu2=mu2, sd2=sd2)
    
#     # Add to final table for reference
#     final_df[[paste0("auc_", m, "_", name1)]] <- d1$auc_mean
#     final_df[[paste0("auc_", m, "_", name2)]] <- d2$auc_mean
#   }
  
#   final_df$total_z2_score <- total_z2
  
#   # Sort by smallest combined deviation
#   final_df <- final_df %>% arrange(total_z2_score)
#   selected_run <- final_df[1, ]
#   sel_id <- selected_run$run_id
  
#   message(sprintf(">>> BEST PAIRED RUN ID: %d (Score: %.4f)", sel_id, selected_run$total_z2_score))
  
#   # --- Step 4: Output & Copy Files ---
#   if (is.null(out_dir)) out_dir <- file.path(dirname(dir1), "Paired_Summary")
#   dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
#   # Save Table
#   readr::write_csv(head(final_df, 20), file.path(out_dir, "paired_best_runs.csv"))
  
#   run_str <- sprintf("%03d", sel_id)
  
#   for (m in valid_models) {
#     # 4.1 Plot Distributions
#     st <- stats_info[[m]]
    
#     # Plot Dataset 1
#     plot_model_dist(st$d1, m, name1, st$mu1, st$sd1, selected_run[[paste0("auc_", m, "_", name1)]],
#                     file.path(out_dir, paste0("Dist_", name1, "_", m, ".png")))
    
#     # Plot Dataset 2
#     plot_model_dist(st$d2, m, name2, st$mu2, st$sd2, selected_run[[paste0("auc_", m, "_", name2)]],
#                     file.path(out_dir, paste0("Dist_", name2, "_", m, ".png")))
    
#     # 4.2 Copy ROC & VarImp (Prefix with Dataset Name)
#     # Helper copy function
#     copy_files <- function(src_root, dataset_prefix) {
#       # ROC
#       roc_src <- file.path(src_root, m, "roc")
#       tifs <- list.files(roc_src, pattern = paste0(run_str, ".tiff?$"), full.names = TRUE)
#       if(length(tifs) > 0) {
#         dest <- file.path(out_dir, paste0(dataset_prefix, "_ROC_", m, ".tiff"))
#         file.copy(tifs[1], dest, overwrite = TRUE)
#       }
      
#       # VarImp
#       vi_src <- file.path(src_root, m, "varImp", paste0(run_str, ".csv"))
#       if(file.exists(vi_src)) {
#         dest <- file.path(out_dir, paste0(dataset_prefix, "_VarImp_", m, ".csv"))
#         file.copy(vi_src, dest, overwrite = TRUE)
#       }
#     }
    
#     copy_files(dir1, name1)
#     copy_files(dir2, name2)
#   }
  
#   message(sprintf("Results saved to: %s", normalizePath(out_dir)))
# }


# # ==============================================================================
# # 4. EXECUTION EXAMPLES
# # ==============================================================================

# # --- CẤU HÌNH ĐƯỜNG DẪN ---
# # Dataset 1
# path_GSE177044 <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/4_genes/results_IBD_MA_actUC_UCrem_GSE186507_4_genes"
# # Dataset 2 (Ví dụ bạn có folder tương ứng cho dataset kia)
# path_GSE186507 <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507/8_genes/results_IBD_MA_actUC_UCrem_GSE186507_8_genes"

# # Thư mục chứa kết quả chung
# paired_output <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/result_4genes_8genes_new/GSE186507_paired_summary/actUC_UCrem"

# # --- A. CHẠY RIÊNG LẺ (Như cũ) ---
# message("\n=== ANALYZING SINGLE DATASETS ===")
# # analyze_auc_single(path_GSE177044, models = c("LOGIT"), out_dir = paste0(path_GSE177044, "_summary"))
# # analyze_auc_single(path_GSE186507, models = c("LOGIT"), out_dir = paste0(path_GSE186507, "_summary"))


# # --- B. CHẠY PAIRED (TÌM RUN CHUNG CHO CẢ 2) ---
# # Chỉ chạy đoạn này nếu bạn muốn tìm run tốt cho CẢ HAI
# if (dir.exists(path_GSE177044) && dir.exists(path_GSE186507)) {
#   tryCatch({
#     analyze_auc_paired(
#       dir1 = path_GSE177044,
#       dir2 = path_GSE186507,
#       name1 = "GSE186507 | Active UC  vs UC Remission | 4_genes",  # Tên nhãn cho dataset 1
#       name2 = "GSE186507 | Active UC  vs UC Remission | 8_genes",  # Tên nhãn cho dataset 2
#       models = c("LOGIT"),  # Model cần phân tích
#       out_dir = paired_output
#     )
#   }, error = function(e) {
#     message("Lỗi Paired Analysis: ", e$message)
#   })
# } else {
#   message("Không tìm thấy đủ 2 thư mục input để chạy Paired Analysis.")
# }
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(tibble); library(stringr)
})

# ==============================================================================
# 1. HELPER FUNCTIONS (GIỮ NGUYÊN)
# ==============================================================================

read_summary <- function(path) {
  if (!file.exists(path)) {
    warning(sprintf("File not found: %s", path))
    return(NULL)
  }
  dat <- readr::read_csv(path, show_col_types = FALSE)
  need <- c("run_id", "auc_mean")
  if (!all(need %in% names(dat))) {
    warning(sprintf("File %s missing columns run_id or auc_mean", basename(path)))
    return(NULL)
  }
  
  dat <- dat %>%
    transmute(
      run_id   = as.integer(run_id),
      auc_mean = suppressWarnings(as.numeric(auc_mean))
    ) %>%
    filter(is.finite(auc_mean))
  
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
  
  if (isTRUE(show_selected)) {
    p <- p + geom_vline(aes(xintercept = x_run, linetype = "Selected"),
                        color = "red", linewidth = 0.6)
  }
  
  ggsave(outfile, plot = p, width = 7, height = 5, dpi = 300)
  p
}

# ==============================================================================
# 2. SINGLE DATASET ANALYSIS (GIỮ NGUYÊN)
# ==============================================================================
analyze_auc_single <- function(
    input_dir, models = c("RF", "PLSDA", "SVM", "LOGIT"),
    top_k = 10, pick  = c("first", "second", "kth"), pick_k_index = 2, out_dir = NULL
) {
  pick <- match.arg(pick)
  paths <- setNames(file.path(input_dir, models, "summary_runs.csv"), models)
  data_list <- lapply(names(paths), function(m) {
    d <- read_summary(paths[[m]])
    if (!is.null(d)) d$model <- m
    d
  })
  names(data_list) <- models
  
  valid_models <- models[!sapply(data_list, is.null)]
  if (length(valid_models) == 0) stop("No valid models found.")
  data_list <- data_list[valid_models]
  
  list_of_ids <- lapply(data_list, `[[`, "run_id")
  common_ids  <- Reduce(intersect, list_of_ids)
  if (length(common_ids) == 0) stop("No common run_ids found.")
  
  for (m in valid_models) {
    data_list[[m]] <- data_list[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
  }
  
  stats <- lapply(data_list, function(d) list(mean = mean(d$auc_mean), sd = sd(d$auc_mean)))
  
  merged <- data_list[[valid_models[1]]] %>% select(run_id, auc_mean)
  names(merged)[2] <- paste0("auc_", valid_models[1])
  
  if (length(valid_models) > 1) {
    for (m in valid_models[-1]) {
      tmp <- data_list[[m]] %>% select(run_id, auc_mean)
      names(tmp)[2] <- paste0("auc_", m)
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
  
  top_tbl <- slice_head(final_df, n = top_k)
  if (pick == "first") selected <- top_tbl[1, , drop = FALSE]
  else if (pick == "second") selected <- top_tbl[2, , drop = FALSE]
  else selected <- top_tbl[pick_k_index, , drop = FALSE]
  
  if (is.null(out_dir)) out_dir <- dirname(normalizePath(input_dir))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  readr::write_csv(top_tbl,  file.path(out_dir, "best_run_topk.csv"))
  readr::write_csv(selected, file.path(out_dir, "best_run_selected.csv"))
  
  run_str <- sprintf("%03d", selected$run_id[1])
  
  for (m in valid_models) {
    plot_model_dist(data_list[[m]], m, "Single Dataset", stats[[m]]$mean, stats[[m]]$sd, 
                    selected[[paste0("auc_", m)]], file.path(out_dir, paste0("auc_dist_", m, ".png")))
    
    roc_dir <- file.path(input_dir, m, "roc")
    tiff_files <- list.files(roc_dir, pattern = paste0(run_str, ".tiff?$"), full.names = TRUE)
    if (length(tiff_files) > 0) file.copy(tiff_files, file.path(out_dir, paste0("roc_", m, "_", basename(tiff_files[1]))), overwrite = TRUE)
    
    vi_path <- file.path(input_dir, m, "varImp", paste0(run_str, ".csv"))
    if(file.exists(vi_path)) file.copy(vi_path, file.path(out_dir, paste0("varImp_", m, ".csv")), overwrite = TRUE)
  }
  
  message(sprintf("[Single] Selected Run ID: %d", selected$run_id))
}

# ==============================================================================
# 3. PAIRED DATASET ANALYSIS (ĐÃ UPDATE WILCOXON RANK SUM)
# ==============================================================================
analyze_auc_paired <- function(
    dir1, dir2, 
    name1 = "Dataset1", name2 = "Dataset2",
    models = c("LOGIT"), 
    out_dir = NULL
) {
  
  message(sprintf("\n>>> START PAIRED ANALYSIS: %s vs %s", name1, name2))
  
  # --- Step 1: Load Data ---
  load_dataset <- function(path, lbl) {
    res <- list()
    for (m in models) {
      f <- file.path(path, m, "summary_runs.csv")
      d <- read_summary(f)
      if (!is.null(d)) {
        d$model <- m
        res[[m]] <- d
      }
    }
    res
  }
  
  data1 <- load_dataset(dir1, name1)
  data2 <- load_dataset(dir2, name2)
  
  valid_models <- intersect(names(data1), names(data2))
  if (length(valid_models) == 0) stop("Không có model chung hợp lệ giữa 2 dataset.")
  
  message("Common Models: ", paste(valid_models, collapse = ", "))
  
  # --- Step 2: Intersection of IDs ---
  ids1 <- Reduce(intersect, lapply(data1[valid_models], `[[`, "run_id"))
  ids2 <- Reduce(intersect, lapply(data2[valid_models], `[[`, "run_id"))
  common_ids <- intersect(ids1, ids2)
  
  if (length(common_ids) == 0) stop("Không tìm thấy Run ID chung giữa 2 Dataset!")
  message(sprintf("Found %d common Run IDs across both datasets.", length(common_ids)))
  
  # --- Step 3: Main Loop (Select Runs + Stats Test) ---
  final_df <- data.frame(run_id = common_ids)
  total_z2 <- rep(0, length(common_ids))
  
  stats_results <- list() # Lưu kết quả Wilcoxon
  
  if (is.null(out_dir)) out_dir <- file.path(dirname(dir1), "Paired_Summary")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  for (m in valid_models) {
    # 3.1 Filter Data
    d1 <- data1[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
    d2 <- data2[[m]] %>% filter(run_id %in% common_ids) %>% arrange(run_id)
    
    # 3.2 Calculate Basic Stats (Mean/SD)
    mu1 <- mean(d1$auc_mean); sd1 <- sd(d1$auc_mean); if(sd1==0) sd1 <- 1e-9
    mu2 <- mean(d2$auc_mean); sd2 <- sd(d2$auc_mean); if(sd2==0) sd2 <- 1e-9
    
    # 3.3 WILCOXON RANK SUM TEST (Two-sided)
    # So sánh xem phân phối AUC của d1 có khác d2 không
    w_test <- wilcox.test(d1$auc_mean, d2$auc_mean, alternative = "two.sided")
    
    # Lưu kết quả test
    stats_results[[m]] <- data.frame(
      Model = m,
      Group1 = name1, Mean1 = mu1, SD1 = sd1,
      Group2 = name2, Mean2 = mu2, SD2 = sd2,
      Test_Type = "Wilcoxon Rank Sum (Two-sided)",
      P_Value = w_test$p.value,
      W_Statistic = w_test$statistic
    )
    
    message(sprintf("[%s] Wilcoxon P-value: %.5f (%s)", 
                    m, w_test$p.value, 
                    ifelse(w_test$p.value < 0.05, "SIGNIFICANT", "ns")))
    
    # 3.4 Plot Comparison Boxplot
    plot_df <- bind_rows(
      d1 %>% mutate(Group = name1),
      d2 %>% mutate(Group = name2)
    )
    
    p_comp <- ggplot(plot_df, aes(x = Group, y = auc_mean, fill = Group)) +
      geom_boxplot(alpha = 0.6, outlier.shape = NA) +
      geom_jitter(width = 0.2, alpha = 0.4, size = 1) +
      labs(
        title = paste0("Comparison: ", m),
        subtitle = sprintf("Wilcoxon Rank Sum Test: p = %.2e", w_test$p.value),
        y = "AUC Mean", x = NULL
      ) +
      theme_minimal() +
      theme(legend.position = "none")
    
    ggsave(file.path(out_dir, paste0("Compare_Boxplot_", m, ".png")), p_comp, width = 6, height = 6)
    
    # 3.5 Calculate Z-scores for Run Selection
    z1 <- abs((d1$auc_mean - mu1) / sd1)
    z2 <- abs((d2$auc_mean - mu2) / sd2)
    total_z2 <- total_z2 + (z1^2) + (z2^2)
    
    # Save info for dist plots later
    final_df[[paste0("auc_", m, "_", name1)]] <- d1$auc_mean
    final_df[[paste0("auc_", m, "_", name2)]] <- d2$auc_mean
    
    # Vẽ distribution riêng lẻ
    plot_model_dist(d1, m, name1, mu1, sd1, NA, 
                    file.path(out_dir, paste0("Dist_", str_replace_all(name1, "[^A-Za-z0-9]", "_"), "_", m, ".png")), show_selected = FALSE)
    plot_model_dist(d2, m, name2, mu2, sd2, NA, 
                    file.path(out_dir, paste0("Dist_", str_replace_all(name2, "[^A-Za-z0-9]", "_"), "_", m, ".png")), show_selected = FALSE)
  }
  
  # --- Step 4: Save Statistical Results ---
  stats_df <- bind_rows(stats_results)
  readr::write_csv(stats_df, file.path(out_dir, "statistical_comparison.csv"))
  
  # --- Step 5: Select Best Common Run ---
  final_df$total_z2_score <- total_z2
  final_df <- final_df %>% arrange(total_z2_score)
  selected_run <- final_df[1, ]
  sel_id <- selected_run$run_id
  
  message(sprintf(">>> BEST PAIRED RUN ID: %d (Score: %.4f)", sel_id, selected_run$total_z2_score))
  readr::write_csv(head(final_df, 20), file.path(out_dir, "paired_best_runs.csv"))
  
  # --- Step 6: Copy Files for Selected Run ---
  run_str <- sprintf("%03d", sel_id)
  
  copy_files <- function(src_root, dataset_prefix, model_name) {
    clean_prefix <- str_replace_all(dataset_prefix, "[^A-Za-z0-9]", "_")
    # ROC
    roc_src <- file.path(src_root, model_name, "roc")
    tifs <- list.files(roc_src, pattern = paste0(run_str, ".tiff?$"), full.names = TRUE)
    if(length(tifs) > 0) {
      dest <- file.path(out_dir, paste0(clean_prefix, "_ROC_", model_name, ".tiff"))
      file.copy(tifs[1], dest, overwrite = TRUE)
    }
    # VarImp
    vi_src <- file.path(src_root, model_name, "varImp", paste0(run_str, ".csv"))
    if(file.exists(vi_src)) {
      dest <- file.path(out_dir, paste0(clean_prefix, "_VarImp_", model_name, ".csv"))
      file.copy(vi_src, dest, overwrite = TRUE)
    }
  }
  
  for (m in valid_models) {
    copy_files(dir1, name1, m)
    copy_files(dir2, name2, m)
  }
  
  message(sprintf("Results & Stats saved to: %s", normalizePath(out_dir)))
}


# ==============================================================================
# 4. EXECUTION
# ==============================================================================

# --- CẤU HÌNH ĐƯỜNG DẪN (VÍ DỤ) ---
# Dataset 1
path_4genes <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/results_IBD_MA_UC_HC_GSE186507_4_genes_test"
# Dataset 2
path_8genes <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/results_IBD_MA_UC_HC_GSE186507_8_genes_test"

# Thư mục chứa kết quả chung
paired_output <- "/mnt/d/Labs/Pharmaco-Omics/ibd meta-analysis/results_IBD_MA_UC_HC_GSE186507_paired_summary_4genes_8genes_test"

# --- CHẠY PAIRED ANALYSIS ---
if (dir.exists(path_4genes) && dir.exists(path_8genes)) {
  tryCatch({
    analyze_auc_paired(
      dir1 = path_4genes,
      dir2 = path_8genes,
      name1 = "CD vs HC | 4 genes",  # Tên ngắn gọn để hiện trên biểu đồ
      name2 = "CD vs HC | 8 genes",
      models = c("LOGIT"), 
      out_dir = paired_output
    )
  }, error = function(e) {
    message("Lỗi: ", e$message)
  })
} else {
  message("Không tìm thấy đủ thư mục input.")
}