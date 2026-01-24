# 1. Cài đặt và load thư viện cần thiết (nếu chưa có)
if(!require(ggplot2)) install.packages("ggplot2")

library(ggplot2)

# 2. Đọc dữ liệu từ file CSV
# Thay 'du_lieu_cua_ban.csv' bằng tên file thực tế của bạn
df <- read.csv("/mnt/d/Labs/Pharmaco-Omics/stabl/results_survival_46/individual_genes_analysis_verify_entire_dataset/prognosis_Biosignature_Transfer_MedianThr.csv", header = TRUE)

# Kiểm tra sơ bộ dữ liệu
head(df)

# 3. Chuẩn bị dữ liệu cho PCA
# Cấu trúc: Cột 1 là ID, Cột 2 là Label, Cột 3 trở đi là Features
# Chúng ta chỉ lấy dữ liệu số (từ cột 3 đến hết) để chạy PCA
pca_data <- df[, 3:ncol(df)]

# Đảm bảo tất cả các cột features đều là dạng số (numeric)
# Nếu có lỗi ở bước này, hãy kiểm tra xem file csv có lẫn ký tự lạ trong cột số liệu không
pca_data <- as.data.frame(lapply(pca_data, as.numeric))

# 4. Thực hiện PCA
# scale. = TRUE là BẮT BUỘC để chuẩn hóa dữ liệu (đưa về cùng đơn vị đo)
pca_result <- prcomp(pca_data, scale. = TRUE, center = TRUE)

# Xem tóm tắt kết quả (Variance explained)
summary(pca_result)

# 5. Trực quan hóa kết quả (Vẽ biểu đồ)

# --- CÁCH 1: Dùng ggplot2 cơ bản (Vẽ PC1 và PC2) ---
# Tạo dataframe chứa kết quả PCA và gắn lại cột Label
pca_plot_df <- data.frame(pca_result$x)
pca_plot_df$Label <- as.factor(df[, 2]) # Lấy cột 2 làm nhãn phân nhóm
pca_plot_df$SampleID <- df[, 1]         # Lấy cột 1 nếu muốn hiện tên mẫu

# Vẽ biểu đồ Scatter plot
p1 <- ggplot(pca_plot_df, aes(x = PC1, y = PC2, color = Label)) +
  geom_point(size = 3, alpha = 0.7) + # Vẽ điểm
  stat_ellipse(level = 0.95) +        # (Tùy chọn) Vẽ vòng elip bao quanh nhóm
  theme_minimal() +
  labs(title = "PCA Plot: PC1 vs PC2",
       x = paste0("PC1 (", round(summary(pca_result)$importance[2,1]*100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca_result)$importance[2,2]*100, 1), "%)"))

print(p1)

# --- TIẾP TỤC TỪ ĐOẠN CODE CỦA BẠN ---

# 6. Vẽ biểu đồ PCA 3D

# Cài đặt thư viện plotly nếu chưa có
if(!require(plotly)) install.packages("plotly")
library(plotly)

# Chuẩn bị dữ liệu cho 3D (Lấy thêm PC3)
pca_3d_df <- data.frame(pca_result$x[, 1:3]) # Lấy 3 PC đầu tiên
pca_3d_df$Label <- as.factor(df[, 2])        # Cột Label
pca_3d_df$SampleID <- df[, 1]                # Cột Sample ID

# Tính phần trăm phương sai để hiển thị lên trục (cho PC1, PC2, PC3)
var_explained <- round(summary(pca_result)$importance[2, 1:3] * 100, 1)

# Vẽ biểu đồ tương tác
p_3d <- plot_ly(data = pca_3d_df, 
                x = ~PC1, 
                y = ~PC2, 
                z = ~PC3, 
                color = ~Label, 
                colors = c('#636EFA', '#EF553B', '#00CC96'), # Tùy chỉnh màu nếu muốn
                text = ~paste("ID:", SampleID, "<br>Label:", Label), # Hiển thị thông tin khi rê chuột
                type = "scatter3d", 
                mode = "markers",
                marker = list(size = 5, opacity = 0.8)) %>%
  layout(
    title = "3D PCA Plot",
    scene = list(
      xaxis = list(title = paste0("PC1 (", var_explained[1], "%)")),
      yaxis = list(title = paste0("PC2 (", var_explained[2], "%)")),
      zaxis = list(title = paste0("PC3 (", var_explained[3], "%)"))
    )
  )

p_3d

# --- TIẾP TỤC TỪ PHẦN VẼ 3D (biến p_3d) ---

# 1. Cài đặt thư viện htmlwidgets (nếu chưa có)
if(!require(htmlwidgets)) install.packages("htmlwidgets")
library(htmlwidgets)

# 2. Định nghĩa tên file và đường dẫn lưu
# Bạn có thể lưu ngay tại thư mục làm việc hiện tại hoặc đường dẫn cụ thể
output_file <- "PCA_3D_Result.html" 

# Hoặc nếu muốn lưu vào cùng thư mục với file dữ liệu gốc của bạn:
# output_file <- "/mnt/d/Labs/Pharmaco-Omics/stabl/results_survival_46/individual_genes_analysis_verify_entire_dataset/PCA_3D_Result.html"

# 3. Thực hiện lưu file
# 'p_3d' là tên biến biểu đồ bạn đã tạo ở bước trước
saveWidget(p_3d, file = output_file, selfcontained = TRUE)

# Thông báo khi hoàn tất
message("Đã lưu file HTML tại: ", getwd(), "/", output_file)