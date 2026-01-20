# =============================================================================
# MACHINE LEARNING PIPELINE
# =============================================================================
# This script contains standardized ML functions for binary classification
# with nested cross-validation, ROC analysis, and variable importance
# =============================================================================

# =============================================================================
# RANDOM FOREST FUNCTION
# =============================================================================
rf_perform_std_Revised <- function(
    data_input_ML,
    classname = c("Survival","NonSurvival"),  # CHANGE: Your class labels (positive class first)
    seed = 1,                                 # CHANGE: Random seed for reproducibility
    split_ratio = 0.7,                        # CHANGE: Train/test split ratio (0.7 = 70% train)
    k_fold_outer = 5,                         # CHANGE: Number of outer CV folds
    k_fold_inner = 5,                         # CHANGE: Number of inner CV folds for tuning
    useModel_for_varImp = TRUE                # CHANGE: Use model-specific importance vs permutation
){
  

  # 1. LOAD REQUIRED LIBRARIES
  # Purpose: Load all necessary packages for Random Forest modeling
  require(caret)      # For model training and cross-validation
  library(ranger)     # Fast Random Forest implementation
  require(pROC)       # For ROC curve analysis
  library(tidyverse)  # Data manipulation
  require(magrittr)   # For extract2() function
  

  # 2. DATA PREPARATION
  # Purpose: Filter data to specified classes and convert to factor
  # CHANGE: Modify if you have different label column name (currently "Label")
  data_input_ML <- filter(data_input_ML, Label %in% classname)
  data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

  # 3. CREATE TRAIN/TEST SPLITS FOR OUTER CV
  # Purpose: Generate multiple train/test splits for nested cross-validation
  # Note: stratified sampling ensures balanced class distribution
  set.seed(seed)
  trainIndex <- caret::createDataPartition(data_input_ML$Label,
                                           p = split_ratio,
                                           list = FALSE,
                                           times = k_fold_outer)
  
  # Create list of training and test sets for each fold
  training <- test <- list()
  for (i in 1:ncol(trainIndex)){
    training[[i]] <- data_input_ML[trainIndex[,i],] 
    test[[i]] <- data_input_ML[-trainIndex[,i], ]
  }
  

  # 4. CONFIGURE INNER CROSS-VALIDATION
  # Purpose: Set up inner CV for hyperparameter tuning
  # CHANGE: Modify method or summaryFunction for different CV strategies
  trControl_man <- trainControl(
    method = "cv",                            # Cross-validation method
    number = k_fold_inner,                    # Number of inner folds
    summaryFunction = twoClassSummary,        # Binary classification metrics
    classProbs = TRUE                         # Required for ROC calculation
  )
  

  # 5. TRAIN RANDOM FOREST MODELS
  # Purpose: Train one RF model for each outer fold with inner CV tuning
  # CHANGE: Modify tuneLength for more/fewer hyperparameter combinations
  # CHANGE: Modify importance type ('impurity' or 'permutation')
  AUCROC_fit <- list()
  for (i in 1:length(training)){
    AUCROC_fit[[i]] <- caret::train(
      Label ~ .,                              # Predict Label using all other variables
      data = training[[i]],
      method = 'ranger',                      # Random Forest via ranger package
      metric = "ROC",                         # Optimize for ROC AUC
      trControl = trControl_man,
      importance = 'impurity',                # Variable importance method
      tuneLength = 10                         # Number of hyperparameter combinations to try
    )
  }
  

  # 6. EXTRACT VARIABLE IMPORTANCE
  # Purpose: Calculate and average variable importance across all folds
  # Note: Provides stable importance estimates through averaging
  varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
  
  # Extract importance from each fold
  for (i in 1:length(AUCROC_fit)){
    varImp_list[[i]] <- varImp(AUCROC_fit[[i]], useModel = useModel_for_varImp, 
                               nonpara = FALSE, scale = TRUE)
    varImp_intermediate[[i]] <- select(extract2(varImp_list[[i]],1),1)
    varImp_score_list[[i]] <- varImp_intermediate[[i]] %>% DBI::sqlRownamesToColumn()
  }
  
  # Average importance scores across all folds
  varImp_score_df <-
    reduce(varImp_score_list, full_join, by = 'row_names') %>% 
    column_to_rownames(var = "row_names") %>% 
    rowMeans() %>% 
    base::as.data.frame()
  
  colnames(varImp_score_df) <- "Importance_Score"
  

  # 7. VALIDATE ON TEST SETS
  # Purpose: Make predictions and generate ROC curves for each test fold
  AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- list()
  for (i in 1:length(AUCROC_fit)){
    AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]])
    AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]], type = "prob")
    AUCROC_ROC[[i]] <- roc(predictor = extract2(AUCROC_test_ROC[[i]], 2),
                           response = test[[i]]$Label)
  }
  

  # 8. GENERATE CONFUSION MATRICES
  # Purpose: Calculate classification metrics for each fold
  AUCROC_confusion_matrix = list()
  for (i in 1:length(AUCROC_fit)){
    AUCROC_confusion_matrix[[i]] = confusionMatrix(AUCROC_test[[i]], 
                                                   as.factor(use_series(test[[i]], "Label")))
  }
  

  # 9. CALCULATE AUC STATISTICS
  # Purpose: Extract AUC values and compute mean ± SD
  auc_tot <- rep(0, k_fold_outer)
  for (i in 1:length(AUCROC_ROC)){
    auc_tot[i] <- AUCROC_ROC[[i]]$auc
  }
  
  auc_mean <- mean(auc_tot)
  auc_sd <- sd(auc_tot)
  

  # 10. INTERPOLATE ROC CURVES
  # Purpose: Align ROC curves from different folds to common grid for averaging
  # CHANGE: Modify common_grid_points for smoother/coarser interpolation
  # Note: Different folds may have different numbers of thresholds
  common_grid_points <- 100  # Standard number of interpolation points
  
  # Initialize matrices for sensitivity and specificity
  roc_point_sen <- data.frame(matrix(nrow = common_grid_points,
                                     ncol = length(test)))
  roc_point_spe <- data.frame(matrix(nrow = common_grid_points,
                                     ncol = length(test)))
  
  # Interpolate each fold's ROC curve to common grid
  for (i in 1:length(AUCROC_ROC)){
    orig_sen <- AUCROC_ROC[[i]]$sensitivities
    orig_spe <- AUCROC_ROC[[i]]$specificities
    orig_x <- seq(0, 1, length.out = length(orig_sen))
    common_x <- seq(0, 1, length.out = common_grid_points)
    
    # Linear interpolation to align curves
    roc_point_sen[,i] <- approx(x = orig_x, y = orig_sen, 
                                xout = common_x, rule = 2)$y
    roc_point_spe[,i] <- approx(x = orig_x, y = orig_spe, 
                                xout = common_x, rule = 2)$y
  }
  
  # Calculate mean and SD for ROC curves
  roc_mean_sen <- apply(roc_point_sen, 1, mean)
  roc_mean_spe <- apply(roc_point_spe, 1, mean)
  roc_sd_sen <- apply(roc_point_sen, 1, sd)
  roc_sd_spe <- apply(roc_point_spe, 1, sd)
  
  # Calculate confidence bands (mean ± 1 SD)
  roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
  roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
  roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
  roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  

  # 11. PREPARE DATA FOR VISUALIZATION
  # Purpose: Create data frames for plotting ROC curves
  # CHANGE: Modify Comparision text to match your study
  # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")  # Description of classification task

  # If you want to customize Comparision, uncomment and modify the line below
  Comparision <- "Good Prognosis vs Poor Prognosis"
  
  # Mean ROC curve data
  avg_AUC <- data.frame(
    roc_mean_spe = rev(roc_mean_spe),
    roc_mean_sen = rev(roc_mean_sen)
  )
  
  # Confidence band polygon data
  polygon_SD_AUC <- data.frame(
    x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
    y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper))
  )
  

  # 12. CREATE ROC CURVE PLOT
  # Purpose: Visualize mean ROC curve with confidence bands
  # CHANGE: Modify colors, sizes, positions for different aesthetics
  ggplot() +
    geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
              color = "#366993", size = 1.18) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
                alpha = 1, color = "grey") +
    theme_Publication() +
    coord_equal() +
    geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), 
                 fill = "steelblue", alpha = 0.2) +
    labs(x = "False Positive Rate", y = "True Positive Rate", 
         title = Comparision) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    theme(
      axis.title = element_text(size = 19, face = "bold"),
      axis.text = element_text(size = 17),
      plot.title = element_text(size = 23, face = "bold")
    ) -> p_RocCurve
  
  # Add AUC annotation to plot
  AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
                        "±", formatC(auc_sd, format = "f", digits = 2))
  
  # CHANGE: Modify annotation positions (x, y) to suit your plot
  p_RocCurve = p_RocCurve +
    annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
    annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
    annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
             vjust = 0, color = "#366993", size = 1.18) +
    annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
    annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
             alpha = .3, fill = "steelblue")
  

  # 13. CREATE VARIABLE IMPORTANCE PLOT
  # Purpose: Visualize top 10% most important features
  # CHANGE: Modify percentage (10/100) to show more/fewer features
  # CHANGE: Modify colors and sizes for different aesthetics
  gsub("`", "", rownames(varImp_score_df)) -> rownames(varImp_score_df)
  p_VarImp_10 = varImp_score_df %>% 
    na.omit() %>%
    rownames_to_column() %>%
    arrange(Importance_Score) %>%
    top_n(n = nrow(varImp_score_df) * 10 / 100) %>%  # Top 10% of features
    mutate(rowname = factor(rowname, levels = rowname)) %>%
    ggplot(aes(x = rowname, y = Importance_Score)) +
    geom_segment(aes(xend = rowname, yend = 0)) +
    geom_point(size = 5, color = "orange") +
    coord_flip() +
    theme_Publication() +
    labs(x = "", y = "Importance Scores") +
    theme(
      axis.title = element_text(size = 15, face = "bold"),
      axis.text = element_text(size = 12.8)
    )
  

  # 14. CALCULATE PERFORMANCE METRICS
  # Purpose: Extract sensitivity, specificity, and balanced accuracy
  # Note: Metrics calculated for each fold then averaged
  confuse_table_lg = AUCROC_confusion_matrix
  
  Balanced_Accuracy_model_lg <- Sensitivity_model_lg <- Specificity_model_lg <- list()
  for (i in 1:length(confuse_table_lg)) {
    Sensitivity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Sensitivity"]]
    Specificity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Specificity"]]
    Balanced_Accuracy_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Balanced Accuracy"]]
  }
  
  # Calculate mean and SD for each metric
  Sensitivity_lg <- Sensitivity_model_lg %>% reduce(append)
  Sensitivity_lg_sd <- sd(Sensitivity_lg)
  Sensitivity_lg_mean <- mean(Sensitivity_lg)
  
  Specificity_lg <- Specificity_model_lg %>% reduce(append)
  Specificity_lg_sd <- sd(Specificity_lg)
  Specificity_lg_mean <- mean(Specificity_lg)
  
  Balanced_Accuracy_lg <- Balanced_Accuracy_model_lg %>% reduce(append)
  Balanced_Accuracy_lg_sd <- sd(Balanced_Accuracy_lg)
  Balanced_Accuracy_lg_mean <- mean(Balanced_Accuracy_lg)
  
  # Create summary table of metrics
  Metrics_Table <- data.frame(
    Sensitivity_Mean = Sensitivity_lg_mean,
    Sensitivity_SD = Sensitivity_lg_sd,
    Specificity_Mean = Specificity_lg_mean,
    Specificity_SD = Specificity_lg_sd,
    Balanced_Accuracy_Mean = Balanced_Accuracy_lg_mean,
    Balanced_Accuracy_SD = Balanced_Accuracy_lg_sd
  ) %>%
    setNames(c("Sensitivity, Mean", "Sensitivity, SD", 
               "Specificity, Mean", "Specificity, SD", 
               "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  

  # 15. RETURN ALL RESULTS
  # Purpose: Package all outputs for downstream analysis
  return(
    list(
      "auc_tot" = auc_tot,                                    # AUC for each fold
      "varImp_score_df" = varImp_score_df,                    # Variable importance
      "AUCROC_fit" = AUCROC_fit,                              # Trained models
      "test" = test,                                          # Test sets
      "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,    # Confusion matrices
      "Comparision" = Comparision,                            # Classification description
      "auc_mean" = auc_mean,                                  # Mean AUC
      "auc_sd" = auc_sd,                                      # SD of AUC
      "avg_AUC" = avg_AUC,                                    # Mean ROC curve data
      "polygon_SD_AUC" = polygon_SD_AUC,                      # Confidence band data
      "p_RocCurve" = p_RocCurve,                              # ROC curve plot
      "p_VarImp_10" = p_VarImp_10,                            # Variable importance plot
      "Metrics_Table" = Metrics_Table                         # Performance metrics table
    )
  )
}

# =============================================================================
# SUPPORT VECTOR MACHINE (LINEAR) FUNCTION
# =============================================================================
# svm_perform_std_Revised <- function(
#     data_input_ML,
#     classname = c("Survival","NonSurvival"),      # CHANGE: Your class labels
#     column_to_process,                            # CHANGE: Specify columns to scale/center
#     dataprocessing_method = c("center", "scale"), # CHANGE: Preprocessing methods
#     seed = 1,
#     split_ratio = 0.7,
#     k_fold_outer = 5,
#     k_fold_inner = 5,
#     useModel_for_varImp = TRUE 
# ){
  

#   # 1. LOAD REQUIRED LIBRARIES
#   # Purpose: Load packages for SVM modeling (uses kernlab via caret)
#   require(readr)
#   require(caret)
#   require(pROC)
#   require(S4Vectors)
#   require(dplyr)
#   require(magrittr)
#   require(ggplot2)
#   require(hrbrthemes)
#   require(tibble)
#   require(purrr)
  

#   # 2. DATA PREPARATION
#   # Purpose: Filter to specified classes and factorize labels
#   data_input_ML <- filter(data_input_ML, Label %in% classname)
#   data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

#   # 3. CREATE TRAIN/TEST SPLITS
#   # Purpose: Generate stratified train/test splits for outer CV
#   set.seed(seed)
#   trainIndex <- caret::createDataPartition(data_input_ML$Label,
#                                            p = split_ratio,
#                                            list = FALSE,
#                                            times = k_fold_outer)
  
#   training <- test <- list()
#   for (i in 1:ncol(trainIndex)){
#     training[[i]] <- data_input_ML[trainIndex[,i],] 
#     test[[i]] <- data_input_ML[-trainIndex[,i], ]
#   }
  

#   # 4. CONFIGURE INNER CV
#   # Purpose: Set up cross-validation for hyperparameter tuning
#   trControl_man <- trainControl(
#     method = "cv",
#     number = k_fold_inner,
#     summaryFunction = twoClassSummary,
#     classProbs = TRUE
#   )
  

#   # 5. PREPROCESS DATA (SCALING/CENTERING)
#   # Purpose: Apply centering and scaling to specified columns
#   # Note: SVM is sensitive to feature scales, so preprocessing is important
#   # CHANGE: Modify dataprocessing_method for different preprocessing
#   train_to_trans <- test_to_trans <- list()
  
#   # Select columns to preprocess
#   for (i in 1:length(training)){
#     train_to_trans[[i]] <- dplyr::select(training[[i]], all_of(column_to_process))
#     test_to_trans[[i]] <- dplyr::select(test[[i]], all_of(column_to_process))
#   }
  
#   # Calculate preprocessing parameters from training data
#   preProcValues <- list()
#   ocTrain <- ocTest <- list()
  
#   for(i in 1:length(train_to_trans)) {
#     # Fit preprocessing on training data
#     preProcValues[[i]] <- preProcess(train_to_trans[[i]],
#                                      method = dataprocessing_method)
    
#     # Apply preprocessing to both train and test
#     ocTrain[[i]] <- predict(preProcValues[[i]], train_to_trans[[i]])
#     ocTest[[i]] <- predict(preProcValues[[i]], test_to_trans[[i]])
#   }
  
#   # Recombine preprocessed and non-preprocessed columns
#   for(i in 1:length(ocTrain)) {
#     ocTrain[[i]] <- cbind(ocTrain[[i]], 
#                           dplyr::select(training[[i]], -all_of(column_to_process)))
#     ocTest[[i]] <- cbind(ocTest[[i]], 
#                          dplyr::select(test[[i]], -all_of(column_to_process)))
#   }
  

#   # TRAIN LINEAR SVM MODELS
#   # Purpose: Train one SVM for each outer fold with inner CV tuning
#   # CHANGE: Modify tuneLength for more/fewer cost parameters to try
#   AUCROC_fit <- list()
#   for (i in 1:length(ocTrain)){
#     AUCROC_fit[[i]] <- caret::train(
#       Label ~ .,
#       data = ocTrain[[i]],
#       method = 'svmLinear',                   # Linear kernel SVM
#       metric = "ROC",
#       trControl = trControl_man,
#       tuneLength = 5                          # Number of cost values to try
#     )
#   }

#   # Variable Importance
#   varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
#   for (i in 1:length(AUCROC_fit)){
#     varImp_list[[i]] <- varImp(AUCROC_fit[[i]], useModel = useModel_for_varImp, 
#                                nonpara = FALSE, scale = TRUE)
#     varImp_intermediate[[i]] <- select(extract2(varImp_list[[i]],1),1)
#     varImp_score_list[[i]] <- varImp_intermediate[[i]] %>% DBI::sqlRownamesToColumn()
#   }
#   varImp_score_df <-
#     reduce(varImp_score_list, full_join, by = 'row_names') %>% 
#     column_to_rownames(var = "row_names") %>% 
#     rowMeans() %>% 
#     base::as.data.frame()
#   colnames(varImp_score_df) <- "Importance_Score"
  
#   # Validation
#   AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- list()
#   for (i in 1:length(AUCROC_fit)){
#     AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]])
#     AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]], type = "prob")
#     AUCROC_ROC[[i]] <- roc(predictor = extract2(AUCROC_test_ROC[[i]], 2),
#                            response = ocTest[[i]]$Label)
#   }
  
#   # Confusion Matrices
#   AUCROC_confusion_matrix = list()
#   for (i in 1:length(AUCROC_fit)){
#     AUCROC_confusion_matrix[[i]] <- confusionMatrix(
#       data = AUCROC_test[[i]],
#       reference = as.factor(magrittr::use_series(test[[i]], "Label")),
#       positive = classname[1]
#     )
#   }
  
#   # AUC Statistics
#   auc_tot <- rep(0, k_fold_outer)
#   for (i in 1:length(AUCROC_ROC)){
#     auc_tot[i] <- AUCROC_ROC[[i]]$auc
#   }
#   auc_mean <- mean(auc_tot)
#   auc_sd <- sd(auc_tot)
  
#   # ROC Interpolation
#   common_grid_points <- 100
#   roc_point_sen <- data.frame(matrix(nrow = common_grid_points, ncol = length(test)))
#   roc_point_spe <- data.frame(matrix(nrow = common_grid_points, ncol = length(test)))
  
#   for (i in 1:length(AUCROC_ROC)){
#     orig_sen <- AUCROC_ROC[[i]]$sensitivities
#     orig_spe <- AUCROC_ROC[[i]]$specificities
#     orig_x <- seq(0, 1, length.out = length(orig_sen))
#     common_x <- seq(0, 1, length.out = common_grid_points)
#     roc_point_sen[,i] <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
#     roc_point_spe[,i] <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
#   }
  
#   roc_mean_sen <- apply(roc_point_sen, 1, mean)
#   roc_mean_spe <- apply(roc_point_spe, 1, mean)
#   roc_sd_sen <- apply(roc_point_sen, 1, sd)
#   roc_sd_spe <- apply(roc_point_spe, 1, sd)
  
#   roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
#   roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
#   roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
#   roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
#   # Visualization Data
#   # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")
#   Comparision <- "Good Prognosis vs Poor Prognosis"
#   avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
#   polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
#                                y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
#   # ROC Plot
#   ggplot() +
#     geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
#               color = "#366993", size = 1.18) +
#     geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
#     theme_Publication() +
#     coord_equal() +
#     geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
#     labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
#     scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     theme(axis.title = element_text(size = 19, face = "bold"),
#           axis.text = element_text(size = 17),
#           plot.title = element_text(size = 23, face = "bold")) -> p_RocCurve
  
#   AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
#                         "±", formatC(auc_sd, format = "f", digits = 2))
  
#   p_RocCurve = p_RocCurve +
#     annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
#     annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
#     annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
#              vjust = 0, color = "#366993", size = 1.18) +
#     annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
#     annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
#              alpha = .3, fill = "steelblue")
  
#   # Variable Importance Plot
#   gsub("`", "", rownames(varImp_score_df)) -> rownames(varImp_score_df)
#   p_VarImp_10 = varImp_score_df %>% 
#     na.omit() %>%
#     rownames_to_column() %>%
#     arrange(Importance_Score) %>%
#     top_n(n = nrow(varImp_score_df) * 10 / 100) %>%
#     mutate(rowname = factor(rowname, levels = rowname)) %>%
#     ggplot(aes(x = rowname, y = Importance_Score)) +
#     geom_segment(aes(xend = rowname, yend = 0)) +
#     geom_point(size = 5, color = "orange") +
#     coord_flip() +
#     theme_Publication() +
#     labs(x = "", y = "Importance Scores") +
#     theme(axis.title = element_text(size = 15, face = "bold"),
#           axis.text = element_text(size = 12.8))
  
#   # Performance Metrics
#   confuse_table_lg = AUCROC_confusion_matrix
#   Balanced_Accuracy_model_lg <- Sensitivity_model_lg <- Specificity_model_lg <- list() # nolint
#   for (i in 1:length(confuse_table_lg)) {
#     Sensitivity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Sensitivity"]]
#     Specificity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Specificity"]]
#     Balanced_Accuracy_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Balanced Accuracy"]]
#   }
  
#   Sensitivity_lg <- Sensitivity_model_lg %>% reduce(append)
#   Sensitivity_lg_sd <- sd(Sensitivity_lg)
#   Sensitivity_lg_mean <- mean(Sensitivity_lg) 
  
#   Specificity_lg <- Specificity_model_lg %>% reduce(append)
#   Specificity_lg_sd <- sd(Specificity_lg)
#   Specificity_lg_mean <- mean(Specificity_lg)
  
#   Balanced_Accuracy_lg <- Balanced_Accuracy_model_lg %>% reduce(append)
#   Balanced_Accuracy_lg_sd <- sd(Balanced_Accuracy_lg)
#   Balanced_Accuracy_lg_mean <- mean(Balanced_Accuracy_lg)
  
#   Metrics_Table <- data.frame(
#     Sensitivity_Mean = Sensitivity_lg_mean,
#     Sensitivity_SD = Sensitivity_lg_sd,
#     Specificity_Mean = Specificity_lg_mean,
#     Specificity_SD = Specificity_lg_sd,
#     Balanced_Accuracy_Mean = Balanced_Accuracy_lg_mean,
#     Balanced_Accuracy_SD = Balanced_Accuracy_lg_sd
#   ) %>%
#     setNames(c("Sensitivity, Mean", "Sensitivity, SD", 
#                "Specificity, Mean", "Specificity, SD", 
#                "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
#   # Return Results
#   return(
#     list(
#       "auc_tot" = auc_tot, 
#       "varImp_score_df" = varImp_score_df, 
#       "AUCROC_fit" = AUCROC_fit, 
#       "test" = test, 
#       "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
#       "Comparision" = Comparision,
#       "auc_mean" = auc_mean,
#       "auc_sd" = auc_sd,
#       "avg_AUC" = avg_AUC,
#       "polygon_SD_AUC" = polygon_SD_AUC,
#       "p_RocCurve" = p_RocCurve,
#       "p_VarImp_10" = p_VarImp_10,
#       "Metrics_Table" = Metrics_Table
#     )
#   )
# }
# =============================================================================
# SUPPORT VECTOR MACHINE (LINEAR) FUNCTION - REVISED (LiblineaR)
# =============================================================================
# =============================================================================
# SUPPORT VECTOR MACHINE (RADIAL) FUNCTION - FINAL FIX
# =============================================================================
# svm_perform_std_Revised <- function(
#     data_input_ML,
#     classname = c("Survival","NonSurvival"),      
#     column_to_process,                            
#     dataprocessing_method = c("center", "scale"), 
#     seed = 1,
#     split_ratio = 0.7,
#     k_fold_outer = 5,
#     k_fold_inner = 5,
#     useModel_for_varImp = TRUE 
# ){
  
#   # 1. LOAD REQUIRED LIBRARIES
#   require(caret)
#   require(pROC)
#   require(dplyr)
#   require(magrittr)
#   require(ggplot2)
#   require(tibble)
#   require(purrr)
#   # Kernlab là thư viện chuẩn cho svmRadial
#   require(kernlab) 
  

#   # 2. DATA PREPARATION
#   data_input_ML <- dplyr::filter(data_input_ML, Label %in% classname)
#   data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

#   # 3. CREATE TRAIN/TEST SPLITS
#   set.seed(seed)
#   trainIndex <- caret::createDataPartition(data_input_ML$Label,
#                                            p = split_ratio,
#                                            list = FALSE,
#                                            times = k_fold_outer)
  
#   training <- test <- list()
#   for (i in 1:ncol(trainIndex)){
#     training[[i]] <- data_input_ML[trainIndex[,i],] 
#     test[[i]] <- data_input_ML[-trainIndex[,i], ]
#   }
  

#   # 4. CONFIGURE INNER CV
#   # Lưu ý: svmRadial không cần thiết phải dùng sampling="down" gắt gao như Linear
#   # nhưng vẫn giữ để đảm bảo cân bằng nếu dữ liệu lệch nhiều.
#   trControl_man <- trainControl(
#     method = "cv",
#     number = k_fold_inner,
#     summaryFunction = twoClassSummary,
#     classProbs = TRUE,
#     savePredictions = "final"
#   )
  

#   # 5. TRAIN SVM RADIAL MODELS
#   # Thay đổi chiến lược: Để caret tự động preProcess (center/scale) toàn bộ.
#   # Việc scale cả biến binary (0/1) trong SVM Radial là chấp nhận được và tốt cho hội tụ.
  
#   AUCROC_fit <- list()
  
#   for (i in 1:length(training)){
#     # QUAN TRỌNG: Sử dụng try() để bắt lỗi nếu một fold cụ thể bị lỗi hội tụ
#     AUCROC_fit[[i]] <- try({
#       caret::train(
#         Label ~ .,
#         data = training[[i]],
#         method = 'svmRadial',           # FIX: Chuyển sang Radial Kernel (Phi tuyến)
#         metric = "ROC",
#         trControl = trControl_man,
#         preProcess = c("center", "scale", "nzv"), # Tự động xử lý biến
#         tuneLength = 10                 # Để caret tự tìm Sigma và Cost tốt nhất
#       )
#     }, silent = TRUE)
#   }

#   # Kiểm tra xem có fold nào bị lỗi không
#   valid_folds <- sapply(AUCROC_fit, function(x) !inherits(x, "try-error"))
#   if(sum(valid_folds) < length(AUCROC_fit)) {
#     warning(sprintf("Có %d/%d fold bị lỗi huấn luyện.", length(AUCROC_fit) - sum(valid_folds), length(AUCROC_fit)))
#   }
  

#   # 6. Variable Importance
#   # Với svmRadial, chúng ta không thể dùng 'coefficients' (useModel=TRUE).
#   # Phải dùng phương pháp filter (ROC curve based) -> useModel = FALSE
#   varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
  
#   for (i in 1:length(AUCROC_fit)){
#     if (valid_folds[i]) {
#       # Luôn dùng useModel = FALSE cho svmRadial
#       varImp_list[[i]] <- varImp(AUCROC_fit[[i]], useModel = FALSE, scale = TRUE)
#       varImp_intermediate[[i]] <- dplyr::select(magrittr::extract2(varImp_list[[i]], 1), 1)
      
#       # Fix lỗi DBI::sqlRownamesToColumn nếu không có thư viện DBI
#       temp_df <- varImp_intermediate[[i]]
#       temp_df$row_names <- rownames(temp_df)
#       varImp_score_list[[i]] <- temp_df
#     }
#   }
  
#   varImp_score_df <-
#     reduce(varImp_score_list, full_join, by = 'row_names') %>% 
#     column_to_rownames(var = "row_names") %>% 
#     rowMeans(na.rm = TRUE) %>% # Thêm na.rm = TRUE để an toàn
#     base::as.data.frame()
#   colnames(varImp_score_df) <- "Importance_Score"
  
  
#   # 7. Validation
#   AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- list()
  
#   for (i in 1:length(AUCROC_fit)){
#     if (valid_folds[i]) {
#       # Predict trực tiếp trên test set gốc, caret sẽ tự apply preProcess đã học từ train
#       AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]])
#       AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]], type = "prob")
#       AUCROC_ROC[[i]] <- roc(predictor = extract2(AUCROC_test_ROC[[i]], 2),
#                              response = test[[i]]$Label,
#                              quiet = TRUE) # Tắt thông báo của pROC
#     }
#   }
  
#   # 8. Confusion Matrices
#   AUCROC_confusion_matrix = list()
#   for (i in 1:length(AUCROC_fit)){
#     if (valid_folds[i]) {
#       AUCROC_confusion_matrix[[i]] <- confusionMatrix(
#         data = AUCROC_test[[i]],
#         reference = as.factor(magrittr::use_series(test[[i]], "Label")),
#         positive = classname[1]
#       )
#     }
#   }
  
#   # 9. AUC Statistics
#   auc_tot <- numeric()
#   for (i in 1:length(AUCROC_ROC)){
#     if (!is.null(AUCROC_ROC[[i]])) {
#       auc_tot <- c(auc_tot, AUCROC_ROC[[i]]$auc)
#     }
#   }
#   auc_mean <- mean(auc_tot)
#   auc_sd <- sd(auc_tot)
  
#   # 10. ROC Interpolation
#   common_grid_points <- 100
#   roc_point_sen <- matrix(nrow = common_grid_points, ncol = 0)
#   roc_point_spe <- matrix(nrow = common_grid_points, ncol = 0)
  
#   for (i in 1:length(AUCROC_ROC)){
#     if (!is.null(AUCROC_ROC[[i]])) {
#       orig_sen <- AUCROC_ROC[[i]]$sensitivities
#       orig_spe <- AUCROC_ROC[[i]]$specificities
#       orig_x <- seq(0, 1, length.out = length(orig_sen))
#       common_x <- seq(0, 1, length.out = common_grid_points)
      
#       interp_sen <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
#       interp_spe <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
      
#       roc_point_sen <- cbind(roc_point_sen, interp_sen)
#       roc_point_spe <- cbind(roc_point_spe, interp_spe)
#     }
#   }
  
#   roc_mean_sen <- apply(roc_point_sen, 1, mean)
#   roc_mean_spe <- apply(roc_point_spe, 1, mean)
#   roc_sd_sen <- apply(roc_point_sen, 1, sd)
#   roc_sd_spe <- apply(roc_point_spe, 1, sd)
  
#   roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
#   roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
#   roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
#   roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
#   # 11. Visualization Data
#   Comparision <- "Good Prognosis vs Poor Prognosis"
#   avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
#   polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
#                                y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
#   # 12. ROC Plot
#   ggplot() +
#     geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
#               color = "#366993", size = 1.18) +
#     geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
#     theme_Publication() +
#     coord_equal() +
#     geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
#     labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
#     scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     theme(axis.title = element_text(size = 19, face = "bold"),
#           axis.text = element_text(size = 17),
#           plot.title = element_text(size = 23, face = "bold")) -> p_RocCurve
  
#   AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
#                         "±", formatC(auc_sd, format = "f", digits = 2))
  
#   p_RocCurve = p_RocCurve +
#     annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
#     annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
#     annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
#              vjust = 0, color = "#366993", size = 1.18) +
#     annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
#     annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
#              alpha = .3, fill = "steelblue")
  
#   # 13. Variable Importance Plot
#   # Clean row names strictly
#   clean_names <- gsub("`", "", rownames(varImp_score_df))
#   rownames(varImp_score_df) <- clean_names
  
#   p_VarImp_10 = varImp_score_df %>% 
#     na.omit() %>%
#     rownames_to_column() %>%
#     arrange(Importance_Score) %>%
#     top_n(n = max(5, nrow(varImp_score_df) * 10 / 100)) %>% # Lấy ít nhất 5 biến
#     mutate(rowname = factor(rowname, levels = rowname)) %>%
#     ggplot(aes(x = rowname, y = Importance_Score)) +
#     geom_segment(aes(xend = rowname, yend = 0)) +
#     geom_point(size = 5, color = "orange") +
#     coord_flip() +
#     theme_Publication() +
#     labs(x = "", y = "Importance Scores") +
#     theme(axis.title = element_text(size = 15, face = "bold"),
#           axis.text = element_text(size = 12.8))
  
#   # 14. Performance Metrics
#   confuse_table_lg = AUCROC_confusion_matrix
  
#   # Khởi tạo vector rỗng an toàn
#   Sensitivity_lg <- numeric()
#   Specificity_lg <- numeric()
#   Balanced_Accuracy_lg <- numeric()
  
#   for (i in 1:length(confuse_table_lg)) {
#     if (!is.null(confuse_table_lg[[i]])) {
#       Sensitivity_lg <- c(Sensitivity_lg, confuse_table_lg[[i]][["byClass"]][["Sensitivity"]])
#       Specificity_lg <- c(Specificity_lg, confuse_table_lg[[i]][["byClass"]][["Specificity"]])
#       Balanced_Accuracy_lg <- c(Balanced_Accuracy_lg, confuse_table_lg[[i]][["byClass"]][["Balanced Accuracy"]])
#     }
#   }
  
#   Metrics_Table <- data.frame(
#     Sensitivity_Mean = mean(Sensitivity_lg, na.rm=TRUE),
#     Sensitivity_SD = sd(Sensitivity_lg, na.rm=TRUE),
#     Specificity_Mean = mean(Specificity_lg, na.rm=TRUE),
#     Specificity_SD = sd(Specificity_lg, na.rm=TRUE),
#     Balanced_Accuracy_Mean = mean(Balanced_Accuracy_lg, na.rm=TRUE),
#     Balanced_Accuracy_SD = sd(Balanced_Accuracy_lg, na.rm=TRUE)
#   ) %>%
#     setNames(c("Sensitivity, Mean", "Sensitivity, SD", 
#                "Specificity, Mean", "Specificity, SD", 
#                "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
#   # Return Results
#   return(
#     list(
#       "auc_tot" = auc_tot, 
#       "varImp_score_df" = varImp_score_df, 
#       "AUCROC_fit" = AUCROC_fit, 
#       "test" = test, 
#       "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
#       "Comparision" = Comparision,
#       "auc_mean" = auc_mean,
#       "auc_sd" = auc_sd,
#       "avg_AUC" = avg_AUC,
#       "polygon_SD_AUC" = polygon_SD_AUC,
#       "p_RocCurve" = p_RocCurve,
#       "p_VarImp_10" = p_VarImp_10,
#       "Metrics_Table" = Metrics_Table
#     )
#   )
# }
# =============================================================================
# SUPPORT VECTOR MACHINE (RADIAL) FUNCTION - FINAL REVISED
# Có cơ chế chọn cột để process (column_to_process)
# =============================================================================
svm_perform_std_Revised <- function(
    data_input_ML,
    classname = c("Survival","NonSurvival"),      
    seed = 1,
    split_ratio = 0.7,
    k_fold_outer = 5,
    k_fold_inner = 5,
    useModel_for_varImp = TRUE,
    # --- THAM SỐ MỚI ---
    column_to_process = NULL,                            
    dataprocessing_method = c("center", "scale")
){
  
  # 1. LOAD REQUIRED LIBRARIES
  require(caret)
  require(pROC)
  require(dplyr)
  require(magrittr)
  require(ggplot2)
  require(tibble)
  require(purrr)
  require(kernlab) # Cho svmRadial
  

  # 2. DATA PREPARATION
  data_input_ML <- dplyr::filter(data_input_ML, Label %in% classname)
  data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

  # 3. CREATE TRAIN/TEST SPLITS
  set.seed(seed)
  trainIndex <- caret::createDataPartition(data_input_ML$Label,
                                           p = split_ratio,
                                           list = FALSE,
                                           times = k_fold_outer)
  
  training <- test <- list()
  for (i in 1:ncol(trainIndex)){
    training[[i]] <- data_input_ML[trainIndex[,i],] 
    test[[i]] <- data_input_ML[-trainIndex[,i], ]
  }
  

  # 4. CONFIGURE INNER CV
  trControl_man <- trainControl(
    method = "cv",
    number = k_fold_inner,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final"
  )
  

  # 5. PREPROCESS DATA (MANUAL SELECTION)
  # Chỉ scale các cột trong column_to_process, giữ nguyên cột khác (như binary)
  
  ocTrain <- list()
  ocTest <- list()
  
  # Kiểm tra nếu có cột cần xử lý
  if(!is.null(column_to_process) && length(column_to_process) > 0) {
    train_to_trans <- test_to_trans <- list()
    
    for (i in 1:length(training)){
      # Tách cột cần scale
      train_to_trans[[i]] <- dplyr::select(training[[i]], all_of(column_to_process))
      test_to_trans[[i]] <- dplyr::select(test[[i]], all_of(column_to_process))
    }
    
    preProcValues <- list()
    for(i in 1:length(train_to_trans)) {
      # Tính toán tham số scale trên tập train (thêm nzv để lọc nhiễu)
      methods_to_use <- unique(c(dataprocessing_method, "nzv"))
      preProcValues[[i]] <- preProcess(train_to_trans[[i]], method = methods_to_use)
      
      # Áp dụng scale
      scaled_train <- predict(preProcValues[[i]], train_to_trans[[i]])
      scaled_test <- predict(preProcValues[[i]], test_to_trans[[i]])
      
      # Ghép lại với các cột KHÔNG cần scale (giữ nguyên bản)
      ocTrain[[i]] <- cbind(scaled_train, 
                            dplyr::select(training[[i]], -all_of(column_to_process)))
      ocTest[[i]] <- cbind(scaled_test, 
                           dplyr::select(test[[i]], -all_of(column_to_process)))
    }
  } else {
    # Nếu không có danh sách cột, dùng nguyên dữ liệu gốc (hoặc caret tự xử lý sau)
    ocTrain <- training
    ocTest <- test
  }


  # 6. TRAIN SVM RADIAL MODELS
  AUCROC_fit <- list()
  
  for (i in 1:length(ocTrain)){
    AUCROC_fit[[i]] <- try({
      caret::train(
        Label ~ .,
        data = ocTrain[[i]],            # Dùng dữ liệu đã xử lý thủ công
        method = 'svmLinear',           # FIX: Chuyển sang Radial Kernel (Phi tuyến)
        metric = "ROC",
        trControl = trControl_man,
        # preProcess = NULL,            # Tắt tự động vì đã làm thủ công ở trên
        tuneLength = 10                 
      )
    }, silent = TRUE)
  }

  # Kiểm tra lỗi hội tụ
  valid_folds <- sapply(AUCROC_fit, function(x) !inherits(x, "try-error"))
  if(sum(valid_folds) < length(AUCROC_fit)) {
    warning(sprintf("Có %d/%d fold bị lỗi huấn luyện SVM.", length(AUCROC_fit) - sum(valid_folds), length(AUCROC_fit)))
  }
  

  # 7. Variable Importance (Filter approach for Radial)
  varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
  
  for (i in 1:length(AUCROC_fit)){
    if (valid_folds[i]) {
      varImp_list[[i]] <- varImp(AUCROC_fit[[i]], useModel = FALSE, scale = TRUE)
      varImp_intermediate[[i]] <- dplyr::select(magrittr::extract2(varImp_list[[i]], 1), 1)
      
      temp_df <- varImp_intermediate[[i]]
      temp_df$row_names <- rownames(temp_df)
      varImp_score_list[[i]] <- temp_df
    }
  }
  
  varImp_score_df <-
    reduce(varImp_score_list, full_join, by = 'row_names') %>% 
    column_to_rownames(var = "row_names") %>% 
    rowMeans(na.rm = TRUE) %>% 
    base::as.data.frame()
  colnames(varImp_score_df) <- "Importance_Score"
  
  
  # 8. Validation
  AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- list()
  
  for (i in 1:length(AUCROC_fit)){
    if (valid_folds[i]) {
      # Predict trên tập test đã xử lý (ocTest)
      AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]])
      AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]], type = "prob")
      AUCROC_ROC[[i]] <- roc(predictor = extract2(AUCROC_test_ROC[[i]], 2),
                             response = ocTest[[i]]$Label,
                             quiet = TRUE)
    }
  }
  
  # 9. Confusion Matrices
  AUCROC_confusion_matrix = list()
  for (i in 1:length(AUCROC_fit)){
    if (valid_folds[i]) {
      AUCROC_confusion_matrix[[i]] <- confusionMatrix(
        data = AUCROC_test[[i]],
        reference = as.factor(magrittr::use_series(ocTest[[i]], "Label")),
        positive = classname[1]
      )
    }
  }
  
  # 10. AUC Statistics & Visualization (Giữ nguyên như cũ)
  auc_tot <- numeric()
  for (i in 1:length(AUCROC_ROC)){
    if (!is.null(AUCROC_ROC[[i]])) {
      auc_tot <- c(auc_tot, AUCROC_ROC[[i]]$auc)
    }
  }
  auc_mean <- mean(auc_tot)
  auc_sd <- sd(auc_tot)
  
  # ROC Interpolation
  common_grid_points <- 100
  roc_point_sen <- matrix(nrow = common_grid_points, ncol = 0)
  roc_point_spe <- matrix(nrow = common_grid_points, ncol = 0)
  
  for (i in 1:length(AUCROC_ROC)){
    if (!is.null(AUCROC_ROC[[i]])) {
      orig_sen <- AUCROC_ROC[[i]]$sensitivities
      orig_spe <- AUCROC_ROC[[i]]$specificities
      orig_x <- seq(0, 1, length.out = length(orig_sen))
      common_x <- seq(0, 1, length.out = common_grid_points)
      
      interp_sen <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
      interp_spe <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
      
      roc_point_sen <- cbind(roc_point_sen, interp_sen)
      roc_point_spe <- cbind(roc_point_spe, interp_spe)
    }
  }
  
  roc_mean_sen <- apply(roc_point_sen, 1, mean)
  roc_mean_spe <- apply(roc_point_spe, 1, mean)
  roc_sd_sen <- apply(roc_point_sen, 1, sd)
  roc_sd_spe <- apply(roc_point_spe, 1, sd)
  
  roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
  roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
  roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
  roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
  # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")
  Comparision <- "Good Prognosis vs Poor Prognosis"
  avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
  polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
                               y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
  p_RocCurve <-
    ggplot() +
    geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
              color = "#366993", size = 1.18) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
    theme_Publication() +
    coord_equal() +
    geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
    labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    theme(axis.title = element_text(size = 19, face = "bold"),
          axis.text = element_text(size = 17),
          plot.title = element_text(size = 23, face = "bold"))
  
  AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
                        "±", formatC(auc_sd, format = "f", digits = 2))
  
  p_RocCurve <- p_RocCurve +
    annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
    annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
    annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
             vjust = 0, color = "#366993", size = 1.18) +
    annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
    annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
             alpha = .3, fill = "steelblue")
  
  clean_names <- gsub("`", "", rownames(varImp_score_df))
  rownames(varImp_score_df) <- clean_names
  
  p_VarImp_10 <- varImp_score_df %>%
    na.omit() %>%
    tibble::rownames_to_column() %>% 
    dplyr::slice_max(order_by = Importance_Score, n = max(5, floor(0.10 * nrow(varImp_score_df))), with_ties = TRUE) %>%
    dplyr::arrange(Importance_Score) %>%
    dplyr::mutate(rowname = factor(rowname, levels = rowname)) %>%
    ggplot2::ggplot(ggplot2::aes(x = rowname, y = Importance_Score)) +
    ggplot2::geom_segment(ggplot2::aes(xend = rowname, yend = 0)) +
    ggplot2::geom_point(size = 5, color = "orange") +
    ggplot2::coord_flip() +
    theme_Publication() +
    ggplot2::labs(x = "", y = "Importance Scores") +
    ggplot2::theme(axis.title = element_text(size = 15, face = "bold"),
                   axis.text = element_text(size = 12.8))
  
  Sensitivity_lg <- numeric()
  Specificity_lg <- numeric()
  Balanced_Accuracy_lg <- numeric()
  
  for (i in 1:length(AUCROC_confusion_matrix)) {
    if(!is.null(AUCROC_confusion_matrix[[i]])){
      Sensitivity_lg <- c(Sensitivity_lg, AUCROC_confusion_matrix[[i]][["byClass"]][["Sensitivity"]])
      Specificity_lg <- c(Specificity_lg, AUCROC_confusion_matrix[[i]][["byClass"]][["Specificity"]])
      Balanced_Accuracy_lg <- c(Balanced_Accuracy_lg, AUCROC_confusion_matrix[[i]][["byClass"]][["Balanced Accuracy"]])
    }
  }
  
  Metrics_Table <- data.frame(
    Sensitivity_Mean = mean(Sensitivity_lg, na.rm=TRUE),
    Sensitivity_SD = sd(Sensitivity_lg, na.rm=TRUE),
    Specificity_Mean = mean(Specificity_lg, na.rm=TRUE),
    Specificity_SD = sd(Specificity_lg, na.rm=TRUE),
    Balanced_Accuracy_Mean = mean(Balanced_Accuracy_lg, na.rm=TRUE),
    Balanced_Accuracy_SD = sd(Balanced_Accuracy_lg, na.rm=TRUE)
  ) %>%
    setNames(c("Sensitivity, Mean", "Sensitivity, SD",
               "Specificity, Mean", "Specificity, SD",
               "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
  return(
    list(
      "auc_tot" = auc_tot, 
      "varImp_score_df" = varImp_score_df, 
      "AUCROC_fit" = AUCROC_fit, 
      "test" = test, 
      "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
      "Comparision" = Comparision,
      "auc_mean" = auc_mean,
      "auc_sd" = auc_sd,
      "avg_AUC" = avg_AUC,
      "polygon_SD_AUC" = polygon_SD_AUC,
      "p_RocCurve" = p_RocCurve,
      "p_VarImp_10" = p_VarImp_10,
      "Metrics_Table" = Metrics_Table
    )
  )
}

# =============================================================================
# LOGISTIC REGRESSION FUNCTION
# =============================================================================
# logit_perform_std_Revised <- function(
#     data_input_ML,
#     classname = c("Survival","NonSurvival"),  # CHANGE: Your class labels
#     seed = 1,
#     split_ratio = 0.7,
#     k_fold_outer = 5,
#     k_fold_inner = 5,
#     useModel_for_varImp = TRUE  
# ){
  

#   # 1. LOAD LIBRARIES & HELPER FUNCTION
#   # Purpose: Load packages and define fallback for DBI function
#   # Note: sqlRownamesToColumn_safe handles cases where DBI is not available
#   require(caret)
#   require(pROC)
#   library(tidyverse)
#   require(magrittr)
  
#   # Helper function for rowname conversion (in case DBI not available)
#   sqlRownamesToColumn_safe <- function(df, name = "row_names"){
#     if ("DBI" %in% .packages(all.available = TRUE) &&
#         "sqlRownamesToColumn" %in% getNamespaceExports("DBI")) {
#       DBI::sqlRownamesToColumn(df, name)
#     } else {
#       tibble::rownames_to_column(df, var = name)
#     }
#   }
  

#   # 2. DATA PREPARATION
#   # Purpose: Filter and factorize labels
#   data_input_ML <- dplyr::filter(data_input_ML, Label %in% classname)
#   data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

#   # 3. CREATE TRAIN/TEST SPLITS
#   # Purpose: Generate stratified splits for outer CV
#   set.seed(seed)
#   trainIndex <- caret::createDataPartition(
#     data_input_ML$Label, p = split_ratio, list = FALSE, times = k_fold_outer
#   )
#   training <- test <- vector("list", ncol(trainIndex))
#   for (i in 1:ncol(trainIndex)){
#     training[[i]] <- data_input_ML[trainIndex[, i], ]
#     test[[i]] <- data_input_ML[-trainIndex[, i], ]
#   }
  

#   # 4. CONFIGURE INNER CV
#   # Purpose: Set up cross-validation for logistic regression
#   # 4. CONFIGURE INNER CV
#   # CẬP NHẬT: Thêm sampling = "down" (hoặc "up") để cân bằng dữ liệu
#   trControl_man <- trainControl(
#     method = "cv",
#     number = k_fold_inner,
#     summaryFunction = twoClassSummary,
#     classProbs = TRUE,
#     sampling = "down" # QUAN TRỌNG: Down-sampling giúp cân bằng số lượng 2 lớp
#   )
  

#   # 5. PREPROCESS DATA (SCALING/CENTERING/NZV)
#   train_to_trans <- test_to_trans <- list()
  
#   # Select columns to preprocess
#   for (i in 1:length(training)){
#     train_to_trans[[i]] <- dplyr::select(training[[i]], all_of(column_to_process))
#     test_to_trans[[i]] <- dplyr::select(test[[i]], all_of(column_to_process))
#   }
  
#   preProcValues <- list()
#   ocTrain <- ocTest <- list()
  
#   for(i in 1:length(train_to_trans)) {
#     # Fit preprocessing on training data
#     # Đảm bảo dùng nzv để loại bỏ biến nhiễu
#     methods_to_use <- unique(c(dataprocessing_method, "nzv"))
#     preProcValues[[i]] <- preProcess(train_to_trans[[i]],
#                                      method = methods_to_use)
    
#     # Apply preprocessing
#     ocTrain[[i]] <- predict(preProcValues[[i]], train_to_trans[[i]])
#     ocTest[[i]] <- predict(preProcValues[[i]], test_to_trans[[i]])
#   }
  
#   # Recombine
#   for(i in 1:length(ocTrain)) {
#     ocTrain[[i]] <- cbind(ocTrain[[i]], 
#                           dplyr::select(training[[i]], -all_of(column_to_process)))
#     ocTest[[i]] <- cbind(ocTest[[i]], 
#                          dplyr::select(test[[i]], -all_of(column_to_process)))
#   }
  

#   # 6. TRAIN LINEAR SVM MODELS (Revised Tuning)
#   # Purpose: Train one SVM for each outer fold with CUSTOM GRID
#   AUCROC_fit <- list()
  
#   # CẬP NHẬT: Tạo lưới tham số Cost thủ công để ép mô hình tìm biên tốt hơn
#   # Cost nhỏ (0.01) = Regularization mạnh (tránh overfitting)
#   # Cost lớn (100) = Regularization yếu (cố gắng phân loại đúng từng điểm)
#   svmGrid <- expand.grid(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 50, 100))
  
#   for (i in 1:length(ocTrain)){
#     AUCROC_fit[[i]] <- caret::train(
#       Label ~ .,
#       data = ocTrain[[i]],
#       method = 'svmLinear2',      # Sử dụng LiblineaR
#       metric = "ROC",
#       trControl = trControl_man,
#       tuneGrid = svmGrid          # QUAN TRỌNG: Dùng grid thủ công thay vì tuneLength
#     )
#   }
  

#   # 7-15. [Same sections as RF/SVM]
#   # Validation, ROC analysis, plotting, and metrics follow same pattern

#   # Validation
#   AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- vector("list", length(AUCROC_fit))
#   for (i in 1:length(AUCROC_fit)){
#     AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]])
#     AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]], type = "prob")
#     AUCROC_ROC[[i]] <- pROC::roc(
#       predictor = magrittr::extract2(AUCROC_test_ROC[[i]], 2),
#       response = test[[i]]$Label
#     )
#   }
  
#   # Confusion Matrices
#   AUCROC_confusion_matrix <- vector("list", length(AUCROC_fit))
#   for (i in 1:length(AUCROC_fit)){
#     AUCROC_confusion_matrix[[i]] <- caret::confusionMatrix(
#       AUCROC_test[[i]],
#       as.factor(magrittr::use_series(test[[i]], "Label"))
#     )
#   }
  
#   # AUC Statistics
#   auc_tot <- rep(0, k_fold_outer)
#   for (i in 1:length(AUCROC_ROC)){
#     auc_tot[i] <- AUCROC_ROC[[i]]$auc
#   }
#   auc_mean <- mean(auc_tot)
#   auc_sd <- sd(auc_tot)
  
#   # ROC Interpolation
#   common_grid_points <- 100
#   roc_point_sen <- data.frame(matrix(nrow = common_grid_points, ncol = length(test)))
#   roc_point_spe <- data.frame(matrix(nrow = common_grid_points, ncol = length(test)))
#   for (i in 1:length(AUCROC_ROC)){
#     orig_sen <- AUCROC_ROC[[i]]$sensitivities
#     orig_spe <- AUCROC_ROC[[i]]$specificities
#     orig_x <- seq(0, 1, length.out = length(orig_sen))
#     common_x <- seq(0, 1, length.out = common_grid_points)
#     roc_point_sen[, i] <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
#     roc_point_spe[, i] <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
#   }
#   roc_mean_sen <- apply(roc_point_sen, 1, mean)
#   roc_mean_spe <- apply(roc_point_spe, 1, mean)
#   roc_sd_sen <- apply(roc_point_sen, 1, sd)
#   roc_sd_spe <- apply(roc_point_spe, 1, sd)
#   roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
#   roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
#   roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
#   roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
#   # Visualization Data
#   # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")
#   # If you want to customize Comparision, uncomment and modify the line below
#   Comparision <- "Good Prognosis vs Poor Prognosis"
#   avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
#   polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
#                                y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
#   # ROC Plot
#   p_RocCurve <-
#     ggplot() +
#     geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
#               color = "#366993", size = 1.18) +
#     geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
#     theme_Publication() +
#     coord_equal() +
#     geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
#     labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
#     scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     theme(axis.title = element_text(size = 19, face = "bold"),
#           axis.text = element_text(size = 17),
#           plot.title = element_text(size = 23, face = "bold"))
  
#   AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
#                         "±", formatC(auc_sd, format = "f", digits = 2))
  
#   p_RocCurve <- p_RocCurve +
#     annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
#     annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
#     annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
#              vjust = 0, color = "#366993", size = 1.18) +
#     annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
#     annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
#              alpha = .3, fill = "steelblue")
  
#   # Variable Importance Plot
#   top_k <- if (nrow(varImp_score_df) < 50) nrow(varImp_score_df) else ceiling(0.10 * nrow(varImp_score_df))
#   gsub("`", "", rownames(varImp_score_df)) -> rownames(varImp_score_df)
#   p_VarImp_10 <- varImp_score_df %>%
#     na.omit() %>%
#     tibble::rownames_to_column() %>% 
#     dplyr::slice_max(order_by = Importance_Score, n = top_k, with_ties = TRUE) %>%
#     dplyr::arrange(Importance_Score) %>%
#     dplyr::mutate(rowname = factor(rowname, levels = rowname)) %>%
#     ggplot2::ggplot(ggplot2::aes(x = rowname, y = Importance_Score)) +
#     ggplot2::geom_segment(ggplot2::aes(xend = rowname, yend = 0)) +
#     ggplot2::geom_point(size = 5, color = "orange") +
#     ggplot2::coord_flip() +
#     theme_Publication() +
#     ggplot2::labs(x = "", y = "Importance Scores") +
#     ggplot2::theme(axis.title = element_text(size = 15, face = "bold"),
#                    axis.text = element_text(size = 12.8))
  
#   # Performance Metrics
#   confuse_table_lg <- AUCROC_confusion_matrix
#   Sensitivity_model_lg <- Specificity_model_lg <- Balanced_Accuracy_model_lg <- vector("list", length(confuse_table_lg))
#   for (i in 1:length(confuse_table_lg)) {
#     Sensitivity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Sensitivity"]]
#     Specificity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Specificity"]]
#     Balanced_Accuracy_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Balanced Accuracy"]]
#   }
#   Sensitivity_lg <- purrr::reduce(Sensitivity_model_lg, append)
#   Specificity_lg <- purrr::reduce(Specificity_model_lg, append)
#   Balanced_Accuracy_lg <- purrr::reduce(Balanced_Accuracy_model_lg, append)
  
#   Metrics_Table <- data.frame(
#     Sensitivity_Mean = mean(Sensitivity_lg),
#     Sensitivity_SD = sd(Sensitivity_lg),
#     Specificity_Mean = mean(Specificity_lg),
#     Specificity_SD = sd(Specificity_lg),
#     Balanced_Accuracy_Mean = mean(Balanced_Accuracy_lg),
#     Balanced_Accuracy_SD = sd(Balanced_Accuracy_lg)
#   ) %>%
#     setNames(c("Sensitivity, Mean", "Sensitivity, SD",
#                "Specificity, Mean", "Specificity, SD",
#                "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
#   # Return Results
#   return(
#     list(
#       "auc_tot" = auc_tot, 
#       "varImp_score_df" = varImp_score_df, 
#       "AUCROC_fit" = AUCROC_fit, 
#       "test" = test, 
#       "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
#       "Comparision" = Comparision,
#       "auc_mean" = auc_mean,
#       "auc_sd" = auc_sd,
#       "avg_AUC" = avg_AUC,
#       "polygon_SD_AUC" = polygon_SD_AUC,
#       "p_RocCurve" = p_RocCurve,
#       "p_VarImp_10" = p_VarImp_10,
#       "Metrics_Table" = Metrics_Table
#     )
#   )
# }
logit_perform_std_Revised <- function(
    data_input_ML,
    classname = c("Survival","NonSurvival"),  # CHANGE: Your class labels
    seed = 1,
    split_ratio = 0.7,
    k_fold_outer = 5,
    k_fold_inner = 5,
    useModel_for_varImp = TRUE,
    # --- [FIX 1] THÊM THAM SỐ ĐỂ TRÁNH LỖI "Run failed" ---
    column_to_process = NULL,
    dataprocessing_method = c("center", "scale")
){
  
  # 1. LOAD LIBRARIES & HELPER FUNCTION
  require(caret)
  require(pROC)
  library(tidyverse)
  require(magrittr)
  
  # Helper function for rowname conversion
  sqlRownamesToColumn_safe <- function(df, name = "row_names"){
    if ("DBI" %in% .packages(all.available = TRUE) &&
        "sqlRownamesToColumn" %in% getNamespaceExports("DBI")) {
      DBI::sqlRownamesToColumn(df, name)
    } else {
      tibble::rownames_to_column(df, var = name)
    }
  }
  
  # 2. DATA PREPARATION
  data_input_ML <- dplyr::filter(data_input_ML, Label %in% classname)
  data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  
  # 3. CREATE TRAIN/TEST SPLITS
  set.seed(seed)
  trainIndex <- caret::createDataPartition(
    data_input_ML$Label, p = split_ratio, list = FALSE, times = k_fold_outer
  )
  training <- test <- vector("list", ncol(trainIndex))
  for (i in 1:ncol(trainIndex)){
    training[[i]] <- data_input_ML[trainIndex[, i], ]
    test[[i]] <- data_input_ML[-trainIndex[, i], ]
  }
  
  # 4. CONFIGURE INNER CV
  trControl_man <- trainControl(
    method = "cv",
    number = k_fold_inner,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    # sampling = "down" # Down-sampling để cân bằng dữ liệu
  )
  
  # 5. PREPROCESS DATA (SCALING/CENTERING/NZV)
  # Lưu ý: Với 18 gene thì scale hay không cũng được, nhưng giữ lại để code không lỗi
  train_to_trans <- test_to_trans <- list()
  
  # Select columns to preprocess if provided
  if(!is.null(column_to_process)) {
      for (i in 1:length(training)){
        train_to_trans[[i]] <- dplyr::select(training[[i]], all_of(column_to_process))
        test_to_trans[[i]] <- dplyr::select(test[[i]], all_of(column_to_process))
      }
      
      preProcValues <- list()
      ocTrain <- ocTest <- list()
      
      for(i in 1:length(train_to_trans)) {
        methods_to_use <- unique(c(dataprocessing_method, "nzv"))
        preProcValues[[i]] <- preProcess(train_to_trans[[i]], method = methods_to_use)
        
        ocTrain[[i]] <- predict(preProcValues[[i]], train_to_trans[[i]])
        ocTest[[i]] <- predict(preProcValues[[i]], test_to_trans[[i]])
      }
      
      # Recombine
      for(i in 1:length(ocTrain)) {
        ocTrain[[i]] <- cbind(ocTrain[[i]], 
                              dplyr::select(training[[i]], -all_of(column_to_process)))
        ocTest[[i]] <- cbind(ocTest[[i]], 
                             dplyr::select(test[[i]], -all_of(column_to_process)))
      }
  } else {
      # Nếu không có cột nào cần scale, dùng dữ liệu gốc
      ocTrain <- training
      ocTest <- test
  }
  print(head(ocTrain[[1]]))

  # 6. TRAIN LOGISTIC REGRESSION MODELS (Standard GLM)
  # --- [FIX 2] CHUYỂN TỪ SVM SANG GLM (LOGIT) ---
  AUCROC_fit <- list()
  
  for (i in 1:length(ocTrain)){
    AUCROC_fit[[i]] <- caret::train(
      Label ~ .,
      data = ocTrain[[i]],
      method = 'glm',             # Sử dụng Generalized Linear Model
      family = 'binomial',        # Phân phối nhị thức (Logistic)
      metric = "ROC",
      trControl = trControl_man
      # Không cần tuneGrid cho Standard GLM
    )
  }
  
  # 7. Variable Importance
  varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
  for (i in 1:length(AUCROC_fit)){
    varImp_list[[i]] <- caret::varImp(
        AUCROC_fit[[i]], 
        useModel = useModel_for_varImp, 
        scale = TRUE
    )
    # Lấy cột đầu tiên (Overall importance)
    varImp_intermediate[[i]] <- dplyr::select(magrittr::extract2(varImp_list[[i]], 1), 1)
    
    # Fix lỗi nếu thư viện DBI chưa load
    temp_df <- varImp_intermediate[[i]]
    temp_df$row_names <- rownames(temp_df)
    varImp_score_list[[i]] <- temp_df
  }
  
  varImp_score_df <-
    purrr::reduce(varImp_score_list, dplyr::full_join, by = "row_names") %>%
    tibble::column_to_rownames(var = "row_names") %>%
    rowMeans(na.rm = TRUE) %>% 
    base::as.data.frame()
  colnames(varImp_score_df) <- "Importance_Score"
  
  # 8. Validation
  AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- vector("list", length(AUCROC_fit))
  for (i in 1:length(AUCROC_fit)){
    # Predict trên tập test đã qua xử lý (ocTest)
    AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]])
    AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]], type = "prob")
    AUCROC_ROC[[i]] <- pROC::roc(
      predictor = magrittr::extract2(AUCROC_test_ROC[[i]], 2),
      response = ocTest[[i]]$Label,
      quiet = TRUE
    )
  }
  
  # 9. Confusion Matrices
  AUCROC_confusion_matrix <- vector("list", length(AUCROC_fit))
  for (i in 1:length(AUCROC_fit)){
    AUCROC_confusion_matrix[[i]] <- caret::confusionMatrix(
      AUCROC_test[[i]],
      as.factor(magrittr::use_series(ocTest[[i]], "Label"))
    )
  }
  
  # 10. AUC Statistics
  auc_tot <- numeric()
  for (i in 1:length(AUCROC_ROC)){
      if(!is.null(AUCROC_ROC[[i]])){
          auc_tot <- c(auc_tot, AUCROC_ROC[[i]]$auc)
      }
  }
  auc_mean <- mean(auc_tot)
  auc_sd <- sd(auc_tot)
  
  # 11. ROC Interpolation
  common_grid_points <- 100
  roc_point_sen <- matrix(nrow = common_grid_points, ncol = 0)
  roc_point_spe <- matrix(nrow = common_grid_points, ncol = 0)
  
  for (i in 1:length(AUCROC_ROC)){
    if(!is.null(AUCROC_ROC[[i]])){
        orig_sen <- AUCROC_ROC[[i]]$sensitivities
        orig_spe <- AUCROC_ROC[[i]]$specificities
        orig_x <- seq(0, 1, length.out = length(orig_sen))
        common_x <- seq(0, 1, length.out = common_grid_points)
        
        interp_sen <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
        interp_spe <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
        
        roc_point_sen <- cbind(roc_point_sen, interp_sen)
        roc_point_spe <- cbind(roc_point_spe, interp_spe)
    }
  }
  roc_mean_sen <- apply(roc_point_sen, 1, mean)
  roc_mean_spe <- apply(roc_point_spe, 1, mean)
  roc_sd_sen <- apply(roc_point_sen, 1, sd)
  roc_sd_spe <- apply(roc_point_spe, 1, sd)
  
  roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
  roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
  roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
  roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
  # 12. Visualization Data
  # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")
  Comparision <- "Good Prognosis vs Poor Prognosis"
  avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
  polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
                               y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
  # 13. ROC Plot
  p_RocCurve <-
    ggplot() +
    geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
              color = "#366993", size = 1.18) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
    theme_Publication() +
    coord_equal() +
    geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
    labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    theme(axis.title = element_text(size = 19, face = "bold"),
          axis.text = element_text(size = 17),
          plot.title = element_text(size = 23, face = "bold"))
  
  AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
                        "±", formatC(auc_sd, format = "f", digits = 2))
  
  p_RocCurve <- p_RocCurve +
    annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
    annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
    annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
             vjust = 0, color = "#366993", size = 1.18) +
    annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
    annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
             alpha = .3, fill = "steelblue")
  
  # 14. Variable Importance Plot
  # Lấy 10% hoặc tối thiểu 5 biến quan trọng nhất
  top_k <- max(5, floor(0.10 * nrow(varImp_score_df)))
  clean_names <- gsub("`", "", rownames(varImp_score_df))
  rownames(varImp_score_df) <- clean_names
  
  p_VarImp_10 <- varImp_score_df %>%
    na.omit() %>%
    tibble::rownames_to_column() %>% 
    dplyr::slice_max(order_by = Importance_Score, n = top_k, with_ties = TRUE) %>%
    dplyr::arrange(Importance_Score) %>%
    dplyr::mutate(rowname = factor(rowname, levels = rowname)) %>%
    ggplot2::ggplot(ggplot2::aes(x = rowname, y = Importance_Score)) +
    ggplot2::geom_segment(ggplot2::aes(xend = rowname, yend = 0)) +
    ggplot2::geom_point(size = 5, color = "orange") +
    ggplot2::coord_flip() +
    theme_Publication() +
    ggplot2::labs(x = "", y = "Importance Scores") +
    ggplot2::theme(axis.title = element_text(size = 15, face = "bold"),
                   axis.text = element_text(size = 12.8))
  
  # 15. Performance Metrics
  confuse_table_lg <- AUCROC_confusion_matrix
  Sensitivity_lg <- numeric()
  Specificity_lg <- numeric()
  Balanced_Accuracy_lg <- numeric()

  for (i in 1:length(confuse_table_lg)) {
      if(!is.null(confuse_table_lg[[i]])){
        Sensitivity_lg <- c(Sensitivity_lg, confuse_table_lg[[i]][["byClass"]][["Sensitivity"]])
        Specificity_lg <- c(Specificity_lg, confuse_table_lg[[i]][["byClass"]][["Specificity"]])
        Balanced_Accuracy_lg <- c(Balanced_Accuracy_lg, confuse_table_lg[[i]][["byClass"]][["Balanced Accuracy"]])
      }
  }
  
  Metrics_Table <- data.frame(
    Sensitivity_Mean = mean(Sensitivity_lg, na.rm=TRUE),
    Sensitivity_SD = sd(Sensitivity_lg, na.rm=TRUE),
    Specificity_Mean = mean(Specificity_lg, na.rm=TRUE),
    Specificity_SD = sd(Specificity_lg, na.rm=TRUE),
    Balanced_Accuracy_Mean = mean(Balanced_Accuracy_lg, na.rm=TRUE),
    Balanced_Accuracy_SD = sd(Balanced_Accuracy_lg, na.rm=TRUE)
  ) %>%
    setNames(c("Sensitivity, Mean", "Sensitivity, SD",
               "Specificity, Mean", "Specificity, SD",
               "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
  # Return Results
  return(
    list(
      "auc_tot" = auc_tot, 
      "varImp_score_df" = varImp_score_df, 
      "AUCROC_fit" = AUCROC_fit, 
      "test" = test, 
      "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
      "Comparision" = Comparision,
      "auc_mean" = auc_mean,
      "auc_sd" = auc_sd,
      "avg_AUC" = avg_AUC,
      "polygon_SD_AUC" = polygon_SD_AUC,
      "p_RocCurve" = p_RocCurve,
      "p_VarImp_10" = p_VarImp_10,
      "Metrics_Table" = Metrics_Table
    )
  )
}

# =============================================================================
# PARTIAL LEAST SQUARES DISCRIMINANT ANALYSIS (PLS-DA) FUNCTION
# =============================================================================
# plsda_perform_std_Revised <- function(
#     data_input_ML,
#     classname = c("Survival","NonSurvival"),  # CHANGE: Your class labels
#     seed = 1,
#     split_ratio = 0.7,
#     k_fold_outer = 5,
#     k_fold_inner = 5,
#     useModel_for_varImp = TRUE  
# ){
  

#   # 1. LOAD REQUIRED LIBRARIES
#   # Purpose: Load packages including 'pls' for PLS-DA
#   require(caret)
#   require(pROC)
#   require(pls)          # Required for PLS method
#   library(tidyverse)
#   require(magrittr)
  

#   # 2. DATA PREPARATION
#   # Purpose: Filter and factorize labels
#   data_input_ML <- dplyr::filter(data_input_ML, Label %in% classname)
#   data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

#   # 3. CREATE TRAIN/TEST SPLITS
#   # Purpose: Generate stratified splits for outer CV
#   set.seed(seed)
#   trainIndex <- caret::createDataPartition(
#     data_input_ML$Label, p = split_ratio, list = FALSE, times = k_fold_outer
#   )
#   training <- test <- vector("list", ncol(trainIndex))
#   for (i in 1:ncol(trainIndex)){
#     training[[i]] <- data_input_ML[trainIndex[, i], ]
#     test[[i]] <- data_input_ML[-trainIndex[, i], ]
#   }
  

#   # 4. CONFIGURE INNER CV
#   # Purpose: Set up cross-validation for PLS-DA
#   trControl_man <- caret::trainControl(
#     method = "cv",
#     number = k_fold_inner,
#     summaryFunction = twoClassSummary,
#     classProbs = TRUE
#   )
  

#   # 5. TRAIN PLS-DA MODELS
#   # Purpose: Train PLS using caret's "pls" method
#   # Note: PLS projects data to latent space before classification
#   # CHANGE: Modify tuneLength to try different numbers of components
#   AUCROC_fit <- vector("list", length(training))
#   for (i in 1:length(training)){
#     AUCROC_fit[[i]] <- caret::train(
#       Label ~ .,
#       data = training[[i]],
#       method = "pls",                         # Partial Least Squares
#       preProcess = c("center", "scale"),      # PLS requires centering/scaling
#       metric = "ROC",
#       trControl = trControl_man,
#       tuneLength = 15                         # Number of components to try
#     )
#   }
  

#   # 6-15. [Same sections as other functions]
#   # Variable importance, validation, ROC analysis, plotting, metrics

#   # Variable Importance
#   varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
#   for (i in 1:length(AUCROC_fit)){
#     varImp_list[[i]] <- caret::varImp(
#       AUCROC_fit[[i]],
#       useModel = useModel_for_varImp,
#       nonpara = FALSE,
#       scale = TRUE
#     )
#     varImp_intermediate[[i]] <- dplyr::select(magrittr::extract2(varImp_list[[i]], 1), 1)
#     varImp_score_list[[i]] <- varImp_intermediate[[i]] %>% DBI::sqlRownamesToColumn()
#   }
#   varImp_score_df <-
#     purrr::reduce(varImp_score_list, dplyr::full_join, by = "row_names") %>%
#     tibble::column_to_rownames(var = "row_names") %>%
#     rowMeans() %>% base::as.data.frame()
#   colnames(varImp_score_df) <- "Importance_Score"
  
#   # Validation
#   AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- vector("list", length(AUCROC_fit))
#   for (i in 1:length(AUCROC_fit)){
#     AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]])
#     AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = test[[i]], type = "prob")
#     AUCROC_ROC[[i]] <- pROC::roc(
#       predictor = magrittr::extract2(AUCROC_test_ROC[[i]], 2),
#       response = test[[i]]$Label
#     )
#   }
  
#   # Confusion Matrices
#   AUCROC_confusion_matrix <- vector("list", length(AUCROC_fit))
#   for (i in 1:length(AUCROC_fit)){
#     AUCROC_confusion_matrix[[i]] <- caret::confusionMatrix(
#       AUCROC_test[[i]],
#       as.factor(magrittr::use_series(test[[i]], "Label"))
#     )
#   }
  
#   # AUC Statistics
#   auc_tot <- rep(0, k_fold_outer)
#   for (i in 1:length(AUCROC_ROC)){
#     auc_tot[i] <- AUCROC_ROC[[i]]$auc
#   }
#   auc_mean <- mean(auc_tot)
#   auc_sd <- sd(auc_tot)
  
#   # ROC Interpolation
#   common_grid_points <- 100
#   roc_point_sen <- data.frame(matrix(nrow = common_grid_points, ncol = length(test)))
#   roc_point_spe <- data.frame(matrix(nrow = common_grid_points, ncol = length(test)))
#   for (i in 1:length(AUCROC_ROC)){
#     orig_sen <- AUCROC_ROC[[i]]$sensitivities
#     orig_spe <- AUCROC_ROC[[i]]$specificities
#     orig_x <- seq(0, 1, length.out = length(orig_sen))
#     common_x <- seq(0, 1, length.out = common_grid_points)
#     roc_point_sen[, i] <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
#     roc_point_spe[, i] <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
#   }
#   roc_mean_sen <- apply(roc_point_sen, 1, mean)
#   roc_mean_spe <- apply(roc_point_spe, 1, mean)
#   roc_sd_sen <- apply(roc_point_sen, 1, sd)
#   roc_sd_spe <- apply(roc_point_spe, 1, sd)
#   roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
#   roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
#   roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
#   roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
#   # Visualization Data
#   # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")
#   # If you want to customize Comparision, uncomment and modify the line below
#   Comparision <- "Good Prognosis vs Poor Prognosis"
#   avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
#   polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
#                                y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
#   # ROC Plot
#   p_RocCurve <-
#     ggplot() +
#     geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
#               color = "#366993", size = 1.18) +
#     geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
#     theme_Publication() +
#     coord_equal() +
#     geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
#     labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
#     scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
#     theme(axis.title = element_text(size = 19, face = "bold"),
#           axis.text = element_text(size = 17),
#           plot.title = element_text(size = 23, face = "bold"))
  
#   AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
#                         "±", formatC(auc_sd, format = "f", digits = 2))
  
#   p_RocCurve <- p_RocCurve +
#     annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
#     annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
#     annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
#              vjust = 0, color = "#366993", size = 1.18) +
#     annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
#     annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
#              alpha = .3, fill = "steelblue")
  
#   # Variable Importance Plot
#   gsub("`", "", rownames(varImp_score_df)) -> rownames(varImp_score_df)
#   p_VarImp_10 <- varImp_score_df %>%
#     na.omit() %>%
#     tibble::rownames_to_column() %>%
#     dplyr::arrange(Importance_Score) %>%
#     dplyr::top_n(n = nrow(varImp_score_df) * 10 / 100) %>%
#     dplyr::mutate(rowname = factor(rowname, levels = rowname)) %>%
#     ggplot2::ggplot(ggplot2::aes(x = rowname, y = Importance_Score)) +
#     ggplot2::geom_segment(ggplot2::aes(xend = rowname, yend = 0)) +
#     ggplot2::geom_point(size = 5, color = "orange") +
#     ggplot2::coord_flip() +
#     theme_Publication() +
#     ggplot2::labs(x = "", y = "Importance Scores") +
#     ggplot2::theme(axis.title = element_text(size = 15, face = "bold"),
#                    axis.text = element_text(size = 12.8))
  
#   # Performance Metrics
#   confuse_table_lg <- AUCROC_confusion_matrix
#   Sensitivity_model_lg <- Specificity_model_lg <- Balanced_Accuracy_model_lg <- vector("list", length(confuse_table_lg))
#   for (i in 1:length(confuse_table_lg)) {
#     Sensitivity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Sensitivity"]]
#     Specificity_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Specificity"]]
#     Balanced_Accuracy_model_lg[[i]] <- confuse_table_lg[[i]][["byClass"]][["Balanced Accuracy"]]
#   }
#   Sensitivity_lg <- purrr::reduce(Sensitivity_model_lg, append)
#   Specificity_lg <- purrr::reduce(Specificity_model_lg, append)
#   Balanced_Accuracy_lg <- purrr::reduce(Balanced_Accuracy_model_lg, append)
  
#   Metrics_Table <- data.frame(
#     Sensitivity_Mean = mean(Sensitivity_lg),
#     Sensitivity_SD = sd(Sensitivity_lg),
#     Specificity_Mean = mean(Specificity_lg),
#     Specificity_SD = sd(Specificity_lg),
#     Balanced_Accuracy_Mean = mean(Balanced_Accuracy_lg),
#     Balanced_Accuracy_SD = sd(Balanced_Accuracy_lg)
#   ) %>%
#     setNames(c("Sensitivity, Mean", "Sensitivity, SD",
#                "Specificity, Mean", "Specificity, SD",
#                "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
#   # Return Results
#   return(
#     list(
#       "auc_tot" = auc_tot, 
#       "varImp_score_df" = varImp_score_df, 
#       "AUCROC_fit" = AUCROC_fit, 
#       "test" = test, 
#       "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
#       "Comparision" = Comparision,
#       "auc_mean" = auc_mean,
#       "auc_sd" = auc_sd,
#       "avg_AUC" = avg_AUC,
#       "polygon_SD_AUC" = polygon_SD_AUC,
#       "p_RocCurve" = p_RocCurve,
#       "p_VarImp_10" = p_VarImp_10,
#       "Metrics_Table" = Metrics_Table
#     )
#   )
# }
# =============================================================================
# PARTIAL LEAST SQUARES DISCRIMINANT ANALYSIS (PLS-DA) FUNCTION - REVISED
# Có cơ chế chọn cột để process (column_to_process)
# =============================================================================
plsda_perform_std_Revised <- function(
    data_input_ML,
    classname = c("Survival","NonSurvival"),  
    seed = 1,
    split_ratio = 0.7,
    k_fold_outer = 5,
    k_fold_inner = 5,
    useModel_for_varImp = TRUE,
    # --- THAM SỐ MỚI ---
    column_to_process = NULL,
    dataprocessing_method = c("center", "scale")
){
  

  # 1. LOAD REQUIRED LIBRARIES
  require(caret)
  require(pROC)
  require(pls)          
  library(tidyverse)
  require(magrittr)
  

  # 2. DATA PREPARATION
  data_input_ML <- dplyr::filter(data_input_ML, Label %in% classname)
  data_input_ML$Label <- factor(data_input_ML$Label, levels = classname)
  

  # 3. CREATE TRAIN/TEST SPLITS
  set.seed(seed)
  trainIndex <- caret::createDataPartition(
    data_input_ML$Label, p = split_ratio, list = FALSE, times = k_fold_outer
  )
  training <- test <- vector("list", ncol(trainIndex))
  for (i in 1:ncol(trainIndex)){
    training[[i]] <- data_input_ML[trainIndex[, i], ]
    test[[i]] <- data_input_ML[-trainIndex[, i], ]
  }
  

  # 4. CONFIGURE INNER CV
  trControl_man <- caret::trainControl(
    method = "cv",
    number = k_fold_inner,
    summaryFunction = twoClassSummary,
    classProbs = TRUE
  )
  

  # 5. PREPROCESS DATA (MANUAL SELECTION)
  # Logic tương tự như SVM: Scale các cột chỉ định, giữ nguyên các cột còn lại
  
  ocTrain <- list()
  ocTest <- list()
  
  if(!is.null(column_to_process) && length(column_to_process) > 0) {
    train_to_trans <- test_to_trans <- list()
    for (i in 1:length(training)){
      train_to_trans[[i]] <- dplyr::select(training[[i]], all_of(column_to_process))
      test_to_trans[[i]] <- dplyr::select(test[[i]], all_of(column_to_process))
    }
    
    preProcValues <- list()
    for(i in 1:length(train_to_trans)) {
      # PLS thường chỉ cần center & scale
      preProcValues[[i]] <- preProcess(train_to_trans[[i]], method = dataprocessing_method)
      
      scaled_train <- predict(preProcValues[[i]], train_to_trans[[i]])
      scaled_test <- predict(preProcValues[[i]], test_to_trans[[i]])
      
      # Ghép lại
      ocTrain[[i]] <- cbind(scaled_train, 
                            dplyr::select(training[[i]], -all_of(column_to_process)))
      ocTest[[i]] <- cbind(scaled_test, 
                           dplyr::select(test[[i]], -all_of(column_to_process)))
    }
  } else {
    ocTrain <- training
    ocTest <- test
  }


  # 6. TRAIN PLS-DA MODELS
  AUCROC_fit <- vector("list", length(ocTrain))
  
  for (i in 1:length(ocTrain)){
    AUCROC_fit[[i]] <- caret::train(
      Label ~ .,
      data = ocTrain[[i]],                    # Dùng dữ liệu đã xử lý thủ công
      method = "pls",                         
      # preProcess = NULL,                    # Tắt tự động vì đã làm thủ công
      metric = "ROC",
      trControl = trControl_man,
      tuneLength = 15                         
    )
  }
  

  # 7-15. Variable Importance & Validation (Giữ nguyên)
  varImp_list <- varImp_score_list <- varImp_score_df <- varImp_intermediate <- list()
  for (i in 1:length(AUCROC_fit)){
    varImp_list[[i]] <- caret::varImp(
      AUCROC_fit[[i]],
      useModel = useModel_for_varImp,
      nonpara = FALSE,
      scale = TRUE
    )
    varImp_intermediate[[i]] <- dplyr::select(magrittr::extract2(varImp_list[[i]], 1), 1)
    
    # Fix lỗi DBI::sqlRownamesToColumn an toàn
    temp_df <- varImp_intermediate[[i]]
    temp_df$row_names <- rownames(temp_df)
    varImp_score_list[[i]] <- temp_df
  }
  varImp_score_df <-
    purrr::reduce(varImp_score_list, dplyr::full_join, by = "row_names") %>%
    tibble::column_to_rownames(var = "row_names") %>%
    rowMeans(na.rm = TRUE) %>% 
    base::as.data.frame()
  colnames(varImp_score_df) <- "Importance_Score"
  
  # Validation
  AUCROC_test <- AUCROC_test_ROC <- AUCROC_ROC <- vector("list", length(AUCROC_fit))
  for (i in 1:length(AUCROC_fit)){
    AUCROC_test[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]])
    AUCROC_test_ROC[[i]] <- predict(AUCROC_fit[[i]], newdata = ocTest[[i]], type = "prob")
    AUCROC_ROC[[i]] <- pROC::roc(
      predictor = magrittr::extract2(AUCROC_test_ROC[[i]], 2),
      response = ocTest[[i]]$Label,
      quiet = TRUE
    )
  }
  
  # Confusion Matrices
  AUCROC_confusion_matrix <- vector("list", length(AUCROC_fit))
  for (i in 1:length(AUCROC_fit)){
    AUCROC_confusion_matrix[[i]] <- caret::confusionMatrix(
      AUCROC_test[[i]],
      as.factor(magrittr::use_series(ocTest[[i]], "Label"))
    )
  }
  
  # AUC Statistics & Visualization (Giữ nguyên)
  auc_tot <- numeric()
  for (i in 1:length(AUCROC_ROC)){
    if (!is.null(AUCROC_ROC[[i]])) {
      auc_tot <- c(auc_tot, AUCROC_ROC[[i]]$auc)
    }
  }
  auc_mean <- mean(auc_tot)
  auc_sd <- sd(auc_tot)
  
  # ROC Interpolation
  common_grid_points <- 100
  roc_point_sen <- matrix(nrow = common_grid_points, ncol = 0)
  roc_point_spe <- matrix(nrow = common_grid_points, ncol = 0)
  
  for (i in 1:length(AUCROC_ROC)){
    if (!is.null(AUCROC_ROC[[i]])) {
      orig_sen <- AUCROC_ROC[[i]]$sensitivities
      orig_spe <- AUCROC_ROC[[i]]$specificities
      orig_x <- seq(0, 1, length.out = length(orig_sen))
      common_x <- seq(0, 1, length.out = common_grid_points)
      
      interp_sen <- approx(x = orig_x, y = orig_sen, xout = common_x, rule = 2)$y
      interp_spe <- approx(x = orig_x, y = orig_spe, xout = common_x, rule = 2)$y
      
      roc_point_sen <- cbind(roc_point_sen, interp_sen)
      roc_point_spe <- cbind(roc_point_spe, interp_spe)
    }
  }
  roc_mean_sen <- apply(roc_point_sen, 1, mean)
  roc_mean_spe <- apply(roc_point_spe, 1, mean)
  roc_sd_sen <- apply(roc_point_sen, 1, sd)
  roc_sd_spe <- apply(roc_point_spe, 1, sd)
  
  roc_sd_spe_upper <- roc_mean_spe + roc_sd_spe
  roc_sd_sen_uppper <- roc_mean_sen + roc_sd_sen
  roc_sd_spe_lower <- roc_mean_spe - roc_sd_spe
  roc_sd_sen_lower <- roc_mean_sen - roc_sd_sen
  
  # Comparision <- data_input_ML$Label %>% unique() %>% paste(collapse = " vs ")
  Comparision <- "Good Prognosis vs Poor Prognosis"
  avg_AUC <- data.frame(roc_mean_spe = rev(roc_mean_spe), roc_mean_sen = rev(roc_mean_sen))
  polygon_SD_AUC <- data.frame(x = c(roc_sd_spe_lower, rev(roc_sd_spe_upper)),
                               y = c(roc_sd_sen_lower, rev(roc_sd_sen_uppper)))
  
  p_RocCurve <-
    ggplot() +
    geom_line(aes(x = 1 - avg_AUC$roc_mean_spe, y = avg_AUC$roc_mean_sen), 
              color = "#366993", size = 1.18) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 1, color = "grey") +
    theme_Publication() +
    coord_equal() +
    geom_polygon(data = polygon_SD_AUC, aes(x = 1 - x, y = y), fill = "steelblue", alpha = 0.2) +
    labs(x = "False Positive Rate", y = "True Positive Rate", title = Comparision) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0.02, 0.02)) +
    theme(axis.title = element_text(size = 19, face = "bold"),
          axis.text = element_text(size = 17),
          plot.title = element_text(size = 23, face = "bold"))
  
  AUC_plot_show = paste("AUC:", formatC(auc_mean, format = "f", digits = 2),
                        "±", formatC(auc_sd, format = "f", digits = 2))
  
  p_RocCurve <- p_RocCurve +
    annotate("text", x = 0.72, y = 0.16, vjust = 0, size = 7, label = AUC_plot_show) +
    annotate("text", x = 0.848, y = 0.09, vjust = 0, size = 5, label = "Mean ROC") +
    annotate("segment", x = 0.64, xend = 0.72, y = 0.106, yend = 0.106, 
             vjust = 0, color = "#366993", size = 1.18) +
    annotate("text", x = 0.772, y = 0.02, vjust = 0, size = 5, label = "Standard Deviation") +
    annotate("rect", xmin = 0.48, xmax = 0.56, ymin = 0.02, ymax = 0.05, 
             alpha = .3, fill = "steelblue")
  
  # Variable Importance Plot
  clean_names <- gsub("`", "", rownames(varImp_score_df))
  rownames(varImp_score_df) <- clean_names
  
  p_VarImp_10 <- varImp_score_df %>%
    na.omit() %>%
    tibble::rownames_to_column() %>%
    dplyr::arrange(Importance_Score) %>%
    dplyr::top_n(n = max(5, floor(0.10 * nrow(varImp_score_df)))) %>%
    dplyr::mutate(rowname = factor(rowname, levels = rowname)) %>%
    ggplot2::ggplot(ggplot2::aes(x = rowname, y = Importance_Score)) +
    ggplot2::geom_segment(ggplot2::aes(xend = rowname, yend = 0)) +
    ggplot2::geom_point(size = 5, color = "orange") +
    ggplot2::coord_flip() +
    theme_Publication() +
    ggplot2::labs(x = "", y = "Importance Scores") +
    ggplot2::theme(axis.title = element_text(size = 15, face = "bold"),
                   axis.text = element_text(size = 12.8))
  
  # Metrics
  Sensitivity_lg <- numeric()
  Specificity_lg <- numeric()
  Balanced_Accuracy_lg <- numeric()
  
  for (i in 1:length(AUCROC_confusion_matrix)) {
    if(!is.null(AUCROC_confusion_matrix[[i]])){
      Sensitivity_lg <- c(Sensitivity_lg, AUCROC_confusion_matrix[[i]][["byClass"]][["Sensitivity"]])
      Specificity_lg <- c(Specificity_lg, AUCROC_confusion_matrix[[i]][["byClass"]][["Specificity"]])
      Balanced_Accuracy_lg <- c(Balanced_Accuracy_lg, AUCROC_confusion_matrix[[i]][["byClass"]][["Balanced Accuracy"]])
    }
  }
  
  Metrics_Table <- data.frame(
    Sensitivity_Mean = mean(Sensitivity_lg, na.rm=TRUE),
    Sensitivity_SD = sd(Sensitivity_lg, na.rm=TRUE),
    Specificity_Mean = mean(Specificity_lg, na.rm=TRUE),
    Specificity_SD = sd(Specificity_lg, na.rm=TRUE),
    Balanced_Accuracy_Mean = mean(Balanced_Accuracy_lg, na.rm=TRUE),
    Balanced_Accuracy_SD = sd(Balanced_Accuracy_lg, na.rm=TRUE)
  ) %>%
    setNames(c("Sensitivity, Mean", "Sensitivity, SD",
               "Specificity, Mean", "Specificity, SD",
               "Balanced Accuracy, Mean", "Balanced Accuracy, SD"))
  
  return(
    list(
      "auc_tot" = auc_tot, 
      "varImp_score_df" = varImp_score_df, 
      "AUCROC_fit" = AUCROC_fit, 
      "test" = test, 
      "AUCROC_confusion_matrix" = AUCROC_confusion_matrix,
      "Comparision" = Comparision,
      "auc_mean" = auc_mean,
      "auc_sd" = auc_sd,
      "avg_AUC" = avg_AUC,
      "polygon_SD_AUC" = polygon_SD_AUC,
      "p_RocCurve" = p_RocCurve,
      "p_VarImp_10" = p_VarImp_10,
      "Metrics_Table" = Metrics_Table
    )
  )
}

# ----------------------------------------------------------------------------
# CUSTOM GGPLOT THEME FOR PUBLICATION-QUALITY PLOTS
# ----------------------------------------------------------------------------
# Purpose: Creates clean, professional-looking plots suitable for publications
# CHANGE: Modify base_size for different font sizes
# CHANGE: Modify colors and grid settings for different aesthetics
# ----------------------------------------------------------------------------
theme_Publication <- function(base_size = 14, base_family = "helvetica") {
  library(grid)
  library(ggthemes)

  (theme_foundation(base_size = base_size, base_family = base_family)
    + theme(
      plot.title = element_text(size = rel(1.2), hjust = 0.5),   # Centered title, slightly larger font
      text = element_text(),                                     # Base text style
      panel.border = element_rect(colour = NA),                  # Remove panel border
      axis.title = element_text(size = rel(1)),                  # Axis title font size
      axis.title.y = element_text(angle = 90, vjust = 2),        # Rotate y-axis title and shift up
      axis.title.x = element_text(vjust = -0.2),                 # Slightly lower x-axis title
      axis.text = element_text(),                                # Axis tick labels
      axis.line = element_line(colour = "black"),                # Black axis lines
      axis.ticks = element_line(),                               # Show tick marks
      panel.grid.major = element_line(colour = "#f0f0f0"),       # Light grey major gridlines
      panel.background = element_rect(fill = "white", colour = NA),  # White panel background
      plot.background = element_rect(fill = "white", colour = NA),   # White plot background
      panel.grid.minor = element_blank(),                        # Hide minor gridlines
      legend.key = element_rect(colour = NA),                    # Remove legend key borders
      legend.position = "bottom",                                # Place legend at the bottom
      legend.direction = "horizontal",                           # Horizontal legend layout
      legend.key.size = unit(0.2, "cm"),                         # Legend key size
      legend.margin = margin(unit(0, "cm")),                     # Remove inner legend margin
      legend.title = element_text(face = "italic"),              # Italic legend title
      plot.margin = unit(c(10, 5, 5, 5), "mm"),                  # Outer plot margins (top, right, bottom, left)
      strip.background = element_rect(colour = "#f0f0f0", fill = "#f0f0f0"), # Facet strip background color
      strip.text = element_text(face = "bold")                   # Bold facet strip text
    ))
}

