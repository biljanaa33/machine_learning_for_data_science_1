#install.packages(c("e1071", "caret", "nnet"))
#.libPaths("C:/Users/Biljana/AppData/Local/R/win-library/4.4")
library("e1071")
library("caret")
library("ggplot2")
library("nnet")
library("dplyr")
library("tidyr")

set.seed(42)

### HELPER FUNCTIONS
### Perform startifed smampling, imbalanced classes 
stratified_folds <- function(y, k) {

  folds <- vector("list", k)
  classes <- unique(y)
  
  for (class in classes) {
    class_indices <- which(y == class)
    class_indices <- sample(class_indices)  # shuffle within class
    class_folds <- split(class_indices, rep(1:k, length.out = length(class_indices)))
    
    for (i in 1:k) {
      folds[[i]] <- c(folds[[i]], class_folds[[i]])
    }
  }
  
  return(folds)
}

transform_data <- function(data) {

    # no additional data processing, transform the data
    # to be suitbale as an input to the models
    data$ShotType <- as.factor(data$ShotType)
    data$ShotType <- as.numeric(data$ShotType)
    data$Angle <- as.numeric(data$Angle)
    data$Distance <- as.numeric(data$Distance)
    data$Movement <- as.numeric(as.factor(data$Movement))
    data$PlayerType <- as.numeric(as.factor(data$PlayerType))
    data$Competition <- as.numeric(as.factor(data$Competition))

    return(data)
}

convert_to_df <- function(result_k){
  
  result_k <- result_k$fold_results
  
  # use to store the scores with log score
  #preds_error = diag(fold$pred_errors), 

  df <- do.call(rbind, lapply(result_k, function(fold) {
  data.frame(
    distance_val = unlist(fold$distance_val),  
    preds_error = unlist(fold$pred_errors), 
    competition = unlist(fold$competition)
    ) 
  }))
  return(df)

}

########################################### CLASSIFIERS #################################################

baseline_classifier <- function(train_data, train_target, test_data, test_target, prob_flag = TRUE) {

   classes <- sort(unique(train_target))
   class_counts <- table(train_target)
   class_prob <- class_counts / sum(class_counts)
   n_test <- length(test_target)
   K <- length(classes)
   # stratifed sampling based on relative freq
   predictions <- sample(classes, size = n_test, replace = TRUE, prob = class_prob)
  return(predictions)
}

logistic_regression <- function(train_data, train_target, test_data, test_target, prob_flag = TRUE) {

 
    train_df <- cbind(train_data, ShotType = train_target)
    model <- multinom(ShotType ~ ., data = train_df, trace = FALSE)
    if(prob_flag) {
   
      predictions_prob <- predict(model, newdata=test_data, type="probs")
      # print(dim(predictions_prob))
      return(predictions_prob)
  } 
  
   predictions_class <- predict(model, newdata=test_data, type="class")
   # print(predictions_class)
   return(predictions_class)

}

radial_svm <- function(train_data, train_target, test_data, test_target, cost = 1, prob_flag = TRUE) {

    train_df <- cbind(train_data, ShotType = train_target)
    train_df$ShotType <- as.numeric(train_df$ShotType)

    model <- svm(ShotType ~ ., data = train_df, kernel = "radial", cost = cost,probability = TRUE, type = "C-classification")
    predictions <- predict(model, test_data)
    return(predictions) 
    
}

 
########################################## LOSS FUNCTIONS ##################################################        

accuracy <- function(predictions, true_values ) {

    return( sum(predictions == true_values) / length(true_values) )
}

log_loss_multi <- function(prob_vector, true_values) {
 
  # use for log reg
  n <- length(true_values)
  K <- max(true_values)
  prob_matrix <- matrix(prob_vector, nrow = n, ncol = K, byrow = TRUE)
  eps <- 1e-15
  prob_matrix[prob_matrix < eps] <- eps
  prob_matrix[prob_matrix > 1 - eps] <- 1 - eps
  
  n <- nrow(prob_matrix)
  log_sum <- 0
  
  for (i in seq_len(n)) {
    col_idx <- true_values[i]
    p_k <- prob_matrix[i, col_idx]  
    log_sum <- log_sum + log(p_k)
  }
  
  return(log_sum / n)
}


log_loss <- function(predictions, true_values){

    # avoid log(0)
    predictions <- as.integer(predictions != true_values)
    predictions[predictions == 0] <- 1e-10
    predictions[predictions == 1] <- 1 - 1e-10
    predictions <- as.numeric(predictions) 
  
    loss <- mean(log(predictions))
    return(loss)
}



################################## CROSS VALIDATON METHODS ######################################                                                  


k_fold_cv <- function(data, k = 10, model_fn, metric_fn, prob_flag = TRUE, log_reg_flag = FALSE){

    # number of samples
    n <- nrow(data)
    data <- data[sample(1:n), ]
    folds <- stratified_folds(data$ShotType,k)

    performance_folds <- c()
    pred_errors <- c()
    distance_val <- c()

    fold_results <- list()

    for(i in 1:k) {

        # get indecies
        test_indecies <- folds[[i]]
        train_indecies <- setdiff(1:n, test_indecies)
       
        train_data <- data[train_indecies, ]
        test_data <- data[test_indecies, ]

        train_target <- train_data$ShotType
        test_target <- test_data$ShotType
        train_data$ShotType <- NULL
        test_data$ShotType <- NULL

        predictions <- model_fn(train_data, train_target, test_data, test_target, prob_flag)
        if(log_reg_flag == TRUE){
          # print(predictions)
          pred_errors <- predictions[, test_target]
          # print(pred_errors)
        }
        predictions <- as.numeric(predictions)
        performance_fold <- metric_fn(as.numeric(predictions), test_target)
        performance_folds <- c(performance_folds, performance_fold)
        print(performance_folds)
        if(log_reg_flag == FALSE){
          
          pred_errors <- as.integer(predictions != test_target)
        
        }

        # store preidction errors and distance data for the current fold
        fold_results[[i]] <- list(
            performance = performance_fold,
            pred_errors = pred_errors,
            distance_val = test_data$Distance,
            competition = test_data$Competition

        )
        
    } 

    performance_se <- sqrt(var(performance_folds)/k)
    performance_mean <- mean(performance_folds)
   
    print(paste("Mean:", performance_mean, "Standard Deviation:", performance_se))

    return(list(
        performance_mean = performance_mean, 
        performance_se = performance_se, 
        fold_results = fold_results ))
}

# optimizing training fold perfromance  cross validation 
opt_training_fold <- function(data, k = 10, model_fn, metric_fn, prob_flag = TRUE, visualize = FALSE){

    # number of samples
    n <- nrow(data)

    data <- data[sample(1:n), ]
    folds <- stratified_folds(data$ShotType,k)

    
    performance_folds <- c()
    pred_errors <- c()
    distance_val <- c()

    fold_results <- list()
    best_cs <- list()

    for(i in 1:k) {

        test_indecies <- folds[[i]]
        train_indecies <- setdiff(1:n, test_indecies)

        train_data <- data[train_indecies, ]
        test_data <- data[test_indecies, ]

        train_target <- train_data$ShotType
        test_target <- test_data$ShotType
        train_data$ShotType <- NULL
        test_data$ShotType <- NULL

        c_values <- 10^seq(-3, 3, length.out = 10)
        # print(c_values)
        best_c = NULL
        best_perf = -Inf
        c_values <- c(1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3)
        for(c in c_values){
            predictions_train <- model_fn(train_data, train_target, train_data, train_target, c = c, prob_flag)
            predictions_train <- as.numeric(predictions_train)
            performance_fold_train <- metric_fn(predictions_train, train_target)
            
            if(performance_fold_train > best_perf){
                best_perf <- performance_fold_train
                best_c <- c
            } 

        }
      
        best_cs <- c(best_cs, best_c)
        predictions <- model_fn(train_data, train_target, test_data, test_target, c = best_c, prob_flag)
        predictions <- as.numeric(predictions)
        performance_fold <- metric_fn(predictions, test_target)
        performance_folds <- c(performance_folds, performance_fold)
        pred_errors <- as.integer(predictions != test_target)


        fold_results[[i]] <- list(
            performance = performance_fold,
            pred_errors = pred_errors,
            distance_val = test_data$Distance,
            competition = test_data$Competition
        )
        
    } 

    performance_se <- sqrt(var(performance_folds)/k)
    performance_mean <- mean(performance_folds)
  
    print(paste("Mean:", performance_mean, "Standard Deviation:", performance_se))

    if(visualize == TRUE){
      return(best_cs)
    }

    return(list(
        performance_mean = performance_mean, 
        performance_se = performance_se, 
        fold_results = fold_results ))
}


# nested cross validation
nested_fold_cv <- function(data, k = 10, model_fn, metric_fn, prob_flag = TRUE){

    n <- nrow(data)
    # shuffle before splitting
    data <- data[sample(1:n), ]
    folds <- stratified_folds(data$ShotType,k)

    performance_folds <- c() 

    # parameter defintion 
    c_values <- c(1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3)
    fold_results <- list()

         
    for(i in 1:k){

        # external test indicies 
        ext_test_indices <- folds[[i]]
        ext_train_indices <- setdiff(1:n, ext_test_indices)

        ext_train_data <- data[ext_train_indices, ]
        ext_test_data <- data[ext_test_indices, ]
        ext_test_target <- ext_test_data$ShotType
        ext_train_target <- ext_train_data$ShotType

        ext_test_data$ShotType <- NULL
        ext_train_data$ShotType <- NULL
        
        # inner loop 
        best_inner_performance <- -Inf
        best_c <- NULL

        for(c in c_values){

            inner_performance <- c()
            inner_predictions <- c()
    
            for(j in setdiff(1:k, i)){
                
                # internal test indices
                int_test_indices <- folds[[j]]
                int_train_indices <- setdiff(ext_train_indices, int_test_indices)
        
                int_train_data <- data[int_train_indices, ]
                int_test_data <- data[int_test_indices, ]

                int_test_target <- int_test_data$ShotType
                int_train_target <- int_train_data$ShotType
                int_test_data$ShotType <- NULL
                int_train_data$ShotType <- NULL
                predictions <- model_fn(int_train_data, int_train_target, int_test_data, int_test_target , c = c, prob_flag)
                performance <- metric_fn(int_test_target, predictions)
                inner_performance <- c(inner_performance, performance)
            }
            
            avg_inner_performance <- mean(inner_performance)
            if (best_inner_performance < avg_inner_performance){
                best_inner_performance <- avg_inner_performance
                best_c <- c
            }

        }
        

        predictions <- model_fn(ext_train_data, ext_train_target , ext_test_data, ext_test_target, c = best_c , prob_flag)
        performance <- metric_fn(ext_test_target, predictions)
        performance_folds <- c(performance_folds, performance)
        pred_errors <- as.integer(predictions != ext_test_target)

        fold_results[[i]] <- list(
            performance = performance,
            pred_errors = pred_errors,
            distance_val = ext_test_data$Distance,
            competition = ext_test_data$Competition
        )
    }

    performance_se <- sqrt(var(performance_folds)/k)
    performance_mean <- mean(performance_folds)

    print(paste("Mean:", performance_mean, "Standard Deviation:", performance_se))

    return(list(
        performance_mean = performance_mean, 
        performance_se = performance_se, 
        fold_results = fold_results ))
}



##################################### PART 1 ##################################################

data <- read.csv2("dataset.csv")
data <- transform_data(data)


### BASELINE 
result_k_baseline_acc <- k_fold_cv(data = data, k = 5, model_fn = baseline_classifier, metric_fn = accuracy, prob_flag = FALSE)
write.csv(convert_to_df(result_k_baseline_acc), "result_k_baseline_acc.csv", row.names = FALSE)
result_k_baseline_logscore <- k_fold_cv(data = data, k = 5, model_fn = baseline_classifier, metric_fn = log_loss)
write.csv(convert_to_df(result_k_baseline_logscore), "result_k_baseline_log_loss.csv", row.names = FALSE)

### LOGISTIC REGRESSION 

result_k_log_reg_acc <- k_fold_cv(data = data, k = 5, model_fn = logistic_regression, metric_fn = accuracy, prob_flag = FALSE)
write.csv(convert_to_df(result_k_log_reg_acc), "result_k_log_reg_accuracy.csv", row.names = FALSE)
result_k_log_reg_logscore <- k_fold_cv(data = data, k = 5, model_fn = logistic_regression, metric_fn = log_loss_multi, prob_flag = TRUE, log_reg_flag = TRUE)
write.csv(convert_to_df(result_k_log_reg_logscore), "result_k_log_reg_log_loss.csv", row.names = FALSE)

### SVM optimizing training fold 

opt_training_fold_per_logscore <- opt_training_fold(data = data, k = 5,model_fn = radial_svm, metric_fn = log_loss, prob_flag = FALSE)
write.csv(convert_to_df(opt_training_fold_per_logscore), "opt_training_fold_log_loss.csv", row.names = FALSE)
opt_training_fold_per_acc <- opt_training_fold(data = data, k = 5,model_fn = radial_svm, metric_fn = accuracy, prob_flag = TRUE)
write.csv(convert_to_df(opt_training_fold_per_acc), "opt_training_fold_acc.csv", row.names = FALSE)

### NESTED SVM 
results_svm_nestedcross_logscore <- nested_fold_cv(data = data, k = 5, model_fn = radial_svm, metric_fn = log_loss, prob_flag = FALSE)
write.csv(convert_to_df(results_svm_nestedcross_logscore), "nested_cv_results_10_log_loss.csv", row.names = FALSE)
results_svm_nestedcross_acc <- nested_fold_cv(data = data, k = 5,model_fn = radial_svm, metric_fn = accuracy, prob_flag = TRUE)
write.csv(convert_to_df(results_svm_nestedcross_acc), "nested_cv_results_acc.csv", row.names = FALSE)



################################# Estimate number of folds for CV ##########################################


# SVM optimizing training fold, using log score
results_svm_logloss <- data.frame(k = integer(0), 
                                        mean_performance = numeric(0),
                                        se_performance = numeric(0),
                                        time_taken = numeric(0))

for(k in seq(2, 25, by = 2)){

    start_time <- Sys.time() 
    opt_training_fold_per <- opt_training_fold(data = data, k = k,model_fn = radial_svm, metric_fn = log_loss)
    end_time <- Sys.time()

    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))

    performance_mean <- opt_training_fold_per$performance_mean
    performance_se <- opt_training_fold_per$performance_se
    performance_folds <- opt_training_fold_per$fold_results
    
    results_svm_logloss <- rbind(results_svm_logloss, data.frame(k = k, 
                                       mean_performance = performance_mean, 
                                       se_performance = performance_se, 
                                       time_taken = time_taken))
}

write.csv(results_svm_logloss, file = "results_k/results_svm_logloss.csv", row.names = FALSE)

# Baseline classifier, using log score
results_baseline_logloss <- data.frame(k = integer(0), 
                                        mean_performance = numeric(0),
                                        se_performance = numeric(0),
                                        time_taken = numeric(0))

for(k in seq(2, 25, by = 2)){

    start_time <- Sys.time() 
    result_k <- k_fold_cv(data = data, k = k, model_fn = baseline_classifier, metric_fn = log_loss)
    end_time <- Sys.time()

    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))

    performance_mean <- result_k$performance_mean
    performance_se <- result_k$performance_se
    performance_folds <- result_k$fold_results
    
    results_baseline_logloss <- rbind(results_baseline_logloss, data.frame(k = k, 
                                       mean_performance = performance_mean, 
                                       se_performance = performance_se, 
                                       time_taken = time_taken))

}

results_logreg_logloss <- data.frame(k = integer(0), 
                                        mean_performance = numeric(0),
                                        se_performance = numeric(0),
                                        time_taken = numeric(0))

for(k in seq(2, 25, by = 2)){

    start_time <- Sys.time() 
    result_k <- k_fold_cv(data = data, k = k, model_fn = logistic_regression, metric_fn = log_loss_multi)
    end_time <- Sys.time()

    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))

    performance_mean <- result_k$performance_mean
    performance_se <- result_k$performance_se
    performance_folds <- result_k$fold_results
    
    results_logreg_logloss <- rbind(results_logreg_logloss, data.frame(k = k, 
                                       mean_performance = performance_mean, 
                                       se_performance = performance_se, 
                                       time_taken = time_taken))

}

write.csv(results_logreg_logloss, file = "results_k/results_logreg_logloss.csv", row.names = FALSE)

svm_logloss_res <- read.csv("results_k/results_svm_logloss.csv", header = TRUE, sep = ",")
print(svm_logloss_res)
baseline_logloss_res <- read.csv("results_k/results_baseline_logloss.csv", header = TRUE, sep = ",")
print(baseline_logloss_res)
logreg_logloss_res <- read.csv("results_k/results_logreg_logloss.csv", header = TRUE, sep = ",")
print(logreg_logloss_res)

svm_logloss_res$model <- "SVM (Train Fold CV)"
logreg_logloss_res$model <- "Logistic Regression"
baseline_logloss_res$model <- "Baseline"
all_res <- bind_rows(svm_logloss_res, logreg_logloss_res, baseline_logloss_res)
 
all_res$time_minutes <- all_res$time_taken / 60


plot_perf <- ggplot(all_res, aes(x = k, y = mean_performance, color = model, fill = model)) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  labs(title = "Log Score vs Number of Folds",
       x = "Number of Folds (k)",
       y = "Mean Log Score") +
  theme_minimal() +
  theme(legend.position = "bottom")

plot_time <- ggplot(all_res, aes(x = k, y = time_minutes, color = model, fill = model)) +
  geom_line() + 
  labs(title = "Computation Time vs Number of Folds",
       x = "Number of Folds (k)",
       y = "Time (minutes)") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("results/estimate_k.pdf", 
       grid.arrange(plot_perf, plot_time, 
                    nrow = 1, ncol = 2),
       width = 16, height = 8, dpi = 300)
