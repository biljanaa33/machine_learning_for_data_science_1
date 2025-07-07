library("e1071")
library("caret")
library("ggplot2")
library("mgcv")  
library("gridExtra")


############################## PREDICTION ERROR DEPENDENCE ON DISTANCE #####################################

set.seed(42)
error_prediction <- function(df_model){

  df_model$distance_val <- as.numeric(df_model$distance_val)
  colnames(df_model) <- c("distance_val","class","competition_type")
  gam_model <- gam(class ~ s(distance_val), data = df_model, family = binomial)
  new_data <- data.frame(distance_val = seq(min(df_model$distance_val), 
                                          max(df_model$distance_val), 
                                          length.out = 100))
  predictions <- predict(gam_model, newdata = new_data, type = "link", se.fit = TRUE)
  new_data$predicted_prob <- plogis(predictions$fit)
  new_data$lower_ci <- plogis(predictions$fit - 1.96 * predictions$se.fit)
  new_data$upper_ci <- plogis(predictions$fit + 1.96 * predictions$se.fit)

  return(new_data)

}

plot_prediction <- function(new_data, title){

  p <- ggplot(new_data, aes(x = distance_val, y = predicted_prob)) +
  geom_line(color = "blue") +  
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.2, fill = "blue") +  
  labs(title = title,
       x = "Distance",
       y = "Predicted Probability of Error (Class = 1)") +
  theme_minimal()
  ggsave("results/baseline_k_accuracy.pdf", plot = p, width = 8, height = 6, dpi = 300)

  return(p)
}

df_baseline_acc <- read.csv("results/result_k_baseline_acc.csv", header = TRUE, sep = ",")
df_baseline_acc_data <- error_prediction(df_baseline_acc)
p1 <- plot_prediction(df_baseline_acc_data, "Baseline Classifier - Accuracy")

df_baseline_logloss <- read.csv("results/result_k_baseline_log_loss.csv", header = TRUE, sep = ",")
df_baseline_logloss_data <- error_prediction(df_baseline_logloss)
p2 <- plot_prediction(df_baseline_logloss_data, "Baseline Classifier - Log-score")

df_logisticregression_acc <- read.csv("results/result_k_log_reg_accuracy.csv", header = TRUE, sep = ",")
df_logisticregression_acc_data <- error_prediction(df_logisticregression_acc)
p3 <- plot_prediction(df_logisticregression_acc_data, "Logistic Regression - Accuracy")


df_logisticregression_logloss <- read.csv("results/result_k_log_reg_log_loss.csv", header = TRUE, sep = ",")
df_logisticregression_logloss_log_score <- read.csv("results/result_k_log_reg_log_score.csv", header = TRUE, sep = ",")
df_logisticregression_logloss_data <- error_prediction(df_logisticregression_logloss)
p4 <- plot_prediction(df_logisticregression_logloss_data, "Logistic Regression - Log-score")


df_svm_trainingfold_acc <- read.csv("results/opt_training_fold_acc.csv", header = TRUE, sep = ",")
df_svm_trainingfold_acc_data <- error_prediction(df_svm_trainingfold_acc)
p5 <- plot_prediction(df_svm_trainingfold_acc_data, "SVM(Train Fold CV) - Accuracy")


df_svm_trainingfold_logloss <- read.csv("results/opt_training_fold_log_loss.csv", header = TRUE, sep = ",")
df_svm_trainingfold_logloss_data <- error_prediction(df_svm_trainingfold_logloss)
p6 <- plot_prediction(df_svm_trainingfold_logloss_data, "SVM(Train Fold CV) - Log-score")

df_svm_nested_acc <- read.csv("results/nested_cv_results_acc.csv", header = TRUE, sep = ",")
df_svm_nested_acc_data <- error_prediction(df_svm_nested_acc,"SVM(Nested CV) - Accuracy" )
p7 <- plot_prediction(df_svm_nested_acc_data)

df_svm_nested_logloss <- read.csv("results/nested_cv_results_log_loss.csv", header = TRUE, sep = ",")
df_svm_nested_logloss_data <- error_prediction(df_svm_nested_logloss)
p8 <- plot_prediction(df_svm_nested_logloss_data, "SVM(Nested CV) - Log-score") 

ggsave("results/all_models_grid.pdf", 
       grid.arrange(p1, p2, p3, p4,
                    p5, p6, p7, p8,
                    nrow = 2, ncol = 4), 
       width = 16, height = 8, dpi = 300)







########################## REWEIGHTED ERROR BASED ON TRUE COMPETITION TYPE DSITRIBUTION ##################

true_weights <- c("NBA" = 0.6, "EURO" = 0.1, "SLO1" = 0.1, "U14" = 0.1, "U16" = 0.1)

# mapping "EURO":1 "NBA":2  "SLO1":3 "U14":4  "U16":5
# competion weights

weighted_logloss_with_se <- function(df, true_weights, n_boot = 1000, prob = FALSE) {
  eps <- 1e-15
  
  df$competition <- factor(df$competition,
                                levels = c(1, 2, 3, 4, 5),
                                labels = c("EURO", "NBA", "SLO1", "U14", "U16"))

  if(prob == TRUE){

    acc <- df$preds_error

  } else {

    acc <- 1 - df$preds_error
    acc <- pmin(pmax(acc, eps), 1 - eps)  

  }
  df$class <- log(acc)

  # aggregate by competition and apply weights
  loss_table <- aggregate(class ~ competition, data = df, FUN = mean)
  # for each competion type get corresponing weight
  loss_table$weight <- true_weights[as.character(loss_table$competition)]
  weighted_loss <- sum(loss_table$class * loss_table$weight)

  # bootstrap SE
  boot_loss <- numeric(n_boot)
  for (i in 1:n_boot) {
    boot_data <- df[sample(nrow(df), replace = TRUE), ]
    loss <- aggregate(class ~ competition, data = boot_data, FUN = mean)
    loss$weight <- true_weights[as.character(loss$competition)]
    boot_loss[i] <- sum(loss$class * loss$weight)
  }

  se <- sd(boot_loss)
  list(logloss = weighted_loss, se = se)
}


weighted_accuracy_with_se <- function(df, true_weights, n_boot = 1000) {

  df$competition <- factor(df$competition, levels = c(1,2,3,4,5),
                                labels = c("EURO", "NBA", "SLO1", "U14", "U16"))


  # compute weighted accuracy on original data
  df$class <- 1 - df$preds_error
  acc_table <- aggregate(class ~ competition, data = df, FUN = function(x)  mean(x))
  acc_table$weight <- true_weights[as.character(acc_table$competition)]
  weighted_acc <- sum(acc_table$class * acc_table$weight)

  # bootstrap stratifed based on the frquencies of competition
  boot_acc <- numeric(n_boot)
  for (i in 1:n_boot) {
    boot_data <- do.call(rbind, lapply(split(df, df$competition), function(group) {
      group[sample(nrow(group), replace = TRUE), ]
    }))
    
    acc <- aggregate(class ~ competition, data = boot_data, FUN = function(x) mean(x))
    acc$weight <- true_weights[as.character(acc$competition)]
    boot_acc[i] <- sum(acc$class * acc$weight)
  }

  se <- sd(boot_acc)
  list(acc = weighted_acc, se = se)
}



### ACCURACY 
weighted_accuracy_with_se(df_baseline_acc, true_weights)
weighted_accuracy_with_se(df_logisticregression_acc, true_weights)
weighted_accuracy_with_se(df_svm_trainingfold_acc, true_weights)
weighted_accuracy_with_se(df_svm_nested_acc, true_weights)


### LOG SCORE 
weighted_logloss_with_se(df_baseline_logloss, true_weights)
# for log reg use only the probability 
weighted_logloss_with_se(df_logisticregression_logloss_log_score, true_weights, prob = TRUE)
weighted_logloss_with_se(df_svm_trainingfold_logloss,true_weights)
weighted_logloss_with_se(df_svm_nested_logloss, true_weights)
