# =============================================================================
# Bank Customer Churn Analysis
# Author: Samidha Basrur
# Date: February 2026
# Description: Predictive analysis of retail bank customer churn using
#              hypothesis testing, logistic regression, LASSO, and random forest
# Dataset: 10,000 bank customers | 14 variables | Binary outcome: Exited
# =============================================================================


# =============================================================================
# SECTION 1: LOAD LIBRARIES
# =============================================================================

library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
library(lsr)
library(pROC)
library(pscl)
library(glmnet)
library(randomForest)


# =============================================================================
# SECTION 2: LOAD & INSPECT DATA
# =============================================================================

churn_data <- read.csv("churn.csv")

# Structure and dimensions
dim(churn_data)
str(churn_data)
head(churn_data, 10)
summary(churn_data)


# =============================================================================
# SECTION 3: DATA QUALITY CHECKS
# =============================================================================

# Missing values
colSums(is.na(churn_data))
sum(is.na(churn_data))

# Duplicate customers
sum(duplicated(churn_data$CustomerId))

# Categorical variable distributions
table(churn_data$Geography)
table(churn_data$Gender)
table(churn_data$HasCrCard)
table(churn_data$IsActiveMember)
table(churn_data$NumOfProducts)
table(churn_data$Exited)


# =============================================================================
# SECTION 4: DATA PREPARATION
# =============================================================================

# Remove identifier columns (no predictive value)
churn_data <- churn_data %>% select(-RowNumber, -CustomerId, -Surname)

# Encode categorical variables as factors
churn_data$Geography <- as.factor(churn_data$Geography)
churn_data$Gender    <- as.factor(churn_data$Gender)

# Create binary balance variable (Zero vs Positive)
churn_data$Balance_Binary <- ifelse(churn_data$Balance == 0, 0, 1)

# Create age groups for segmentation analysis
churn_data$Age_Group <- cut(churn_data$Age,
                            breaks = c(0, 30, 40, 50, 60, 100),
                            labels = c("18-30", "31-40", "41-50", "51-60", "61+"))

# Train-test split (70/30, stratified by churn outcome)
set.seed(123)
train_index <- createDataPartition(churn_data$Exited, p = 0.7, list = FALSE)
train_data  <- churn_data[train_index, ]
test_data   <- churn_data[-train_index, ]

cat("Training set:", nrow(train_data), "| Churn rate:", round(mean(train_data$Exited) * 100, 1), "%\n")
cat("Testing set: ", nrow(test_data),  "| Churn rate:", round(mean(test_data$Exited)  * 100, 1), "%\n")


# =============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS
# =============================================================================

# --- 5.1 Overall Churn Rate ---

churn_summary <- churn_data %>%
  summarise(
    Total_Customers = n(),
    Churned         = sum(Exited),
    Retained        = sum(Exited == 0),
    Churn_Rate      = mean(Exited) * 100
  )
print(churn_summary)

churn_counts <- table(churn_data$Exited)
barplot(churn_counts,
        xlab = "Exited (0 = Stayed, 1 = Left)",
        ylab = "Count",
        col  = c("lightgreen", "lightcoral"),
        names.arg = c("Stayed (0)", "Left (1)"),
        ylim = c(0, max(churn_counts) + 1000))
title(main = "Customer Churn Distribution", adj = 0, line = 1, font.main = 2, cex.main = 1.3)
text(x = c(0.7, 1.9), y = churn_counts + 400, labels = churn_counts, cex = 1.2, font = 2)


# 5.2 Continuous Variable Distributions

continuous_vars <- churn_data %>%
  select(CreditScore, Age, Tenure, Balance, EstimatedSalary) %>%
  summary()
print(continuous_vars)

par(mfrow = c(2, 3))
hist(churn_data$CreditScore,     main = "Credit Score Distribution",    col = "lightblue")
hist(churn_data$Age,             main = "Age Distribution",             col = "lightgreen")
hist(churn_data$Tenure,          main = "Tenure Distribution",          col = "lightcoral")
hist(churn_data$Balance,         main = "Balance Distribution",         col = "lightyellow")
hist(churn_data$EstimatedSalary, main = "Estimated Salary Distribution",col = "lightpink")
par(mfrow = c(1, 1))

par(mfrow = c(2, 3))
boxplot(churn_data$CreditScore,     main = "Credit Score",     col = "lightblue")
boxplot(churn_data$Age,             main = "Age",              col = "lightgreen")
boxplot(churn_data$Tenure,          main = "Tenure",           col = "lightcoral")
boxplot(churn_data$Balance,         main = "Balance",          col = "lightyellow")
boxplot(churn_data$EstimatedSalary, main = "Estimated Salary", col = "lightpink")
par(mfrow = c(1, 1))


# 5.3 Balance Paradox: Zero vs Positive Balance 

zero_balance_count <- sum(churn_data$Balance == 0)
zero_balance_pct   <- mean(churn_data$Balance == 0) * 100
cat("Zero balance customers:", zero_balance_count, "(", round(zero_balance_pct, 1), "%)\n")

zero_balance_analysis <- churn_data %>%
  mutate(Balance_Category = ifelse(Balance == 0, "Zero Balance", "Positive Balance")) %>%
  group_by(Balance_Category) %>%
  summarise(
    Total     = n(),
    Churned   = sum(Exited),
    Churn_Rate = mean(Exited) * 100
  )
print(zero_balance_analysis)

barplot(zero_balance_analysis$Churn_Rate,
        names.arg = zero_balance_analysis$Balance_Category,
        main = "Churn Rate: Zero vs Positive Balance",
        ylab = "Churn Rate (%)",
        col  = c("lightcoral", "lightgreen"),
        ylim = c(0, max(zero_balance_analysis$Churn_Rate) + 10))
text(x = c(0.7, 1.9),
     y = zero_balance_analysis$Churn_Rate + 3,
     labels = paste0(round(zero_balance_analysis$Churn_Rate, 1), "%"), font = 2)


# 5.4 Churn by Categorical Variables

# Geography
churn_by_geography <- churn_data %>%
  group_by(Geography) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100)
print(churn_by_geography)

barplot(churn_by_geography$Churn_Rate,
        names.arg = churn_by_geography$Geography,
        main = "Churn Rate by Geography",
        ylab = "Churn Rate (%)",
        col  = c("steelblue", "coral", "gold"),
        ylim = c(0, max(churn_by_geography$Churn_Rate) + 5))
text(x = c(0.7, 1.9, 3.1),
     y = churn_by_geography$Churn_Rate + 1,
     labels = paste0(round(churn_by_geography$Churn_Rate, 1), "%"), font = 2)

# Gender
churn_by_gender <- churn_data %>%
  group_by(Gender) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100)
print(churn_by_gender)

barplot(churn_by_gender$Churn_Rate,
        names.arg = churn_by_gender$Gender,
        main = "Churn Rate by Gender",
        ylab = "Churn Rate (%)",
        col  = c("pink", "lightblue"),
        ylim = c(0, max(churn_by_gender$Churn_Rate) + 5))
text(x = c(0.7, 1.9),
     y = churn_by_gender$Churn_Rate + 1,
     labels = paste0(round(churn_by_gender$Churn_Rate, 1), "%"), font = 2)

# Number of Products
churn_by_products <- churn_data %>%
  group_by(NumOfProducts) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100)
print(churn_by_products)

barplot(churn_by_products$Churn_Rate,
        names.arg = churn_by_products$NumOfProducts,
        main = "Churn Rate by Number of Products",
        xlab = "Number of Products",
        ylab = "Churn Rate (%)",
        col  = rainbow(4),
        ylim = c(0, max(churn_by_products$Churn_Rate) + 10))
text(x = c(0.7, 1.9, 3.1, 4.3),
     y = churn_by_products$Churn_Rate + 2,
     labels = paste0(round(churn_by_products$Churn_Rate, 1), "%"))

# Activity Status
churn_by_active <- churn_data %>%
  group_by(IsActiveMember) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100)
print(churn_by_active)

barplot(churn_by_active$Churn_Rate,
        names.arg = c("Inactive (0)", "Active (1)"),
        main = "Churn Rate by Activity Status",
        ylab = "Churn Rate (%)",
        col  = c("lightcoral", "lightgreen"),
        ylim = c(0, max(churn_by_active$Churn_Rate) + 5))
text(x = c(0.7, 1.9),
     y = churn_by_active$Churn_Rate + 1,
     labels = paste0(round(churn_by_active$Churn_Rate, 1), "%"), font = 2)

# Credit Card Ownership
churn_by_card <- churn_data %>%
  group_by(HasCrCard) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100)
print(churn_by_card)

barplot(churn_by_card$Churn_Rate,
        names.arg = c("No Card (0)", "Has Card (1)"),
        main = "Churn Rate by Credit Card Ownership",
        ylab = "Churn Rate (%)",
        col  = c("lightcoral", "lightgreen"),
        ylim = c(0, max(churn_by_card$Churn_Rate) + 5))
text(x = c(0.7, 1.9),
     y = churn_by_card$Churn_Rate + 1,
     labels = paste0(round(churn_by_card$Churn_Rate, 1), "%"), font = 2)


# 5.5 Continuous Variables vs Churn 

comparison <- churn_data %>%
  group_by(Exited) %>%
  summarise(
    Count          = n(),
    Avg_CreditScore = mean(CreditScore),
    Avg_Age        = mean(Age),
    Avg_Tenure     = mean(Tenure),
    Avg_Balance    = mean(Balance),
    Avg_Salary     = mean(EstimatedSalary)
  )
print(comparison)

par(mfrow = c(2, 3))
boxplot(CreditScore     ~ Exited, data = churn_data, main = "Credit Score by Churn",
        names = c("Retained", "Churned"), col = c("lightgreen", "lightcoral"))
boxplot(Age             ~ Exited, data = churn_data, main = "Age by Churn",
        names = c("Retained", "Churned"), col = c("lightgreen", "lightcoral"))
boxplot(Tenure          ~ Exited, data = churn_data, main = "Tenure by Churn",
        names = c("Retained", "Churned"), col = c("lightgreen", "lightcoral"))
boxplot(Balance         ~ Exited, data = churn_data, main = "Balance by Churn",
        names = c("Retained", "Churned"), col = c("lightgreen", "lightcoral"))
boxplot(EstimatedSalary ~ Exited, data = churn_data, main = "Salary by Churn",
        names = c("Retained", "Churned"), col = c("lightgreen", "lightcoral"))
par(mfrow = c(1, 1))


# 5.6 Balance x Activity Status Interaction 

balance_activity <- churn_data %>%
  mutate(Balance_Cat = ifelse(Balance == 0, "Zero Balance", "Positive Balance")) %>%
  group_by(Balance_Cat, IsActiveMember) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100, .groups = "drop")
print(balance_activity)

churn_matrix <- matrix(balance_activity$Churn_Rate, nrow = 2, ncol = 2)
colnames(churn_matrix) <- c("Positive Balance", "Zero Balance")
rownames(churn_matrix) <- c("Inactive", "Active")

barplot(churn_matrix,
        beside = TRUE,
        main   = "Churn Rate: Balance x Activity Status",
        ylab   = "Churn Rate (%)",
        col    = c("lightcoral", "lightgreen"),
        legend = rownames(churn_matrix),
        ylim   = c(0, max(churn_matrix) + 10))


# 5.7 Correlation Matrix

cor_data <- churn_data %>%
  select(CreditScore, Age, Tenure, Balance, EstimatedSalary,
         NumOfProducts, HasCrCard, IsActiveMember, Exited)

cor_matrix <- cor(cor_data)
print(round(cor_matrix, 3))

corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.6,
         title = "Correlation Matrix", mar = c(0, 0, 2, 0))


# 5.8 Churn by Age Group

churn_by_age_group <- churn_data %>%
  group_by(Age_Group) %>%
  summarise(Total = n(), Churned = sum(Exited), Churn_Rate = mean(Exited) * 100)
print(churn_by_age_group)

barplot(churn_by_age_group$Churn_Rate,
        names.arg = churn_by_age_group$Age_Group,
        main = "Churn Rate by Age Group",
        ylab = "Churn Rate (%)",
        col  = terrain.colors(5),
        ylim = c(0, max(churn_by_age_group$Churn_Rate) + 5))
text(x = seq(0.7, by = 1.2, length.out = 5),
     y = churn_by_age_group$Churn_Rate + 2,
     labels = paste0(round(churn_by_age_group$Churn_Rate, 1), "%"), font = 2)


# =============================================================================
# SECTION 6: HYPOTHESIS TESTING
# =============================================================================

# 6.1 Independent T-Tests (Continuous Variables)

t_credit  <- t.test(CreditScore     ~ Exited, data = churn_data)
t_age     <- t.test(Age             ~ Exited, data = churn_data)
t_tenure  <- t.test(Tenure          ~ Exited, data = churn_data)
t_balance <- t.test(Balance         ~ Exited, data = churn_data)
t_salary  <- t.test(EstimatedSalary ~ Exited, data = churn_data)

t_summary <- data.frame(
  Variable      = c("CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"),
  Mean_Retained = c(t_credit$estimate[1],  t_age$estimate[1],    t_tenure$estimate[1],
                    t_balance$estimate[1], t_salary$estimate[1]),
  Mean_Churned  = c(t_credit$estimate[2],  t_age$estimate[2],    t_tenure$estimate[2],
                    t_balance$estimate[2], t_salary$estimate[2]),
  t_statistic   = c(t_credit$statistic,    t_age$statistic,      t_tenure$statistic,
                    t_balance$statistic,   t_salary$statistic),
  p_value       = c(t_credit$p.value,      t_age$p.value,        t_tenure$p.value,
                    t_balance$p.value,     t_salary$p.value),
  Significant   = c(t_credit$p.value < 0.05, t_age$p.value < 0.05, t_tenure$p.value < 0.05,
                    t_balance$p.value < 0.05, t_salary$p.value < 0.05)
)
print(t_summary)


# 6.2 Chi-Square Tests (Categorical Variables)

chi_geo     <- chisq.test(table(churn_data$Geography,     churn_data$Exited))
chi_gender  <- chisq.test(table(churn_data$Gender,        churn_data$Exited))
chi_products<- chisq.test(table(churn_data$NumOfProducts, churn_data$Exited))
chi_card    <- chisq.test(table(churn_data$HasCrCard,     churn_data$Exited))
chi_active  <- chisq.test(table(churn_data$IsActiveMember,churn_data$Exited))

cramer_geo      <- cramersV(churn_data$Geography,      churn_data$Exited)
cramer_gender   <- cramersV(churn_data$Gender,         churn_data$Exited)
cramer_products <- cramersV(churn_data$NumOfProducts,  churn_data$Exited)
cramer_card     <- cramersV(churn_data$HasCrCard,      churn_data$Exited)
cramer_active   <- cramersV(churn_data$IsActiveMember, churn_data$Exited)

chi_summary <- data.frame(
  Variable    = c("Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"),
  Chi_Square  = c(chi_geo$statistic,  chi_gender$statistic,  chi_products$statistic,
                  chi_card$statistic, chi_active$statistic),
  df          = c(chi_geo$parameter,  chi_gender$parameter,  chi_products$parameter,
                  chi_card$parameter, chi_active$parameter),
  p_value     = c(chi_geo$p.value,    chi_gender$p.value,    chi_products$p.value,
                  chi_card$p.value,   chi_active$p.value),
  Cramers_V   = c(cramer_geo, cramer_gender, cramer_products, cramer_card, cramer_active),
  Significant = c(chi_geo$p.value < 0.05,  chi_gender$p.value < 0.05,
                  chi_products$p.value < 0.05, chi_card$p.value < 0.05,
                  chi_active$p.value < 0.05)
)
print(chi_summary)


# 6.3 ANOVA Tests 

# Geography
anova_geo <- aov(Exited ~ Geography, data = churn_data)
print(summary(anova_geo))
print(TukeyHSD(anova_geo))

# Number of Products
anova_products <- aov(Exited ~ as.factor(NumOfProducts), data = churn_data)
print(summary(anova_products))
print(TukeyHSD(anova_products))

# Age Group
anova_age <- aov(Exited ~ Age_Group, data = churn_data)
print(summary(anova_age))
print(TukeyHSD(anova_age))


# 6.4 Effect Sizes (Cohen's d) 

library(effsize)

cohen_age     <- cohen.d(churn_data$Age         ~ as.factor(churn_data$Exited))
cohen_balance <- cohen.d(churn_data$Balance      ~ as.factor(churn_data$Exited))
cohen_credit  <- cohen.d(churn_data$CreditScore  ~ as.factor(churn_data$Exited))

effect_summary <- data.frame(
  Variable  = c("Age", "Balance", "CreditScore"),
  Cohens_d  = c(cohen_age$estimate, cohen_balance$estimate, cohen_credit$estimate),
  Magnitude = c(as.character(cohen_age$magnitude), as.character(cohen_balance$magnitude),
                as.character(cohen_credit$magnitude))
)
print(effect_summary)


# =============================================================================
# SECTION 7: LOGISTIC REGRESSION
# =============================================================================

# 7.1 Model 1: Main Effects

model1 <- glm(Exited ~ CreditScore + Age + Tenure + Balance_Binary +
                EstimatedSalary + Geography + Gender + NumOfProducts +
                HasCrCard + IsActiveMember,
              family = binomial(link = "logit"),
              data   = train_data)

print(summary(model1))

# Odds ratios with confidence intervals
or_table1 <- data.frame(
  Variable  = names(exp(coef(model1))),
  Odds_Ratio = exp(coef(model1)),
  CI_Lower  = exp(confint(model1))[, 1],
  CI_Upper  = exp(confint(model1))[, 2]
)
print(or_table1)


# 7.2 Model 1 Evaluation

train_pred_prob1  <- predict(model1, type = "response")
train_pred_class1 <- ifelse(train_pred_prob1 > 0.5, 1, 0)
test_pred_prob1   <- predict(model1, newdata = test_data, type = "response")
test_pred_class1  <- ifelse(test_pred_prob1 > 0.5, 1, 0)

conf_matrix_train1 <- confusionMatrix(as.factor(train_pred_class1), as.factor(train_data$Exited), positive = "1")
conf_matrix_test1  <- confusionMatrix(as.factor(test_pred_class1),  as.factor(test_data$Exited),  positive = "1")
print(conf_matrix_train1)
print(conf_matrix_test1)

roc_train1 <- roc(train_data$Exited, train_pred_prob1)
roc_test1  <- roc(test_data$Exited,  test_pred_prob1)
cat("Train AUC:", auc(roc_train1), "| Test AUC:", auc(roc_test1), "\n")

plot(roc_test1, main = "ROC Curve - Model 1 (Test Set)", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.6, 0.3, paste("AUC =", round(auc(roc_test1), 3)), cex = 1.2)

print(pR2(model1))


# 7.3 Model 2: With Interaction Term (Balance x Activity) 

model2 <- glm(Exited ~ CreditScore + Age + Tenure + Balance_Binary +
                EstimatedSalary + Geography + Gender + NumOfProducts +
                HasCrCard + IsActiveMember +
                Balance_Binary:IsActiveMember,
              family = binomial(link = "logit"),
              data   = train_data)

print(summary(model2))
print(exp(coef(model2)))

test_pred_prob2  <- predict(model2, newdata = test_data, type = "response")
test_pred_class2 <- ifelse(test_pred_prob2 > 0.5, 1, 0)

conf_matrix_test2 <- confusionMatrix(as.factor(test_pred_class2), as.factor(test_data$Exited), positive = "1")
print(conf_matrix_test2)

roc_test2 <- roc(test_data$Exited, test_pred_prob2)
print(pR2(model2))


# 7.4 Model Comparison: Model 1 vs Model 2

print(anova(model1, model2, test = "Chisq"))

comparison_table <- data.frame(
  Metric    = c("Testing Accuracy", "Testing AUC", "McFadden R²", "AIC", "BIC"),
  Model_1   = c(conf_matrix_test1$overall["Accuracy"], auc(roc_test1),
                pR2(model1)["McFadden"], AIC(model1), BIC(model1)),
  Model_2   = c(conf_matrix_test2$overall["Accuracy"], auc(roc_test2),
                pR2(model2)["McFadden"], AIC(model2), BIC(model2))
)
print(comparison_table)

# ROC curve comparison
plot(roc_test1, col = "blue", lwd = 2, main = "ROC Curves: Model 1 vs Model 2")
lines(roc_test2, col = "red", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Model 1 (AUC =", round(auc(roc_test1), 3), ")"),
                  paste("Model 2 (AUC =", round(auc(roc_test2), 3), ")")),
       col = c("blue", "red"), lwd = 2)


# =============================================================================
# SECTION 8: LASSO REGRESSION
# =============================================================================

#  8.1 Prepare Matrices 

x_train <- model.matrix(Exited ~ CreditScore + Age + Tenure + Balance_Binary +
                          EstimatedSalary + Geography + Gender + NumOfProducts +
                          HasCrCard + IsActiveMember,
                        data = train_data)[, -1]
y_train <- train_data$Exited

x_test  <- model.matrix(Exited ~ CreditScore + Age + Tenure + Balance_Binary +
                          EstimatedSalary + Geography + Gender + NumOfProducts +
                          HasCrCard + IsActiveMember,
                        data = test_data)[, -1]
y_test  <- test_data$Exited


# 8.2 Cross-Validation to Find Optimal Lambda 

set.seed(123)
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, nfolds = 10)
plot(cv_lasso, main = "LASSO Cross-Validation: Deviance vs Lambda")

best_lambda <- cv_lasso$lambda.min
cat("Optimal lambda (lambda.min):", best_lambda, "\n")
cat("Lambda within 1 SE (lambda.1se):", cv_lasso$lambda.1se, "\n")


#  8.3 Fit Final LASSO Model 

lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

lasso_coef        <- coef(lasso_model)
lasso_coef_matrix <- as.matrix(lasso_coef)
non_zero_vars     <- lasso_coef_matrix[lasso_coef_matrix[, 1] != 0, , drop = FALSE]

print(non_zero_vars)
cat("Variables selected:", nrow(non_zero_vars) - 1, "out of", ncol(x_train), "\n")


#  8.4 LASSO Coefficient Path Plot 

lasso_path <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
plot(lasso_path, xvar = "lambda",
     main = "LASSO Regularization: Coefficient Shrinkage Path",
     xlab = expression(paste("Log(", lambda, ")")),
     ylab = "Standardized Coefficients",
     lwd  = 2.5,
     col  = rainbow(11, alpha = 0.7))
grid(col = "gray90", lty = 1)
abline(v = log(best_lambda), col = "red", lty = 2, lwd = 3)
text(log(best_lambda) + 0.5, 0.7,
     paste0("Optimal λ\n", round(best_lambda, 5)), cex = 0.85, col = "red", font = 2)


# 8.5 LASSO Predictions and Performance

lasso_train_pred_prob  <- predict(lasso_model, newx = x_train, type = "response")
lasso_train_pred_class <- ifelse(lasso_train_pred_prob > 0.5, 1, 0)
lasso_test_pred_prob   <- predict(lasso_model, newx = x_test,  type = "response")
lasso_test_pred_class  <- ifelse(lasso_test_pred_prob  > 0.5, 1, 0)

lasso_conf_train <- confusionMatrix(as.factor(lasso_train_pred_class), as.factor(y_train), positive = "1")
lasso_conf_test  <- confusionMatrix(as.factor(lasso_test_pred_class),  as.factor(y_test),  positive = "1")
print(lasso_conf_train)
print(lasso_conf_test)

lasso_roc_train <- roc(y_train, as.numeric(lasso_train_pred_prob))
lasso_roc_test  <- roc(y_test,  as.numeric(lasso_test_pred_prob))
cat("Train AUC:", auc(lasso_roc_train), "| Test AUC:", auc(lasso_roc_test), "\n")

plot(lasso_roc_test, col = "darkgreen", lwd = 2, main = "ROC Curve - LASSO (Test Set)")
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.6, 0.3, paste("AUC =", round(auc(lasso_roc_test), 3)), cex = 1.2)


#  8.6 LASSO Variable Importance 

lasso_importance <- data.frame(
  Variable        = rownames(lasso_coef)[-1],
  Coefficient     = as.numeric(lasso_coef[-1, ]),
  Abs_Coefficient = abs(as.numeric(lasso_coef[-1, ]))
)
lasso_importance <- lasso_importance[lasso_importance$Coefficient != 0, ]
lasso_importance <- lasso_importance[order(-lasso_importance$Abs_Coefficient), ]
print(lasso_importance)

par(mar = c(8, 4, 4, 2))
barplot(lasso_importance$Abs_Coefficient,
        names.arg = lasso_importance$Variable,
        las  = 2,
        col  = "darkgreen",
        main = "LASSO Variable Importance",
        ylab = "Absolute Coefficient Value",
        cex.names = 0.5)
par(mar = c(5, 4, 4, 2))


# =============================================================================
# SECTION 9: RANDOM FOREST
# =============================================================================

# 9.1 Prepare Data 

train_data_rf <- train_data
test_data_rf  <- test_data
train_data_rf$Exited_factor <- as.factor(train_data_rf$Exited)
test_data_rf$Exited_factor  <- as.factor(test_data_rf$Exited)


#  9.2 Build Random Forest Model

set.seed(123)
rf_model <- randomForest(Exited_factor ~ CreditScore + Age + Tenure + Balance_Binary +
                           EstimatedSalary + Geography + Gender + NumOfProducts +
                           HasCrCard + IsActiveMember,
                         data       = train_data_rf,
                         ntree      = 500,
                         importance = TRUE,
                         mtry       = 3)
print(rf_model)


# 9.3 Variable Importance 

importance_scores <- importance(rf_model)
print(importance_scores)

varImpPlot(rf_model,
           main = "Random Forest Variable Importance",
           pch  = 19, col = "darkblue", cex = 0.7)

importance_df <- data.frame(
  Variable             = rownames(importance_scores),
  MeanDecreaseAccuracy = importance_scores[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini     = importance_scores[, "MeanDecreaseGini"]
)
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]
print(importance_df)


#  9.4 Predictions and Performance

rf_train_pred <- predict(rf_model, train_data_rf, type = "class")
rf_train_prob <- predict(rf_model, train_data_rf, type = "prob")[, 2]
rf_test_pred  <- predict(rf_model, test_data_rf,  type = "class")
rf_test_prob  <- predict(rf_model, test_data_rf,  type = "prob")[, 2]

rf_conf_train <- confusionMatrix(rf_train_pred, train_data_rf$Exited_factor, positive = "1")
rf_conf_test  <- confusionMatrix(rf_test_pred,  test_data_rf$Exited_factor,  positive = "1")
print(rf_conf_train)
print(rf_conf_test)

rf_roc_train <- roc(train_data_rf$Exited, rf_train_prob)
rf_roc_test  <- roc(test_data_rf$Exited,  rf_test_prob)
cat("Train AUC:", auc(rf_roc_train), "| Test AUC:", auc(rf_roc_test), "\n")

plot(rf_roc_test, col = "purple", lwd = 2, main = "ROC Curve - Random Forest (Test Set)")
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.6, 0.3, paste("AUC =", round(auc(rf_roc_test), 3)), cex = 1.2)


# =============================================================================
# SECTION 10: FINAL MODEL COMPARISON
# =============================================================================

final_comparison <- data.frame(
  Metric         = c("Training Accuracy", "Testing Accuracy", "Testing AUC",
                     "Testing Sensitivity", "Testing Specificity", "Testing Precision"),
  Logistic       = c(conf_matrix_train1$overall["Accuracy"],
                     conf_matrix_test1$overall["Accuracy"],
                     auc(roc_test1),
                     conf_matrix_test1$byClass["Sensitivity"],
                     conf_matrix_test1$byClass["Specificity"],
                     conf_matrix_test1$byClass["Pos Pred Value"]),
  LASSO          = c(lasso_conf_train$overall["Accuracy"],
                     lasso_conf_test$overall["Accuracy"],
                     auc(lasso_roc_test),
                     lasso_conf_test$byClass["Sensitivity"],
                     lasso_conf_test$byClass["Specificity"],
                     lasso_conf_test$byClass["Pos Pred Value"]),
  Random_Forest  = c(rf_conf_train$overall["Accuracy"],
                     rf_conf_test$overall["Accuracy"],
                     auc(rf_roc_test),
                     rf_conf_test$byClass["Sensitivity"],
                     rf_conf_test$byClass["Specificity"],
                     rf_conf_test$byClass["Pos Pred Value"])
)
print(final_comparison)

# Best model per metric
for (i in 1:nrow(final_comparison)) {
  values     <- final_comparison[i, 2:4]
  best_model <- names(values)[which.max(values)]
  cat(final_comparison$Metric[i], "→ Best:", best_model,
      "=", round(max(values), 4), "\n")
}

# All three ROC curves
plot(roc_test1, col = "blue", lwd = 2, main = "ROC Curves: All Models (Test Set)")
lines(lasso_roc_test, col = "darkgreen", lwd = 2)
lines(rf_roc_test,    col = "purple",    lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Logistic     (AUC =", round(auc(roc_test1),       3), ")"),
                  paste("LASSO        (AUC =", round(auc(lasso_roc_test),  3), ")"),
                  paste("Random Forest(AUC =", round(auc(rf_roc_test),     3), ")")),
       col = c("blue", "darkgreen", "purple"), lwd = 2, cex = 0.8)

# Top predictors across all three methods
cat("\n=== TOP PREDICTORS ACROSS ALL MODELS ===\n")
cat("Logistic Regression  : Age, IsActiveMember, GeographyGermany, GenderMale, Balance_Binary\n")
cat("LASSO                : (see lasso_importance table above)\n")
cat("Random Forest        : (see importance_df table above)\n")

# =============================================================================
# END OF ANALYSIS
# =============================================================================