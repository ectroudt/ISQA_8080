

library(ISLR)
library(GGally)
library(tidyverse)
library(caret)
library(Hmisc)
library(e1071)
library(pROC)
library(RANN)
library(DMwR)
library(kernlab)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(doParallel)
num_cores <- detectCores()
registerDoParallel(num_cores)

#stopImplicitCluster()


######################
model_data <- read.csv("Model Dataset for Students.csv")


firm_Data <- read.csv("Firmographic Data for Students.csv")

model_data$churned <- as.factor(model_data$churned)

model_data <- model_data %>% mutate(churned, churned = recode(churned, '0' = 'No', '1' = 'Yes'))

model_data[["churned"]] <-relevel(model_data[["churned"]], "Yes")

model_data$Company_Creation_Date <- as.character(model_data$Company_Creation_Date)

model_data$Company_Creation_Date <- dmy_hms(model_data$Company_Creation_Date)

merge_Data <- merge(model_data, firm_Data, by.x = 'Company_Number', by.y = 'Company_Number')

model_Variables <- c("Business_Code", "Location_Type", "churned", "total_products", 
                     "total_transactions", "total_accounts", "total_revenue", "total_usage", 
                     "Employee_Count_Total", "Company_Number", "Public_Private_Indicator", 
                     "Small_Business_Indicator", "Minority_Owned_Indicator", "Major_Industry_Category_Name", 
                     "Number_of_Family_Members", "BEMFAB__Marketability_", "Legal_Status_Code", "Manufacturing_Indicator")

merge_Data_subset <- subset(merge_Data, select = model_Variables)


merge_Data_subset <- merge_Data_subset %>% 
  mutate(BEMFAB__Marketability_ = case_when(
    BEMFAB__Marketability_ == "M" ~ "Matched_Full_Marketing",
    BEMFAB__Marketability_ == "N" ~ "Unmatched",
    BEMFAB__Marketability_ == "X" ~ "Matched_NonMarketing",
    BEMFAB__Marketability_ == "A" ~ "Undeliverable",
    BEMFAB__Marketability_ == "O" ~ "Out_Business",
    BEMFAB__Marketability_ == "S" ~ "Undetermined_SIC",
    BEMFAB__Marketability_ == "D" ~ "Delisted_Record"
    
  )
  )


merge_Data_subset <- merge_Data_subset %>% 
  mutate(Legal_Status_Code = case_when(
    Legal_Status_Code == 3 ~ "corporation",
    Legal_Status_Code == 8 ~ "joint_venture",
    Legal_Status_Code == 12 ~ "partnership_of_unknown_type",
    Legal_Status_Code == 13 ~ "proprietorship",
    Legal_Status_Code == 50 ~ "government_body",
    Legal_Status_Code == 100 ~ "cooperative",
    Legal_Status_Code == 101 ~ "non_profit_organization",
    Legal_Status_Code == 118 ~ "local_government_body",
    Legal_Status_Code == 120 ~ "foreign company",
    TRUE ~ "Other"
  )
  )


#merge_Data_subset <- merge_Data_subset %>% 
  mutate(Population_Code = case_when(
    Population_Code == 0 ~ "Under_1000",
    Population_Code == 1 ~ "1000_to_2499",
    Population_Code == 2 ~ "2500_to_4999",
    Population_Code == 3 ~ "5000_to_9999",
    Population_Code == 4 ~ "10000_to_24999",
    Population_Code == 5 ~ "25000_to_49999",
    Population_Code == 6 ~ "50000_to_99999",
    Population_Code == 7 ~ "100000_to_249999",
    Population_Code == 8 ~ "250000_to_499999",
    Population_Code == 9 ~ "500000_and_over"
  )
  )

merge_Data_subset <- merge_Data_subset %>% 
  mutate(Manufacturing_Indicator = case_when(
    Manufacturing_Indicator == 0 ~ "Y",
    Manufacturing_Indicator == 1 ~ "N"
  )
  )


names(merge_Data_subset) <- make.names(names(merge_Data_subset))


merge_Data_subset <- merge_Data_subset %>% mutate_all(na_if, "")


merge_Data_subset_NUM <- merge_Data_subset %>% select_if(is.numeric)

merge_Data_subset_nonNUM <- merge_Data_subset %>% select_if(~!is.numeric(.x))

for(Var in colnames(merge_Data_subset_nonNUM)) {
  
  merge_Data_subset_nonNUM[[Var]] <- addNA(merge_Data_subset_nonNUM[[Var]], ifany = TRUE)
  merge_Data_subset_nonNUM[[Var]] <- droplevels(merge_Data_subset_nonNUM[[Var]])
  
}

(high_Correlation <- findCorrelation(cor(merge_Data_subset_NUM)))


# impute missing values using bag approach
pre_Imputed_model_Data <- preProcess(merge_Data_subset_NUM, method = c("bagImpute", "scale", "center"))
imputed_model_Data <- predict(pre_Imputed_model_Data, merge_Data_subset_NUM)

(high_Correlation <- findCorrelation(cor(imputed_model_Data)))

model_data_bagImputed<- cbind(merge_Data_subset_nonNUM, imputed_model_Data)

chrned_Data <- subset(merge_Data, merge_Data[,"churned"] == "Yes")


summary(model_data_bagImputed)

#ggpairs(model_data_bagImputed)


#ggpairs(model_data_bagImputed, columns = c(3, 8), aes(color = churned))
#ggpairs(model_data_bagImputed, columns = c(4, 8), aes(color = churned))
#ggpairs(model_data_bagImputed, columns = c(5, 8), aes(color = churned))
#ggpairs(model_data_bagImputed, columns = c(6, 8), aes(color = churned))
#ggpairs(model_data_bagImputed, columns = c(7, 8), aes(color = churned))
#ggpairs(model_data_bagImputed, columns = c(9, 8), aes(color = churned))

write.csv(model_data_bagImputed, file = "model_data_bagImputed.csv")

dataset=model_data_bagImputed

dataset=dataset %>% select(-c(17))
dataset=dataset %>% select(-c(6))
dataset=dataset %>% select(-c(5))

dataset <- dataset %>% mutate_if(is.character, as.factor)

set.seed(sample(10000,1))

target_var='churned'

trainIndex <- createDataPartition(dataset[[target_var]], p = 0.7, list = FALSE)
data_train <- dataset[trainIndex,]
data_test <- dataset[-trainIndex,]

summary(data_train)
summary(data_test)

write.csv(data_train, file = "train.csv")
write.csv(data_test, file = "test.csv")

data_train <- read.csv("train.csv")
data_train=data_train %>% select(-c(1))

data_test=read.csv("test.csv")
data_test=data_test %>% select(-c(1))

####
###log model
####
target_var <- 'churned'
model_form <- churned ~ .
model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members
model_type <- "glm"
positive_class <- "Yes"
negative_class <- "No"



trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, 
                          summaryFunction = twoClassSummary, sampling = 'smote')

glm=train(as.formula(model_form), data = data_train, method = model_type, family = binomial, 
          trControl = trControl, metric='ROC')
glm

#fit confusion matrix
confusionMatrix(glm)

glm$finalModel

#test predictions
glm_test_class <- glm %>%  predict(type ='raw', newdata = data_test) 
glm_test_probs <- glm %>% predict(type = 'prob', newdata = data_test)

# test confusion matrix
confusionMatrix(glm_test_class, data_test[[target_var]], positive = positive_class)

# test ROC curve
roc(data_test[[target_var]], glm_test_probs[,positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE)

#####
###lda model
#####
target_var <- 'churned'
model_form <- churned ~ .
model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members
model_type <- "lda"
positive_class <- "Yes"
negative_class <- "No"

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, 
                          summaryFunction = twoClassSummary, sampling='smote')

lda=train(as.formula(model_form), data = data_train, method = model_type, family = binomial, 
          trControl = trControl, metric='ROC')
lda


#fit confusion matrix
confusionMatrix(lda)

#test predictions
lda_test_class <- lda %>%  predict(type ='raw', newdata = data_test) 
lda_test_probs <- lda %>% predict(type = 'prob', newdata = data_test)

# test confusion matrix
confusionMatrix(lda_test_class, data_test[[target_var]], positive = positive_class)

# test ROC curve
roc(data_test[[target_var]], lda_test_probs[,positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE)


#####
###qda model
#####
target_var <- 'churned'
model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members

#Location type gives 'rank deficiency in group Yes' error. 
#Looking into multicollinearity and other possibilities but so far haven't reached a conclusion as to why.
model_type <- "qda"
positive_class <- "Yes"
negative_class <- "No"

#ggpairs(data_train, columns = c(2, 3), aes(color = churned))

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, 
                          summaryFunction = twoClassSummary, sampling='smote')

qda=train(as.formula(model_form), data = data_train, method = model_type, family = binomial, 
          trControl = trControl, metric='ROC')
qda


#fit confusion matrix
confusionMatrix(qda)


#test predictions
qda_test_class <- qda %>%  predict(type ='raw', newdata = data_test) 
qda_test_probs <- qda %>% predict(type = 'prob', newdata = data_test)

# test confusion matrix
confusionMatrix(qda_test_class, data_test[[target_var]], positive = positive_class)

# test ROC curve
roc(data_test[[target_var]], qda_test_probs[,positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE)


####
###knn model
####
target_var <- 'churned'
model_form <- churned ~ .
model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members
model_type <- "knn"
positive_class <- "Yes"
negative_class <- "No"
threshold <- 0.5

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, 
                          summaryFunction = twoClassSummary, sampling = 'smote')

#tuneGrid <- expand.grid(k = c(20,30,50,70,100,150,200)) 
tuneGrid <- expand.grid(k = 150) 
knn_fit <- train(as.formula(model_form), data = data_train, method = model_type, 
                 trControl = trControl, metric='ROC', tuneGrid = tuneGrid)

knn_fit

#fit confusion matrix
confusionMatrix(knn_fit)


# test predictions
knn_pred_class <- knn_fit %>%  predict(type ='raw', newdata = data_test) 
knn_pred_probs <- knn_fit %>% predict(type = 'prob', newdata = data_test)

# test confusion matrix
confusionMatrix(knn_pred_class, data_test[[target_var]], positive = positive_class)

# test ROC curve
roc(data_test[[target_var]], knn_pred_probs[,positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE)




##
###Pruned classification tree
##

num_cores <- detectCores()
registerDoParallel(num_cores)

target_var <- 'churned'
model_form <- churned ~ .
model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members
model_type <- "rpart"
positive_class <- "Yes"
negative_class <- "No"

#tGrid <- expand.grid(cp=c(0.0))

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, summaryFunction = twoClassSummary, sampling='smote')

tree_prune <- train(as.formula(model_form), data = data_train, method = model_type, trControl = trControl, metric = 'ROC', tuneLength = 10)

tree_prune

#summary(tree_prune)
plot(tree_prune)
rpart.plot(tree_prune$finalModel, type = 1, extra = 1, under = TRUE, cex=0.6)

#Fit confusion matrix
confusionMatrix(tree_prune)


#test performance
tree_prune_pred = tree_prune %>% predict(newdata = data_test, type = 'raw')
tree_prune_probs <- tree_prune %>% predict(newdata = data_test, type = 'prob')
confusionMatrix(tree_prune_pred, data_test[[target_var]], positive = positive_class)
roc(data_test[[target_var]], tree_prune_probs[, positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE, levels = c(negative_class, positive_class))



###################################
#Bagging and Boosting
###


target_var <- 'churned'
model_form <- churned ~ .
model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members
model_type <- "rf"
positive_class <- "Yes"
negative_class <- "No"


trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, summaryFunction = twoClassSummary, search = 'grid', sampling = 'smote')

tGrid <- expand.grid(mtry = c(1,3,5,7,9,11,20))
#tGrid <- expand.grid(mtry = 10)


rf <- train(model_form, data = data_train, method = model_type, trControl = trControl, metric = 'ROC', tuneGrid = tGrid)

rf
#plot(rf3)

#training performance
confusionMatrix(rf)
confusionMatrix(rf$pred$pred, rf$pred$obs)


#test performance
rf_pred_test = rf %>% predict(newdata = data_test, type = 'raw')
rf_probs_test <- rf %>% predict(newdata = data_test, type = 'prob')
confusionMatrix(rf_pred_test, data_test[[target_var]], positive = positive_class)
roc(data_test[[target_var]], rf_probs_test[, positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE, levels = c(negative_class, positive_class))



#model_form <- churned ~ Business_Code+Location_Type+Major_Industry_Category_Name+total_products+
#  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+Public_Private_Indicator+
#  Number_of_Family_Members+BEMFAB__Marketability_+Legal_Status_Code+Manufacturing_Indicator

#remove legal status code
#model_form <- churned ~ Business_Code+Location_Type+total_products+
 # total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+Public_Private_Indicator+
  #Number_of_Family_Members

model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members

#model_form = churned ~ .

model_type <- 'xgbTree'

trControl <- trainControl(method = 'cv', savePredictions = TRUE, classProbs = TRUE, summaryFunction = twoClassSummary, sampling='smote')


xgb <- train(as.formula(model_form), data = data_train, method = model_type, trControl = trControl, metric = 'ROC')

xgb
summary(xgb)
xgb$finalModel

plot(varImp(xgb))

#Training CF
confusionMatrix(xgb)

?predict

#test test
xgb_pred <- xgb %>% predict(newdata = data_test, type = 'raw', threshold=0.3)
xgb_probs <- xgb %>% predict(newdata = data_test, type = 'prob')
confusionMatrix(xgb_pred, data_test[[target_var]], positive = positive_class)
roc(data_test[[target_var]], xgb_probs[, positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE, levels = c(negative_class, positive_class))


# then, let's convert this to binary predictions with a default threshold value of 0.5
threshold <- 0.4
xgb_pred2 <- factor(ifelse(xgb_probs[, positive_class] > threshold, positive_class, negative_class) , levels = c(positive_class, negative_class))

confusionMatrix(xgb_pred2, data_test[[target_var]], positive = positive_class)


#############################
#SVM
####

#Linear kernal

num_cores <- detectCores()
registerDoParallel(num_cores)

target_var <- 'churned'
model_form <- churned ~ .
positive_class <- "Yes"
negative_class <- "No"

model_type <- "svmLinear"

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, summaryFunction = twoClassSummary, search = 'grid', sampling = 'smote')


tGrid <- expand.grid(C = c(0.001,0.01,0.1,0.5,1,5,10,100))
#tGrid <- expand.grid(C = c(3,4,5,6,8,500))

svm2 <- train(as.formula(model_form), data = data_train, method = model_type, trControl = trControl, tuneGrid = tGrid, metric = 'ROC')
svm2

# textual summary of the outcome
svm2$finalModel
confusionMatrix(svm2)

# test CF
svm2_pred <- svm2 %>% predict(newdata = data_test, type = 'raw')
confusionMatrix(svm2_pred, data_test[[target_var]], positive = positive_class)

#test ROC
svm2_probs <- svm2 %>% predict(newdata = data_test, type = 'prob')
roc(data_test[[target_var]], svm2_probs[, positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE, levels = c(negative_class, positive_class))

##
#Poly Corn
##

model_form <- churned ~ Business_Code+total_products+
  total_transactions+total_accounts+total_revenue+total_usage+Employee_Count_Total+
  Number_of_Family_Members+Location_Type+Public_Private_Indicator+Major_Industry_Category_Name

model_type = 'svmPoly'

modelLookup(model_type)


tLength <- 27 

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, summaryFunction = twoClassSummary, search = 'random', sampling='smote')

svmp <- train(as.formula(model_form), data = data_train, method = model_type, trControl = trControl, tuneLength = tLength, metric = 'ROC')
svmp

summary(data_train)


svmp$finalModel
confusionMatrix(svmp)

#test CF
svmp_pred <- svmp %>% predict(newdata = data_test, type = 'raw')
confusionMatrix(svmp_pred, data_test[[target_var]], positive = positive_class)

#test ROC
svmp_probs <- svmp %>% predict(newdata = data_test, type = 'prob')
roc(data_test[[target_var]], svmp_probs[, positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE, levels = c(negative_class, positive_class))


##
#Radial Kernal
##

model_type <- "svmRadial"

trControl <- trainControl(method = 'cv', number = 10, savePredictions = TRUE, classProbs = TRUE, summaryFunction = twoClassSummary, search = 'grid', sampling='smote')

tGrid <- expand.grid(C = c(0.1, 1,5,10, 20), sigma = c(1/nrow(data_train), 0.01, 0.1, 0.5, 1))
#tGrid <- expand.grid(C = c(1, 3,5,7), sigma = c(1/nrow(data_train), 0.005, 0.01, 0.015, 0.03))
#tGrid <- expand.grid(C = c(5), sigma = c(1/nrow(data_train), 0.015, 0.03,0.07))

radial <- train(as.formula(model_form), data = data_train, method = model_type, trControl = trControl, tuneGrid = tGrid, metric = 'ROC')
plot(radial)
radial

radial$finalModel
confusionMatrix(radial)

#test CF
radial_pred <- radial %>% predict(newdata = data_test, type = 'raw')
confusionMatrix(radial_pred, data_test[[target_var]], positive = positive_class)

#test ROC
radial_probs <- radial %>% predict(newdata = data_test, type = 'prob')
roc(data_test[[target_var]], radial_probs[, positive_class], plot = TRUE, print.auc = TRUE, legacy.axes = TRUE, levels = c(negative_class, positive_class))


stopImplicitCluster()




