library(randomForest)
library(caret)
library(e1071)
library(ROCR)
library(ggplot2)

# Reading the dataset to an object
data1 <- read.table("adult.data.txt",sep = ",",stringsAsFactors = TRUE)
View(data1)  

#summary of the data
summary(data1)
str(data1)

# Renaming the column names of the dataset
colnames(data1) <- c("age","workclass","fnlwgt","education","education_num",
                 "marital_status","occupation","relationship","race","sex","capital_gain",
                 "capital_loss","hours_per_week","native_country","Income_Category")
?ggplot
#Exploratory Data Analysis
# Test-1 To find the which group of people earn more
ggplot(data1,aes(x= age,color = Income_Category,
                 fill = Income_Category))  + geom_density(alpha = 0.5) +
  labs(x = "age",y="Density",title = "Income Vs age",substitle = "Density plot")

# Test - 2 compare Income Vs working calss
ggplot(data1,aes(x= workclass,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.5) + coord_flip()
  labs(x = "workclass",y="Income_Category",title = "Income Vs working class",substitle = "Stacked Bar plot")

# Test -3 compare Income Vs Education
  ggplot(data1,aes(x= education,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.9) + coord_flip()
  labs(x = "education",y="Income_Category",title = "Income Vs Education",substitle = "Stacked Bar plot")

# Test - 4 compare Income Vs Race  
  
  ggplot(data1,aes(x= race,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.9) + coord_flip()
  labs(x = "race",y="Income_Category",title = "Income Vs Race",substitle = "Stacked Bar plot")
  
# Test - 5 compare Income Vs Occupation
  ggplot(data1,aes(x= occupation,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.9) + coord_flip()
  labs(x = "occupation",y="Income_Category",title = "Income Vs occupation",substitle = "Stacked Bar plot")

# Test - 6 compare Income Vs hours_per_week  
  ggplot(data1,aes(y= hours_per_week,x = Income_Category,fill = Income_Category))  + geom_boxplot(alpha = 0.5,outlier.shape = NA) 
  labs(x = "hours_per_week",y="Income_Category",title = "Income Vs hours_per_week",substitle = "Box and whisker plot")
  
  
  
#splitting the dataset into train and test dataset
trainindex<-sample(1:(nrow(data1)-1),0.6*nrow(data1))
traindata<-data1[trainindex,]
testdata<-data1[-trainindex,]
str(traindata)

str(data1)

# Changing the feature into numeric features
data1$education_num <- as.numeric(data1$education_num)
data1[11:13] <- lapply(data1[11:13],as.numeric)

summary(traindata)

View(traindata)

# Formula Assiging
formula_rf <- "Income_Category ~ ."

formula_ran <- as.formula(formula_rf)
traindata <- as.data.frame(traindata)

## Model 1 without normalisation factor 
# No features are eliminated, all the features are used for building model
rf.model <- randomForest(formula_ran, data = traindata, importance = T, mtry = 3,
                         nodesize = 5, ntrees = 500)

#View the model result 
# OOB rate is around 18%
print(rf.model)

# Plotting the variable importance
importance(rf.model,sort = TRUE, type = 1)
varImpPlot(rf.model, sort = TRUE,n.var = 14, type = 1)

test.feature.vars <- testdata[,-15]
test.class.vars <- testdata[,15]

# Confusion matrix gives us 95% accuracy
rf.predictions <- predict(rf.model, test.feature.vars, type = "prob")
confusionMatrix(data = rf.predictions,reference = test.class.vars)

# Tunning of model 1
# Hyperparameter optimisation for model 1
nodesize_mod = c(2,3,4)
ntrees_mod = c(200,300,400)

tune_rfmodel <- tune.randomForest(formula_ran, data = traindata, importance = T, mtry = 3,
                                  nodesize = nodesize_mod, ntrees = ntrees_mod) 

print(tune_rfmodel)
rf_bestmodel <- tune_rfmodel$best.model
rf.predictions.best <- predict(rf_bestmodel, test.feature.vars,type = "class")
confusionMatrix(data = rf.predictions.best, reference = test.class.vars)

# Plotting the ROC curve and find the AUC
pred <- prediction(as.numeric(rf.prediction),as.numeric(testdata$Income_Category))

perf <- performance(pred , "tpr", "fpr")
plot(perf,col="black",lty=1, lwd=2,
     main="title.text", cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
abline(0,1, col="red")
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc,2)
legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")


###### Logistic Regression ######

logi_model <- glm(Income_Category ~ age + workclass + fnlwgt + education + education_num
                 + marital_status + occupation + relationship + race + sex + capital_gain
                  + capital_loss + hours_per_week + native_country,family = binomial,data = traindata)
summary(logi_model)


traindata["Income_binary"] <- NA
traindata$Income_binary <- ifelse(traindata$Income_Category == " <=50K",1,0)
testdata["Income_binary"] <- NA
testdata$Income_binary <- ifelse(testdata$Income_Category == " <=50K",1,0)
traindata["native_country_binned"] <-NA
traindata$native_country_binned <- ifelse(traindata$native_country ==" United-States","US","Non-US")
testdata["native_country_binned"] <-NA
testdata$native_country_binned <- ifelse(testdata$native_country ==" United-States","US","Non-US")
View(traindata)

#model with Binned native country
logi_model1 <- glm(Income_binary ~ age+workclass+fnlwgt+education_num+marital_status+occupation+relationship+race+sex+capital_gain+
                    capital_loss+hours_per_week+native_country_binned,family = binomial,data = traindata)
summary(logi_model1)

# model without capital loss and capital gain
logi_model2 <- glm(Income_binary ~ age+workclass+fnlwgt+education_num+marital_status+occupation+sex+
                     +hours_per_week+native_country_binned,family = binomial(link = "logit"),data = traindata)
summary(logi_model2)

# Anova Test 
anova(logi_model1,logi_model2)

#Prediction and Confusion Matrix
testdata$logi_model1.yhat <- predict(logi_model1,testdata,type = "response")

model <- table(testdata$Income_binary,round(testdata$logi_model1.yhat,0))
confusionMatrix(model)


# PLotting of ROC and Finding the AUC curve
pred1 <- prediction(as.numeric(testdata$logi_model1.yhat),as.numeric(testdata$Income_binary))

perf1 <- performance(pred1 , "tpr", "fpr")
plot(perf1,col="black",lty=1, lwd=2,
     main="Logistic Regrssion", cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
abline(0,1, col="red")
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc,2)
legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")
