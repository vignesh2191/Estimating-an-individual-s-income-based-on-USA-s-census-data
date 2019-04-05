getwd()
setwd("C:\\Users\\Admin\\Desktop\\R\\Project")
library(randomForest)
library(caret)
library(e1071)
library(ROCR)
library(ggplot2)
data1 <- read.table("adult.data.txt",sep = ",",stringsAsFactors = TRUE)
View(data1)  
summary(data1)
str(data1)
colnames(data1) <- c("age","workclass","fnlwgt","education","education_num",
                     "marital_status","occupation","relationship","race","sex","capital_gain",
                     "capital_loss","hours_per_week","native_country","Income_Category")

ggplot(data1,aes(x= age,color = Income_Category,fill = Income_Category))  + geom_density(alpha = 0.5) +
  labs(x = "age",y="Density",title = "Income Vs age",substitle = "Density plot")

ggplot(data1,aes(x= workclass,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.5) + coord_flip()
labs(x = "workclass",y="Income_Category",title = "Income Vs working class",substitle = "Stacked Bar plot")

ggplot(data1,aes(x= education,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.9) + coord_flip()
labs(x = "education",y="Income_Category",title = "Income Vs Education",substitle = "Stacked Bar plot")

ggplot(data1,aes(x= race,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.9) + coord_flip()
labs(x = "race",y="Income_Category",title = "Income Vs Race",substitle = "Stacked Bar plot")

ggplot(data1,aes(x= occupation,color = Income_Category,fill = Income_Category))  + geom_bar(alpha = 0.9) + coord_flip()
labs(x = "occupation",y="Income_Category",title = "Income Vs occupation",substitle = "Stacked Bar plot")

ggplot(data1,aes(y= hours_per_week,x = Income_Category,fill = Income_Category))  + geom_boxplot(alpha = 0.5,outlier.shape = NA) 
labs(x = "hours_per_week",y="Income_Category",title = "Income Vs hours_per_week",substitle = "Box and whisker plot")

trainindex<-sample(1:(nrow(data1)-1),0.6*nrow(data1))
traindata<-data1[trainindex,]
testdata<-data1[-trainindex,]
str(traindata)
str(data1)
data1$education_num <- as.numeric(data1$education_num)
data1[11:13] <- lapply(data1[11:13],as.numeric)
summary(traindata)
View(traindata)
formula_rf <- "Income_Category ~ ."
formula_ran <- as.formula(formula_rf)
traindata <- as.data.frame(traindata)
rf.model <- randomForest(formula_ran, data = traindata, importance = T, mtry = 3,
                         nodesize = 5, ntrees = 500)
print(rf.model)
importance(rf.model,sort = TRUE, type = 1)
varImpPlot(rf.model, sort = TRUE,n.var = 14, type = 1)
test.feature.vars <- testdata[,-15]
test.class.vars <- testdata[,15]
rf.predictions <- predict(rf.model, test.feature.vars, type = "prob")
confusionMatrix(data = rf.predictions,reference = test.class.vars)
nodesize_mod = c(2,3,4)
ntrees_mod = c(200,300,400)
tune_rfmodel <- tune.randomForest(formula_ran, data = traindata, importance = T, mtry = 3,
                                  nodesize = nodesize_mod, ntrees = ntrees_mod) 
print(tune_rfmodel)
rf_bestmodel <- tune_rfmodel$best.model
rf.predictions.best <- predict(rf_bestmodel, test.feature.vars,type = "class")
confusionMatrix(data = rf.predictions.best, reference = test.class.vars)
pred <- prediction(as.numeric(rf.prediction),as.numeric(testdata$Income_Category))
perf <- performance(pred , "tpr", "fpr")
plot(perf,col="black",lty=1, lwd=2,
     main="title.text", cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
abline(0,1, col="red")
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc,2)
legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")
